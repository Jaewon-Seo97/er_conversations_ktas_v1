import os, sys
import warnings
warnings.filterwarnings(action='ignore')
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import json
import numpy as np
import pandas as pd
import datetime

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

import _utils.utils_binary as utils
import _utils.models as ai_model
import _utils.ml_train as ml_model
import _utils.eda as _aug_eda
import _utils._set_class_weights as _set_cls

import argparse
# import classification_nn as cls_nn    ###########################

def parse_args(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', help = 'Machine learning List "LR,SVM,XGB,MLP,RF,ET" and Neural network List "tf_MLP, BiLSTM"', default='RF')
    parser.add_argument('--fold_k', default = 0, type = int)
    parser.add_argument('--gpus', default = '0')
    parser.add_argument('--using_eda', action= 'store_true')
    parser.add_argument('--using_translate', action= 'store_true')
    parser.add_argument('--n_aug', default = 1, type = int)
    parser.add_argument('--trans_lang', default = 'en', type = str)

    return parser.parse_args(args)


class _get_prepdata:
    def __init__(self, configs, df_all, using_eda, n_aug, using_translate, trans_lang, fold_k):
        self.acceptList = configs["acceptList"]
        self.stop_words = configs["Stop_word"]
        self.df_all = df_all
        self.configs = configs
        self.AugEDA = using_eda
        self.n_aug = n_aug
        self.AugTrans = using_translate
        self.trans_lang = trans_lang
        self.fold_k = fold_k

    def _data_split(self):
        df_all = self.df_all
        k = self.fold_k
        skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=self.configs["random_seed"])
        train_index = [*skf.split(df_all['context'], df_all['class'].astype(int))][k][0]
        val_index = [*skf.split(df_all['context'], df_all['class'].astype(int))][k][1]
        train_x, x_test = df_all['context'][train_index], df_all['context'][val_index]
        train_y, y_test = df_all['class'][train_index].astype(int), df_all['class'][val_index].astype(int)
        train_f, f_test = df_all['file_name'][train_index], df_all['file_name'][val_index]

        df_train = df_all.loc[list(train_index)].reset_index()
        skf=StratifiedKFold(n_splits=9, shuffle=True, random_state=self.configs["random_seed"])
        train_index = [*skf.split(train_x, train_y.astype(int))][0][0]
        val_index = [*skf.split(train_x, train_y.astype(int))][0][1]
        x_train, x_val = df_train['context'][train_index], df_train['context'][val_index]
        y_train, y_val = df_train['class'][train_index].astype(int), df_train['class'][val_index].astype(int)
        f_train, f_val = df_train['file_name'][train_index], df_train['file_name'][val_index]

        print(f"N data\nTrain:{x_train.shape}, Validation:{x_val.shape}, Test:{x_test.shape}")
        print(f"Label shape: {y_train.shape}")

        if self.AugEDA or self.AugTrans:
            x_train, y_train, f_train = self.aug_eda(x_train, y_train, f_train, self.n_aug)
        else:
            pass

        return x_train, y_train, f_train, x_val, y_val, f_val, x_test, y_test, f_test
    
    def aug_eda(self, x_train, y_train, f_train, n_aug=1):
        new_aug = x_train.copy()
        new_lab = y_train.copy()
        new_name = f_train.copy()
        new_index = ['0' for n in range(n_aug)]

        if self.AugTrans == True and self.AugEDA == True:
            new_index.append('0')
            n_aug+=1
        else:
            pass

        from textaugment import Translate
        t = Translate(src = "ko", to=self.trans_lang)
        
        print(f'origin train: {new_aug.shape}')
        auged = 0
        for a, lab in enumerate(y_train.to_list()):
            if lab == 1:
                # augmented_sentences = aug_eda.EDA(sentence=x_train.to_list()[a], alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.5, p_rd=0.5, num_aug=n_aug)
                if self.AugEDA == True and self.AugTrans == False:
                    augmented_sentences = _aug_eda.EDA(sentence=x_train.to_list()[a], alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.7, p_rd=0.7, num_aug=n_aug)

                elif self.AugTrans == True and self.AugEDA == False:
                    augmented_sentences = [t.augment(x_train.to_list()[a])]
                    augmented_sentences.append(x_train.to_list()[a])

                elif self.AugTrans == True and self.AugEDA == True:
                    
                    augmented_sentences = [t.augment(x_train.to_list()[a])]
                    auged_eda = _aug_eda.EDA(sentence=x_train.to_list()[a], alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.7, p_rd=0.7, num_aug=n_aug-1) 
                    augmented_sentences+=auged_eda

                    
                else:
                    print("Check augmentation type: AugEDA= {self.AugEDA}, AugTrans ={self.AugTrans}")

                # print(len(new_aug), len(new_index), len(augmented_sentences))
                new_aug = pd.concat([new_aug, pd.Series(augmented_sentences[:-1], index=new_index)])
                new_lab = pd.concat([new_lab, pd.Series([1 for n in range(n_aug)], index=new_index)])
                new_name = pd.concat([new_name, pd.Series([f"{f_train.to_list()[a]}_aug{'0'}" for n in range(n_aug)], index=new_index)])
                auged+=1
        print('\nUsing easy data augmentation')
        print(f'before \ntotal: {len(x_train)}, ktas3: {len(x_train)-auged}, ktas4,5: {auged}')
        print(f'after  \ntotal: {len(new_aug)}, ktas3: {len([n for n in new_lab.to_list() if n == 0])}, ktas4,5: {len([n for n in new_lab.to_list() if n == 1])}')

        print(new_aug.to_list()[-1])
        return new_aug, new_lab, new_name
    
    ### MLP
    def _get_corpus(self, x_train, y_train, f_train):
        X_train_token = utils.pos_select(x_train, y_train, f_train, self.stop_words, self.acceptList)
        corp_Xtrain = utils._making_token(X_train_token)
        Y_train = tf.keras.utils.to_categorical(y_train)

        return corp_Xtrain, Y_train
    
    def _get_tfidfVec(self, parm):
        
        x_train, y_train, f_train, x_val, y_val, f_val, x_test, y_test, f_test = self._data_split()
        
        corp_Xtrain, Y_train = self._get_corpus(x_train, y_train, f_train)
        corp_Xval, Y_val = self._get_corpus(x_val, y_val, f_val)
        corp_Xtest, Y_test = self._get_corpus(x_test, y_test, f_test)


        vectorizer = TfidfVectorizer(max_features=parm["max_features"],ngram_range=(1,parm["max_ngram"]), min_df=1)
        
        X_train = vectorizer.fit_transform(corp_Xtrain).toarray()
        X_val = vectorizer.transform(corp_Xval).toarray()
        X_test = vectorizer.transform(corp_Xtest).toarray()
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test, f_test

    ### BiLSTM    
    def _get_trainVocab(self, X_train_token, threshold = 2):
        from tensorflow.keras.preprocessing.text import Tokenizer    
        rare_cnt, total_freq, rare_freq = 0, 0, 0

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_token['morphs'])
        total_cnt = len(tokenizer.word_index)

        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value

            if(value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        vocab_size = total_cnt - rare_cnt + 2

        return vocab_size

    def _pad_seq(self, X_train, y_train, maxlen):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X_train= pad_sequences(X_train, maxlen=maxlen)
        Y_train = tf.keras.utils.to_categorical(y_train)
        return X_train, Y_train

    def _get_LSTMtoken(self, parm, _get_vocab):

        from tensorflow.keras.preprocessing.text import Tokenizer


        x_train, y_train, f_train, x_val, y_val, f_val, x_test, y_test, f_test = self._data_split()

        X_train_token = utils.pos_select(x_train, y_train, f_train, self.stop_words, self.acceptList)
        X_val_token = utils.pos_select(x_val, y_val, f_val, self.stop_words, self.acceptList)
        X_test_token = utils.pos_select(x_test, y_test, f_test, self.stop_words, self.acceptList) 
        
        if _get_vocab == True:
            vocab_size = self._get_trainVocab(X_train_token, parm["token_th"])
        else:
            vocab_size = parm["vocab_size"]
        
        # tokenization
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train_token['morphs'])
        X_train = tokenizer.texts_to_sequences(X_train_token['morphs'])
        X_val = tokenizer.texts_to_sequences(X_val_token['morphs'])
        X_test = tokenizer.texts_to_sequences(X_test_token['morphs'])

        # padding
        maxlen = (max(len(review) for review in X_train)//100)*100

        X_train, Y_train = self._pad_seq(X_train, y_train, maxlen)
        X_val, Y_val = self._pad_seq(X_val, y_val, maxlen)
        X_test, Y_test = self._pad_seq(X_test, y_test, maxlen)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, f_test, vocab_size    
    

def main(args=None):

    if args is None:
        args = sys.argv[1:]
    
    args = parse_args(args)

    ######################
    # Setting parameters
    ######################
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    date = f"{str(datetime.date.today()).split('-')[-2]}{str(datetime.date.today()).split('-')[-1]}"
    s = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(s)
    ## Setting class (config)
    class_dict = {'KTAS3':0,
                'KTAS4,5':1}
    n_class = len(class_dict)
    ## Loading configuration file
    with open('config.json', "r", encoding="UTF-8") as f:
        configs = json.load(f)

    ## Setting default path
    path_dataDF = os.path.join(os.path.abspath('..'), '..', '00_NIA_Emergency', 'data')
    path_saveDF = os.path.join(os.path.abspath('.'), date, 'model', f'fold{args.fold_k}', f'fold{args.model_type}_aug{args.using_eda}{args.n_aug}_augTrans{args.using_translate}{args.trans_lang}_{configs["model_name"]}')
    path_result = os.path.join(os.path.abspath('.'), date, 'Result', f'fold{args.fold_k}', f'{args.model_type}_aug{args.using_eda}{args.n_aug}_augTrans{args.using_translate}{args.trans_lang}_{configs["model_name"]}')
    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_saveDF, exist_ok=True)



    ######################
    # Loading data
    ######################
    print('Loading data...')
    df_all = utils.preprocessing_data2morph(path_dataDF, class_dict)

    ######################
    # Setting model
    ######################
        


    ## Setting configuration
    random_seed = configs["random_seed"]
    epochs = configs["epochs"]
    batch = configs["batch"]
    model_type = args.model_type
    LR = configs["LR"]
    cls_prep = _get_prepdata(configs, df_all, args.using_eda, args.n_aug, args.using_translate, args.trans_lang, args.fold_k)
    if ('tf_MLP' in model_type) or ('BiLSTM' in model_type):
        print('Neural network using tensorflow')

        if model_type == 'tf_MLP':
            model_par = configs["mlp_parameter"]

            ## tf-idf vectorazation
            
            X_train, Y_train, X_val, Y_val, X_test, Y_test, F_test = cls_prep._get_tfidfVec(model_par)

            ## Loading a model structure
            model = ai_model.model_mlp(n_input_dim=model_par["max_features"], n_unit=model_par["n_units"], n_class=n_class)
        
        elif model_type == 'BiLSTM':
            model_par = configs["BiLSTM_parameter"]

            if model_par["vocab_size"] == "None":
                _get_vocab = True
            else:
                _get_vocab = False
                
            X_train, Y_train, X_val, Y_val, X_test, Y_test, F_test, vocab_size = cls_prep._get_LSTMtoken(model_par, _get_vocab)
            
            ## Loading a model structure
            model = ai_model.model_BiLSTM(n_input_dim=vocab_size, emb_dim = model_par["embedding_dim"], hidden_units=model_par["hidden_units"], n_class=n_class)

        ######################
        # Training model
        ######################


        ## the model compile

        dic_class_weights, ls_class_weights = _set_cls.generate_class_weights(Y_train)
        print(f'class_weights: {dic_class_weights} | n class: {len(dic_class_weights)} list class weight: {ls_class_weights}')
        # weighted_loss = _set_cls.weighted_categorical_crossentropy(ls_class_weights)
        weighted_loss = _set_cls.weighted_categorical_crossentropy([0.69568823, 1.77754237])
        model.compile(loss=weighted_loss, optimizer=Adam(lr=LR), metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])     ######################

        ## Setting callbacks
        checkpointer = ModelCheckpoint(filepath=os.path.join(path_saveDF,'cp_{epoch:02d}.h5'), save_weights_only=True,
                                        verbose=1, monitor='val_loss', period=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                    patience=10, min_lr=0, min_delta=1e-10, verbose=1)
        
        EarlyStopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
        callbacks_list = [checkpointer, reduce_lr, EarlyStopper]
        
        ## training the model
        with tf.device("/device:GPU:0"):
            training_model = model.fit(x= X_train, y= Y_train, 
                                    batch_size=batch, 
                                    validation_data = (X_val, Y_val),
                                    # validation_split = 1/9, 
                                    epochs= epochs, 
                                    verbose=1, 
                                    shuffle=True, 
                                    callbacks=callbacks_list)
        
        ## Saving a last weight
        model.save_weights(os.path.join(path_saveDF,f'last_model.h5'))

        ######################
        # Result prediction
        ######################
        print('\n'+'#'*(len('Accuracy Training data: ')+5))
        print('Results')
        predicted_train_tfidf = model.predict(X_train,verbose=0)
        predicted_test_tfidf = model.predict(X_test,verbose=0)

        accuracy_train_tfidf = accuracy_score(Y_train.argmax(axis=1), predicted_train_tfidf.argmax(axis=1))
        accuracy_test_tfidf = accuracy_score(Y_test.argmax(axis=1), predicted_test_tfidf.argmax(axis=1))
        print('Accuracy Training data: {:.1%}'.format(accuracy_train_tfidf))
        print('Accuracy Test data: {:.1%}'.format(accuracy_test_tfidf))
        print('#'*(len('Accuracy Training data: ')+5))
        cof_mat = confusion_matrix(Y_test.argmax(axis=1), predicted_test_tfidf.argmax(axis=1))
        cof_mat = pd.DataFrame(cof_mat, columns=list(class_dict.keys()), index = list(class_dict.keys()))
        cof_mat.to_excel(os.path.join(path_result, f'ConfusionMatrix.xlsx'))

        ## Saving each result        
        resultList = pd.DataFrame(columns=['File_name', 'GT', 'Prediction', 'Result', 'disease', 'disease_type'])
        i=0
        for gt, yp in zip(Y_test.argmax(axis=1), predicted_test_tfidf.argmax(axis=1)):
            gt_name = list(class_dict.keys())[gt]
            pr_name = list(class_dict.keys())[yp]
            if gt != yp:
                match = 'F'
            else:
                match = 'T'
            sub_df = pd.DataFrame({'File_name': f'{np.array(F_test)[i]}', 
                                'GT': gt_name, 
                                'Prediction':pr_name,
                                'Result': match,
                                'disease':df_all[df_all['file_name']==np.array(F_test)[i]]['disease'].values[0], 
                                'disease_type': df_all[df_all['file_name']==np.array(F_test)[i]]['disease_type'].values[0]
                                }, index=[i])
            resultList = pd.concat([resultList, sub_df])
            i+=1
        resultList.to_excel(os.path.join(path_result,f'ResultList.xlsx'))


        ####################################
        # ROC curves
        ####################################

        nameList = [model_type]
        all_fpr = dict()
        all_tpr = dict()
        each_aucs = dict()
        # for model, clf_name in zip([model_lr], nameList):
        for model, clf_name in zip([model], nameList):

            # proba, probList, resultList = utils.ml_predict(model, X_train, Y_train, X_test, Y_test, F_test, clf_name, class_dict, path_result, date='0113')
            utils.ml_predict(predicted_test_tfidf, accuracy_test_tfidf, Y_test, F_test, clf_name, class_dict, path_result, date='0113')
            mic_fpr, mic_tpr, mic_roc_auc, mac_fpr, mac_tpr, mac_roc_auc= utils.drawing_ROCcurve(Y_test, predicted_test_tfidf, savepath= path_result, clf_name = clf_name, name_dict = class_dict)
            all_fpr[f"{clf_name}_fpr"] = mac_fpr
            all_tpr[f"{clf_name}_tpr"] = mac_tpr
            each_aucs[f"{clf_name}_auc"] = mac_roc_auc


        # Saving ROC curves: macro average
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.rc('font', family='NanumGothic')
        for clf_name in nameList:
            each_auc = each_aucs[f'{clf_name}_auc']
            plt.plot(
                all_fpr[f"{clf_name}_fpr"],
                all_tpr[f"{clf_name}_tpr"],
                label=f"{clf_name} (auc = {each_auc:0.2f})",
                linestyle="-",
                linewidth=4,
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.rc('font', family='NanumGothic')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.title("ALL ROC curves", fontsize=20)
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(f'{path_result}/avg_ROC_{model_type}.png')    


        print('Done!')    

        # ## Saving meta graph
        # import tensorflow.compat.v1 as tf1
        # tf1.disable_v2_behavior()
        # tf1.compat.v1.train.export_meta_graph(filename=os.path.join(path_saveDF,'checkpoint','metagraph.meta'),
        #                                     collection_list=["input_tensor", "output_tensor"])
        
        print('All process done!')
        s = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(s)


    else:
        # train machine learning model (no neural network except for MLP)
        
        model_nameList = [name for name in args.model_type.split(',')]
        print(f'ML training: {model_nameList}')   
        x_train, y_train, f_train, x_val, y_val, f_val, x_test, y_test, f_test = cls_prep._data_split()
        print('Data split done!\n')
        corp_Xtrain, Y_train = cls_prep._get_corpus(pd.concat([x_train, x_val]), pd.concat([y_train,y_val]), pd.concat([f_train, f_val]))
        Y_train = np.array(pd.concat([y_train,y_val]), dtype = np.uint64)

        corp_Xtest, Y_test = cls_prep._get_corpus(x_test, y_test, f_test)
        Y_test = np.array(y_test, dtype = np.uint64)

        print(f'Input shape: {Y_test}, ')
        print('Data preprocessing (making corpus) done!\n')
        model_par = configs["mlp_parameter"]
        tfml = ml_model._get_tfidf_ml(df_all, corp_Xtrain, Y_train, savepath = path_result, name_dict = class_dict, max_features=model_par["max_features"],ngram_range=(1,model_par["max_ngram"]), date = date)
        print('Setting model done!\n')
        tfml.train_MLmodel(model_nameList, corp_Xtest, Y_test, f_test, predict=True)
        
        print(f'Train {model_nameList} Done!')
if __name__ == '__main__':
    main()
    
