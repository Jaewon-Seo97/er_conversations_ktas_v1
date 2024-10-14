import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import tensorflow as tf
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,roc_curve,auc, roc_auc_score, multilabel_confusion_matrix, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import joblib



class _get_tfidf_ml:
    def __init__(self, df_all, corp_Xtrain, y_train, savepath, name_dict, max_features=None,ngram_range=(1,2), fold= 0, date = datetime.date.today()):
        self.df_all = df_all
        self.vectorizer = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=1)
        self.vectorizer.fit_transform(corp_Xtrain)    
        self.corp_Xtrain = corp_Xtrain
        self.y_train = y_train
        self.savepath = savepath
        self.name_dict = name_dict
        self.fold = fold
        self.date = date

    def ml_class(self, clf, clf_name):
        print(f'Training {clf_name}')
        model= Pipeline([("vectorizer", self.vectorizer), ("classifier", clf)])
        start_time = datetime.datetime.now()
        model.fit(self.corp_Xtrain, self.y_train)
        end_time = datetime.datetime.now()

        training_time_tfidf = (end_time - start_time).total_seconds()
        print('Training time: {:.1f}s'.format(training_time_tfidf)) 

        return model

    def ml_predict(self, model, corp_Xtest, y_test, f_test, clf_name):

        test_y = tf.keras.utils.to_categorical(y_test)
        test_y = test_y[:,list(self.name_dict.values())]
        
        print(f'Prediction: {clf_name}')
        predicted_train_tfidf = model.predict(self.corp_Xtrain)
        accuracy_train_tfidf = accuracy_score(self.y_train, predicted_train_tfidf)
        print('Accuracy Training data: {:.1%}'.format(accuracy_train_tfidf))

        predicted_test_tfidf = model.predict(corp_Xtest)
        accuracy_test_tfidf = accuracy_score(y_test, predicted_test_tfidf)
        print('Accuracy Test data: {:.1%}'.format(accuracy_test_tfidf))

        proba = model.predict_proba(corp_Xtest)

        # Saving result values

        mcof_mat = multilabel_confusion_matrix(y_test, predicted_test_tfidf)
        f1_each = f1_score(y_test, predicted_test_tfidf, average=None)
        mcmList = pd.DataFrame(columns=['class', 'f1-score', 'AUC', 'ACC', 'TP', 'FP', 'TN', 'FN'])
        print(list(self.name_dict.keys()))
        print(len(mcof_mat))
        aucs = []
        for c, cm in enumerate(mcof_mat):

            name_cls = f'{list(self.name_dict.keys())[c]}'
            each_auc = roc_auc_score(test_y[:, c], proba[:, c], multi_class="ovr")
            each_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
            print(f'confusion matrix: {name_cls}')
            print(cm)
            print(f'AUC: {each_auc}')
            cat_cm = pd.DataFrame({'class': name_cls,
                                    'f1-score': f'{f1_each[c]:0.2f}',
                                    'AUC': f'{each_auc:0.2f}',
                                    'ACC': f'{each_acc:0.2f}',
                                    'TN': cm[0][0],
                                    'FP': cm[0][1],
                                    'FN': cm[1][0],
                                    'TP': cm[1][1]}, index=[name_cls])
            mcmList = pd.concat([mcmList, cat_cm], axis = 0)
            aucs.append(each_auc)
        mcmList.to_excel(os.path.join(self.savepath, f'{clf_name}_Multi-ConfusionMatrix_fold{self.fold}_{self.date}.xlsx'))
        cof_mat = confusion_matrix(y_test, predicted_test_tfidf)
        cof_mat = pd.DataFrame(cof_mat, columns=list(self.name_dict.keys()), index = list(self.name_dict.keys()))
        cof_mat = pd.concat([cof_mat, pd.DataFrame({'f1_score_macro':f1_score(y_test, predicted_test_tfidf, average='macro'),
                                                        'AUC_macro':f'{sum(aucs)/len(aucs):0.3f}',
                                                        'accurcy': f'{accuracy_test_tfidf:0.3f}',
                                                        }, index = ['Total'])])
        cof_mat.to_excel(os.path.join(self.savepath, f'{clf_name}_ConfusionMatrix_fold{self.fold}_{self.date}.xlsx'))
        
        return proba, accuracy_test_tfidf
        ###############################################################################################################################        
        # probclassList = [f'prob_{name}' for name in list(self.name_dict.values())]
        # probList = pd.DataFrame(proba, columns=probclassList)

        # resultList = pd.DataFrame(columns=['File_name', 'GT', 'Prediction', 'Result', 'disease', 'disease_type'])
        # fail_idx = []
        # i = 0
        # for gt, yp in zip(np.array(y_test), predicted_test_tfidf):
        #     gt_name = list(self.name_dict.keys())[gt]
        #     pr_name = list(self.name_dict.keys())[yp]
        #     if gt != yp:
        #         match = 'F'
        #         fail_idx.append(i)
        #     else:
        #         match = 'T'

        #     sub_df = pd.DataFrame({'File_name': f'{np.array(f_test)[i]}', 
        #                         'GT': gt_name, 
        #                         'Prediction':pr_name,
        #                         'Result': match,
        #                         'disease':self.df_all[self.df_all['file_name']==np.array(f_test)[i]]['disease'].values[0], 
        #                         'disease_type': self.df_all[self.df_all['file_name']==np.array(f_test)[i]]['disease_type'].values[0]
        #                         }, index=[i])
        #     resultList = pd.concat([resultList, sub_df])
                
        #     i+=1
        # resultList = pd.concat([resultList, probList], axis=1)
        # failList = resultList.loc[fail_idx]
        # failList.to_excel(os.path.join(self.savepath, f'{clf_name}_FailList_{self.date}.xlsx'))
        # resultList.to_excel(os.path.join(self.savepath, f'{clf_name}_ResultList_{self.date}.xlsx'))
        # return proba, probList, resultList
    
    def drawing_ROCcurve(self, y_test, test_result, savepath, clf_name, name_dict):
        test_y = tf.keras.utils.to_categorical(y_test)
        test_y = test_y[:,list(name_dict.values())]
        n_classes = test_y.shape[-1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_y[:, i], test_result[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), test_result.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        # First aggregate all false positive rates
        lw = 2
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(10,10))
        plt.rc('font', family='NanumBarunGothic')
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average (auc = {0:0.2f})".format(roc_auc["micro"]),
            color="maroon",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average (auc = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = ["b", "r", "g", "c", "m"]
        i = 0
        for nc, color in zip(list(name_dict.keys()), colors[:n_classes]):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="class {0} (auc = {1:0.2f})".format(nc, roc_auc[i]),
            )
            i+=1

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.rc('font', family='NanumBarunGothic')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.title("Receiver operating characteristic curves", fontsize=20)
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(f'{savepath}/ROCcurve_{clf_name}_{n_classes}class_{self.date}.png')
        
        return fpr["micro"], tpr["micro"], roc_auc["micro"], fpr["macro"], tpr["macro"], roc_auc["macro"]

    # def train_MLmodel(self, model_name, corp_Xtest, y_test, f_test, predict=True):
    def train_MLmodel(self, model_name):
        """
        Setting Machine learning model & Training
        model List
         - LR(Loginistic regression)
         - SVM(Support vector machine)
         - XGB(Extreme gradient boost)
        """
        # model_nameList= list(model_nameList)
        os.makedirs(os.path.join(self.savepath, 'Result'), exist_ok=True)
        os.makedirs(os.path.join(self.savepath, 'model'), exist_ok=True)

        model_dict = {'LR': LogisticRegression(), 
                    'SVM': svm.SVC(decision_function_shape='ovo', probability=True), 
                    'XGB': XGBClassifier(), 
                    'MLP': MLPClassifier(verbose=True), 
                    'RF':RandomForestClassifier(), 
                    'ET':ExtraTreesClassifier()}

        # all_fpr = dict()
        # all_tpr = dict()
        # each_aucs = dict()    

        # for n, model_name in enumerate(model_nameList):
        model = self.ml_class(model_dict[model_name], model_name)
        joblib.dump(model, os.path.join(os.path.join(self.savepath, 'model'), f'{model_name}_{self.date}.pkl'))
        return model
        ############################################################################################################
        # proba, probList, resultList  = self.ml_predict(model, corp_Xtest, y_test, f_test, model_name)
        # mic_fpr, mic_tpr, mic_roc_auc, mac_fpr, mac_tpr, mac_roc_auc= self.drawing_ROCcurve(y_test, proba, savepath= os.path.join(self.savepath, 'Result'), clf_name = model_name, name_dict = self.name_dict)
        # all_fpr[f"{model_name}_fpr"] = mac_fpr
        # all_tpr[f"{model_name}_tpr"] = mac_tpr
        # each_aucs[f"{model_name}_auc"] = mac_roc_auc

        # # Saving all ROC curves: macro average
        # plt.figure(figsize=(10,10))
        # plt.rc('font', family='NanumGothic')

        # for model_name in model_nameList:
        #     each_auc = each_aucs[f'{model_name}_auc']
        #     plt.plot(
        #         all_fpr[f"{model_name}_fpr"],
        #         all_tpr[f"{model_name}_tpr"],
        #         label=f"{model_name} (auc = {each_auc:0.2f})",
        #         linestyle="-",
        #         linewidth=4,
        #     )

        # plt.plot([0, 1], [0, 1], "k--", lw=2)
        # plt.rc('font', family='NanumGothic')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel("False Positive Rate", fontsize=20)
        # plt.ylabel("True Positive Rate", fontsize=20)
        # plt.title("ALL ROC curves", fontsize=20)
        # plt.legend(loc="lower right", fontsize=14)
        # plt.savefig(os.path.join(self.savepath, 'Result', f'ROCcurve_all_{self.date}.png'))

        return 0