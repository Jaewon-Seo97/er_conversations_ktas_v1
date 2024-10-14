from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten

from transformers import *

def model_mlp(n_input_dim, n_unit, n_class):
    
    unitList = [n_input_dim] + n_unit + [n_class]
    
    inputs = Input((n_input_dim,))
    
    for nu, nunit in enumerate(unitList[1:-1]):
        x = Dense(unitList[nu+1], #################
                        input_dim=nunit,
                        kernel_initializer=tf.keras.initializers.HeNormal(),
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0.0001),
                        # bias_regularizer=regularizers.L1L2(l1=0, l2=0.001),
                        activation='relu')(inputs)
        
    outputs = Dense(n_class, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def model_cnn(n_input_dim, n_class, n_gram=2):
    
    inputs = Input((n_input_dim,1))
    x = Conv1D(filters=64, kernel_size=3, kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0.0001),activation='relu')(inputs)
    x = Conv1D(filters=64, kernel_size=3, kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0.0001),activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=n_gram)(x)
    x = Flatten()(x)
    print(f'Flatten: {x.shape}')
    x = Dense(100, activation='relu')(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    return model
    
def model_BiLSTM(n_input_dim, emb_dim, max_len, hidden_units, n_class):
    model = Sequential()
    model.add(Embedding(input_dim = n_input_dim,
                        output_dim = emb_dim,
                        input_length = max_len))
    model.add(Bidirectional(LSTM(hidden_units,
                                dropout=0.3))) # Bidirectional LSTM을 사용
    model.add(Dense(n_class, activation='softmax'))
    return model

class BERT_prep():
    def __init__(self, MAX_LEN):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt', do_lower_case=False)
        self.MAX_LEN = MAX_LEN


    def _get_vocab(self, context):
        from collections import Counter
        from konlpy.tag import Okt
        okt = Okt()
        okt.morphs(context, stem=True)

        
    def prep_data(self, df_set):

        input_ids = []
        attention_masks = []
        token_type_ids = []
        train_data_labels = []

        for train_sent, train_label in tqdm(zip(df_set["context"], df_set["class"]), total=len(df_set)):
            # print(f"train_sent, train_label\n{train_sent}, \n{train_label}")
            try:
                input_id, attention_mask, token_type_id = self.bert_tokenizer(train_sent)
                
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
                train_data_labels.append(train_label)

            except Exception as e:
                print(e)
                print(train_sent)
                pass
        
        train_movie_input_ids = np.array(input_ids, dtype=int)
        train_movie_attention_masks = np.array(attention_masks, dtype=int)
        train_movie_type_ids = np.array(token_type_ids, dtype=int)
        train_movie_inputs = (train_movie_input_ids, train_movie_attention_masks, train_movie_type_ids)

        train_data_labels = tf.keras.utils.to_categorical(np.asarray(train_data_labels, dtype=np.int32))

        return train_movie_inputs, train_data_labels

    def bert_tokenizer(self, sent):

        


        encoded_dict = self.tokenizer.encode_plus(
            text = sent,
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = self.MAX_LEN,           # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True   # Construct attn. masks.
            
        )
        print(f'origin sentence length: {len(sent)}')
        print(f'df_sentence: {sent}')
        input_id = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask'] # And its attention mask (simply differentiates padding from non-padding).
        token_type_id = encoded_dict['token_type_ids'] # differentiate two sentences
        
        print(f'token type id: {token_type_id}')
        print(f'input id: {input_id}')
        print(f'attention mask: {attention_mask}')

        return input_id, attention_mask, token_type_id

class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class, model_par, pretrained=False):
        super(TFBertClassifier, self).__init__()
        """
        default_configs = BertConfig( vocab_size = 30522,
                                 hidden_size = 768,
                                 num_hidden_layers = 12,
                                 num_attention_heads = 12,
                                 intermediate_size = 3072,
                                 hidden_act = 'gelu',
                                 hidden_dropout_prob = 0.1,
                                 attention_probs_dropout_prob = 0.1,
                                 max_position_embeddings = 512,
                                 type_vocab_size = 2,
                                 initializer_range = 0.02,
                                 layer_norm_eps = 1e-12,
                                 pad_token_id = 0,
                                 position_embedding_type = 'absolute',
                                 use_cache = True,
                                 classifier_dropout = None,
                                 **kwargs )
        """
        if pretrained:
            self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        else:
            configs = BertConfig( vocab_size = model_par['vocab_size'], 
                                 hidden_size = model_par['max_len'],
                                 num_hidden_layers = 12,
                                 num_attention_heads = model_par['max_len']//64,
                                 intermediate_size = model_par['max_len']*4, #######
                                 hidden_act = 'gelu',
                                 hidden_dropout_prob = 0.1,
                                 attention_probs_dropout_prob = 0.1,
                                 max_position_embeddings = 512,   ######
                                 type_vocab_size = 2,
                                 initializer_range = 0.02,
                                 layer_norm_eps = 1e-12,
                                 pad_token_id = 0,
                                 position_embedding_type = 'absolute',
                                 use_cache = True,
                                 classifier_dropout = None )
            self.bert = TFBertModel(configs)
            print(f'new_config: {configs}')
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), 
                                                name="classifier")
        
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=True):
        
        #outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(outputs.shape)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        # print(logits)
        # logits.summary()

        return logits
    
    

