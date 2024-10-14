import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten

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
        
    outputs = Dense(1, activation='sigmoid')(x)

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
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    return model
    
def model_BiLSTM(n_input_dim, emb_dim, hidden_units, n_class):
    model = Sequential()
    model.add(Embedding(input_dim = n_input_dim,
                        output_dim = emb_dim))
    model.add(Bidirectional(LSTM(hidden_units,
                                dropout=0.3))) # Bidirectional LSTM을 사용
    model.add(Dense(1, activation='sigmoid'))
    return model

# def model_BERT():
