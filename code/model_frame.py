import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Attention, Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam

import model_gat
import data_config as cf 

def our_gat_block(inputs):
    b, h = model_gat.our_gat_conv(cf.dense_size, attn_heads=4, activation='elu')(inputs)
    h = Dense(cf.dense_size, activation='elu')(h)
    h = LayerNormalization(epsilon=1e-3)(inputs[0] + h)
    h = Dropout(cf.dropout)(h)
    return b, h

def custom_loss(labels, y_pred):
    def exclude_minus1(x): 
        return 1+x*(1-x)/2

    def binary_loss(a, b):
        return -a*tf.math.log(b)-(1.0-a)*tf.math.log(1.0-b)

    loss = 0.0
    for idx in range(labels.shape[1]):
        loss += exclude_minus1(labels[:,idx])*binary_loss(labels[:,idx], y_pred[:,idx])
    loss = loss/float(y_pred.shape[1])
    return loss

def make_model(task_length):
    atom_input = Input(shape=(cf.atom_cutoff, cf.atom_length))
    bond_input = Input(shape=(cf.atom_cutoff, cf.atom_cutoff, 2*cf.bond_length))
    atom_graph = Input(shape=(cf.atom_cutoff, cf.atom_cutoff))

    atom = Dense(cf.dense_size)(atom_input)
    atom = LayerNormalization(epsilon=1e-3)(atom)
    atom = Dropout(cf.dropout)(atom)

    bond = Dense(int(cf.dense_size/8))(bond_input)
    bond = LayerNormalization(epsilon=1e-3)(bond)
    bond = Dropout(cf.dropout)(bond)

    bond, h = our_gat_block([atom, atom_graph, bond])
    bond, h = our_gat_block([h, atom_graph, bond])

    h = Attention(dropout=cf.dropout)([h,h]) 
    h = Dense(cf.dense_size, activation='elu')(h)
    h = LayerNormalization(epsilon=1e-3)(h)
    h = Dropout(cf.dropout)(h)

    H = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(h)
    H = Dense(cf.dense_size, activation='elu')(H)
    H = LayerNormalization(epsilon=1e-3)(H)
    H = Dropout(cf.dropout)(H)

    outs = []
    for i in range(task_length):
        i_out = Dense(cf.task_dense_size, activation='elu', name='i_out'+str(i))(H)
        outs.append(Dense(1, activation='sigmoid', name='out'+str(i))(i_out))
    outs = Concatenate(axis=-1)(outs)

    model = Model(inputs=[atom_input, atom_graph, bond_input], outputs=outs)
    #model.summary()
    model.compile(loss=custom_loss, optimizer=Nadam(learning_rate=cf.learning_rate), metrics=[])
    return model
