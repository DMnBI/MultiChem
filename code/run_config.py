import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
        tf.config.experimental.set_memory_growth(gpus[2], True)
        tf.config.experimental.set_memory_growth(gpus[3], True)
    except:
        raise RuntimeError
else:
    raise RuntimeError

from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard
import time

import model_util
import model_frame
import data_config as cf 

import numpy as np
from sklearn.metrics import roc_auc_score

def train_model(model_name, model, inputs):
    train_feature, train_label, valid_feature, valid_label = inputs

    #################################
    myCB = model_util.customCallback((valid_feature, valid_label))
    #es = model_util.customEarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=cf.patience)
    #mc = model_util.customModelCheckpoint(os.path.join(cf.result_dir, model_name+'.h5'), monitor='val_auc', mode='max', save_best_only=True)
    #################################
    es = model_util.customEarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=cf.patience)
    mc = model_util.customModelCheckpoint(os.path.join(cf.result_dir, model_name+'.h5'), monitor='val_loss', mode='min', save_best_only=True)
    #################################
    tb = TensorBoard(log_dir=os.path.join(cf.log_dir, model_name, time.strftime('%c',time.localtime())), histogram_freq=1)

    modelLog = model.fit(train_feature,train_label, validation_data=(valid_feature,valid_label), \
            batch_size=cf.batch_size, epochs=cf.epochs, verbose=2, callbacks=[myCB,es,mc,tb])
    #        batch_size=cf.batch_size, epochs=cf.epochs, verbose=2, callbacks=[es,mc,tb])

    validationLogFile = open(os.path.join(cf.result_dir, model_name+'.txt'), 'w')
    #################################
    #for l in model_util.totalLogs:
    #    validationLogFile.write("{:<6.3f}\n".format(l))
    #################################
    for logs in model_util.totalLogs:
        for l in logs:
            validationLogFile.write("{:<6.3f}\t".format(l))
        validationLogFile.write("\n")
    validationLogFile.close()

    clear_session()

def test_model(model_name, model, inputs):
    test_feature, test_label = inputs

    model.load_weights(os.path.join(cf.result_dir, model_name+'.h5'))

    pred = model.predict(test_feature, 32, verbose=2)
    pred = np.array(pred)

    clear_session()

    scores = []
    for idx in range(test_label.shape[1]):
        answer, predict = model_util.exceptNone(test_label, pred, idx)
        try:
            scores.append(roc_auc_score(answer, predict))
        except ValueError:
            scores.append(0.0)

    return np.array(scores)
