import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*20)])
    except RuntimeError as e:
        print(e)
from tensorflow.keras.backend import clear_session

import model_util
import model_frame
import data_config as cf 

import numpy as np
from sklearn.metrics import roc_auc_score

def training_multi(name):
    train_feature, train_label = model_util.load_data(name, 'train')
    valid_feature, valid_label = model_util.load_data(name, 'valid')
    train_label = train_label[:,0:7]
    valid_label = valid_label[:,0:7]

    model = model_frame.make_model(train_label.shape[1])
    model_name = str(name)+'_NR'

    myCB = model_util.customCallback((valid_feature, valid_label))
    es = model_util.customEarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=cf.patience)
    mc = model_util.customModelCheckpoint(os.path.join(cf.result_dir, model_name+'.h5'), monitor='val_auc', mode='max', save_best_only=True)

    modelLog = model.fit(train_feature,train_label, validation_data=(valid_feature,valid_label), \
            batch_size=cf.batch_size, epochs=cf.epochs, verbose=2, callbacks=[myCB,es,mc])

    validationLogFile = open(os.path.join(cf.result_dir, model_name+'.txt'), 'w')
    for l in model_util.totalLogs:
        validationLogFile.write("{:<6.3f}\n".format(l))
    validationLogFile.close()

    clear_session()

def test_multi(name):
    test_feature, test_label = model_util.load_data(name, 'test')
    test_label = test_label[:,0:7]

    model = model_frame.make_model(test_label.shape[1])
    model_name = str(name)+'_NR'

    model.load_weights(os.path.join(cf.result_dir, model_name+'.h5'))

    pred = model.predict(test_feature, 32, verbose=2)
    pred = np.array(pred)

    mean = 0 
    for idx in range(test_label.shape[1]):
        answer, predict = model_util.exceptNone(test_label, pred, idx)
        try:
            score = roc_auc_score(answer, predict)
        except ValueError:
            score = 0.0
        mean += score
        print(score)
    mean /= test_label.shape[1] 
    print('mean', mean)

    clear_session()

if __name__ == '__main__':
    training_multi('tox21')
    test_multi('tox21')
