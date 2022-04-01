import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])
    except RuntimeError as e:
        print(e)
from tensorflow.keras.backend import clear_session

import model_util
import model_frame
import data_config as cf 

from sklearn.metrics import roc_auc_score

def test_single(name):
    test_feature, test_label = model_util.load_data(name, 'test')

    mean = 0 
    for i in range(test_label.shape[1]):
        model = model_frame.make_model(1)
        model_name = str(name)+'_single_'+str(i)

        model.load_weights(os.path.join(cf.result_dir, model_name+'.h5'))

        pred = model.predict(test_feature, 32, verbose=2)
        pred = np.array(pred)

        answer, predict = model_util.exceptNone(test_label[:,i:i+1], pred, 0)
        try:
            score = roc_auc_score(answer, predict)
        except ValueError:
            score = 0.0
        mean += score
        print(score)

        clear_session()

    mean /= test_label.shape[1] 
    print('mean', mean)

if __name__ == '__main__':
    #test_single('tox21')
    test_single('hiv')
