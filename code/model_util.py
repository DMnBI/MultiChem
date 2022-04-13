import os
import numpy as np

import data_config as cf 

def load_data(name, mode):
    load_file = np.load(os.path.join(cf.pp_dir, name+'_'+mode+'.npz'))
    load_A = load_file['atom']
    load_B = load_file['bond']
    load_G = load_file['graph']
    load_L = load_file['label']
    feature = [load_A, load_G, load_B]
    label = load_L

    return feature, label 

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session

from sklearn.metrics import roc_auc_score

from tensorflow.keras import backend as K

global totalLogs
totalLogs = []
global ourLogs
ourLogs = {} 

class customEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_auc', mode='max', verbose=1, patience=10):
        super(customEarlyStopping, self).__init__(monitor=monitor, mode=mode, verbose=verbose, patience=patience)

    def on_epoch_end(self, epoch, logs=None):
        global ourLogs
        if ourLogs.items():
            super().on_epoch_end(epoch, ourLogs)
        else:
            super().on_epoch_end(epoch, logs)

class customModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_auc', save_best_only=True, mode='max'):
        super(customModelCheckpoint, self).__init__(filepath=filepath, monitor=monitor, save_best_only=save_best_only, mode=mode)

    def on_epoch_end(self, epoch, logs=None):
        global ourLogs
        if ourLogs.items():
            super().on_epoch_end(epoch, ourLogs)
        else:
            super().on_epoch_end(epoch, logs)

def exceptNone(labels, preds, idx):
    answer = []
    predict = []
    for a, p in zip(labels[:, idx], preds[:, idx]):
        if a == -1:
            continue
        else:
            answer.append(a)
            predict.append(p)
    return answer, predict

def custom_run_batch(data, size, func):
    result = []
    for i in range(len(data[0])//size):
        result.append(func([data[j][i*size:i*size+size] for j in range(len(data))])[0])
    result.append(func([data[j][i*size+size:len(data[0])] for j in range(len(data))])[0])
    return np.concatenate(result)

class customCallback(Callback):
    def __init__(self, valid):
        super(customCallback, self).__init__()
        self.validFeature = valid[0]
        self.validLabel = valid[1]

    def on_epoch_end(self, epoch, logs=None):
        get_layer_output = K.function([self.model.input], [self.model.output])
        pred = custom_run_batch(self.validFeature, 32, get_layer_output) 

        pred = np.array(pred)

        ###############
        log_list = []
        ###############
        mean = 0 
        for idx in range(self.validLabel.shape[1]):
            answer, predict = exceptNone(self.validLabel, pred, idx)
            try:
                score = roc_auc_score(answer, predict)
            except ValueError:
                score = 0.0
            mean += score
            ###############
            log_list.append(score)
            ###############
        mean /= self.validLabel.shape[1] 

        clear_session()

        global totalLogs
        global ourLogs
        ourLogs = logs
        ourLogs['val_auc'] = mean 
        ###############
        #totalLogs.append(mean)
        ###############
        ###############
        log_list.append(mean)
        totalLogs.append(log_list)
        ###############
