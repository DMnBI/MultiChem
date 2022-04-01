import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*20)])
    except RuntimeError as e:
        print(e)
from tensorflow.keras.backend import clear_session

import model_util
import model_frame
import data_config as cf 

def training_multi(name):
    train_feature, train_label = model_util.load_data(name, 'train')
    valid_feature, valid_label = model_util.load_data(name, 'valid')

    model = model_frame.make_model(train_label.shape[1])
    model_name = str(name)+'_multi'

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

if __name__ == '__main__':
    #training_multi('tox21')
    #training_multi('toxcast')
    #training_multi('sider')
