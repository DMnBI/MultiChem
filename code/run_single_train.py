from run_config import *
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

def training_single(name):
    train_feature, train_label = model_util.load_data(name, 'train')
    valid_feature, valid_label = model_util.load_data(name, 'valid')
    for i in range(train_label.shape[1]):
        model = model_frame.make_model(1)
        model_name = str(name)+'_single_'+str(i)
        train_model(model_name, model, [train_feature, train_label[:,i:i+1], valid_feature, valid_label[:,i:i+1]])

if __name__ == '__main__':
    training_single('tox21')
    #training_single('hiv')
    #training_single('bbbp')
