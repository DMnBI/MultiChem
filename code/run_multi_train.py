from run_config import *
tf.config.experimental.set_visible_devices(gpus[2], 'GPU')

def training_multi(name):
    train_feature, train_label = model_util.load_data(name, 'train')
    valid_feature, valid_label = model_util.load_data(name, 'valid')

    model = model_frame.make_model(train_label.shape[1])
    model_name = str(name)+'_multi'
    train_model(model_name, model, [train_feature, train_label, valid_feature, valid_label])

if __name__ == '__main__':
    training_multi('tox21')
    #training_multi('toxcast')
    #training_multi('sider')
    #training_multi('clintox')
