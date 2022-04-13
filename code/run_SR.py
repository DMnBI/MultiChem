from run_config import *
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

def training_multi(name):
    train_feature, train_label = model_util.load_data(name, 'train')
    valid_feature, valid_label = model_util.load_data(name, 'valid')
    train_label = train_label[:,7:12]
    valid_label = valid_label[:,7:12]

    model = model_frame.make_model(train_label.shape[1])
    model_name = str(name)+'_SR'
    train_model(model_name, model, [train_feature, train_label, valid_feature, valid_label])

def test_multi(name):
    test_feature, test_label = model_util.load_data(name, 'test')
    test_label = test_label[:,7:12]

    model = model_frame.make_model(test_label.shape[1])
    model_name = str(name)+'_SR'
    scores = test_model(model_name, model, [test_feature, test_label])

    for score in scores:
        print(score)
    print('mean', np.mean(scores))

if __name__ == '__main__':
    training_multi('tox21')
    test_multi('tox21')
