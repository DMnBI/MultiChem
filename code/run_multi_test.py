from run_config import *
tf.config.experimental.set_visible_devices(gpus[2], 'GPU')

def test_multi(name):
    test_feature, test_label = model_util.load_data(name, 'test')

    model = model_frame.make_model(test_label.shape[1])
    model_name = str(name)+'_multi'
    scores = test_model(model_name, model, [test_feature, test_label])

    for score in scores:
        print(score)
    print('mean', np.mean(scores))

if __name__ == '__main__':
    test_multi('tox21')
    #test_multi('toxcast')
    #test_multi('sider')
    #test_multi('clintox')
