from run_config import *
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

def test_single(name):
    test_feature, test_label = model_util.load_data(name, 'test')

    scores = []
    for i in range(test_label.shape[1]):
        model = model_frame.make_model(1)
        model_name = str(name)+'_single_'+str(i)
        scores.append(test_model(model_name, model, [test_feature, test_label[:,i:i+1]])[0])

    for score in scores:
        print(score)
    print('mean', np.mean(scores))

if __name__ == '__main__':
    test_single('tox21')
    #test_single('hiv')
    #test_single('bbbp')
