# -- coding: utf-8 --

from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam


def get_opt(str):
    '''
    See below for hyper parameters
    https://keras.io/ja/optimizers/
    '''

    if str == 'SGD':
        return SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif str == 'SGDM':
        return SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    elif str == 'Adam':
        return Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
    elif str == 'RMSprop':
        return RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
    elif str == 'Adagrad':
        return Adagrad(lr=0.01, epsilon=None, decay=0.0)
    elif str == 'Adadelta':
        return Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, decay=0.0)
    elif str == 'Adamax':
        return Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif str == 'Nadam':
        return Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    else:
        raise NotImplementedError
