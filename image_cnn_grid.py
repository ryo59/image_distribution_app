import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam


classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

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
        return Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    elif str == 'RMSprop':
        return RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    elif str == 'Adagrad':
        return Adagrad(lr=0.01, epsilon=None, decay=0.0)
    elif str == 'Adadelta':
        return Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    elif str == 'Adamax':
        return Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif str == 'Nadam':
        return Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    else:
        raise NotImplementedError

opt_list = ["Adam", "SGD", "SGDM", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam"]
act_list = ["sigmoid", "relu"]
dr_list = [0.01, 0.1, 0.25, 0.5]
batch_list = [True, False]

total_grid = len(opt_list) * len(act_list) * len(dr_list) * len(batch_list)


def main():
    x_train, x_test, y_train, y_test = np.load("./data/animal_aug.npy", allow_pickle=True)
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    best_acc = 0
    grid_num = 0
    if not os.path.exists("./result"): os.makedirs("./result")
    with open("./result/result.txt", "w") as f:
        for opt in opt_list:
            for activation in act_list:
                for dropout in dr_list:
                    for batch_norm in batch_list:
                        print("\nCombination {} Left-> {}".format((grid_num + 1), (total_grid - (grid_num + 1))))
                        grid_num += 1
                        model = model_train(x_train, y_train, activation, dropout, batch_norm, opt)
                        scores = model.evaluate(x_test, y_test, verbose=1)
                        f.write("Optimizer: {} | Activation: {} | Dropout_rate: {} | Batch Normalization: {} | Test Loss: {} | Test Accuracy: {}\n".\
                                format(opt, activation, dropout, batch_norm, scores[0], scores[1]))

                        if scores[1] > best_acc:
                            best_param = []
                            best_param.append((opt, activation, dropout, batch_norm, scores[0], scores[1]))
                            model.save("./models/image_cnn_aug_grid.h5")
                            best_acc = scores[1]

        [param] = best_param

        f.write("\n---Best parameter---\nOptimizer: {} \nActivation: {} \nDropout_rate {} \nBatch Normalization: {} \nTest Loss: {} \nTest Accuracy: {}".\
                format(param[0], param[1], param[2], param[3], param[4], param[5]))
        print("\n---Best parameter---\nOptimizer: {} \nActivation: {} \nDropout_rate {} \nBatch Normalization: {} \nTest Loss: {} \nTest Accuracy: {}".\
                format(param[0], param[1], param[2], param[3], param[4], param[5]))

def model_train(x, y, activation, dropout_r, batch_norm, opt):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=x.shape[1:]))
    model.add(Activation(activation))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_r))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation(activation))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_r))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(dropout_r))
    model.add(Dense(3))
    model.add(Activation("softmax"))

    optiminzers = get_opt(opt)

    model.compile(loss="categorical_crossentropy", optimizer=optiminzers, metrics=["accuracy"])
    model.fit(x, y, batch_size=32, epochs=1)

    return model


if __name__ == "__main__":
    main()
