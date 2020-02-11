import keras
from keras.applications import VGG16
from keras.models import Sequential, load_model
from keras import models, optimizers
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
import glob

batch_size = 32
num_classes = 3
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_vgg16_trained_model.h5'

### データセットの準備

folder = ["boar", "crow", "monkey"]

image_size = 50
num_testdata = 100

x_train, x_test = [], []
y_train, y_test = [], []

for index, name in enumerate(folder):
    dir = "images/" + name
    files = glob.glob(dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 200:
            break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        if i < num_testdata:
            x_test.append(data)
            y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                x_train.append(data)
                y_train.append(index)

                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                x_train.append(data)
                y_train.append(index)

X_train, X_test = np.asarray(x_train), np.asarray(x_test)
Y_train, Y_test = np.asarray(y_train), np.asarray(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 256
X_test /= 256

# ｙ　ラベルをワンホット表現に
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

### モデル構築
vgg_conv = VGG16(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = models.Sequential()
model.add(vgg_conv)

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  ### データは3種類

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.rmsprop(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

### 学習
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, Y_test),
                    shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
