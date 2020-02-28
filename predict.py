from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import numpy as np
from PIL import Image
import sys
from utils.optimizer import get_opt
import keras

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(50, 50, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation("softmax"))

    opt = keras.optimizers.adam()

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # モデルのロード
    model = load_model("./models/image_cnn_aug.h5")

    return model

def main():
    image = Image.open(sys.argv[1])
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    persentage = int(result[predicted] * 100)
    print("{} ({}%)".format(classes[predicted], persentage))

if __name__ == "__main__":
    main()
