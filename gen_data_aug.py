from PIL import Image
import os, glob
import numpy as np
#from sklearn import cross_validation
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

#画像の読み込み

x_train, x_test = [], []
y_train, y_test = [], []



for index, class_name in enumerate(classes):
    photos_dir = "images/" + class_name
    files = glob.glob(photos_dir + "/*.jpg")
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



x_train, x_test = np.asarray(x_train), np.asarray(x_test)
Y_train, Y_test = np.asarray(y_train), np.asarray(y_test)

#x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)
xy = (x_train, x_test, Y_train, Y_test)
np.save("./data/animal_aug.npy", xy)

# print(len(y_train))
# print(len(y_test))
