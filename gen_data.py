from PIL import Image
import os, glob
import numpy as np
#from sklearn import cross_validation
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

#画像の読み込み

x, y = [], []

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
        x.append(data)
        y.append(index)

x = np.asarray(x)
y = np.asarray(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)
np.save("./animal.npy", xy)

# print(len(y_train))
# print(len(y_test))
