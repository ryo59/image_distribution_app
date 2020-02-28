# -- coding: utf-8 --
import coremltools
from keras.models import load_model

coreml_model =\
coremltools.converters.keras.convert('../models/image_cnn_aug.h5', input_names='image', image_input_names='image', \
                                     output_names='Prediction', class_labels=['monkey', 'boar', 'crow'])
coreml_model.save('../models/image_cnn_aug.mlmodel')
