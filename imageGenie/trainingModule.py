import numpy as np
import tensorflow as tf

import os
from glob import glob

from imageGenie.constants import Constants

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as  PretrainedModel, preprocess_input

constants = Constants()

acc_thresh = constants.acc_threshold
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > acc_thresh):
            print("\nStopping training as desired accuracy reached....")
            self.model.stop_training = True

class TrainingModule(object):
    def __init__(self, root_folder, model_folder, resize_dim = [200,200,3], modelName = "effNetV2B0"):

        self.root_folder = root_folder
        self.IMAGE_SIZE = constants.image_size

        self.ptm = PretrainedModel(
        input_shape = self.IMAGE_SIZE + [3],
        weights = 'imagenet',
        include_top = False)

        self.model_folder = model_folder

        self.train_path = os.path.join(root_folder,"train")
        self.test_path = os.path.join(root_folder,"test")

        self.train_image_files = glob(self.train_path + '/*/*.png')
        self.test_image_files = glob(self.test_path + '/*/*.png')

        self.folders = glob(self.train_path + "/*")

    def train(self):
        self.ptm.trainable = False
        K = len(self.folders)

        x = Flatten()(self.ptm.output)
        x = Dense(K, activation = 'softmax')(x)

        model = Model(inputs = self.ptm.input , outputs = x)
        
        gen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.2,
        horizontal_flip = True,
        preprocessing_function = preprocess_input
        )

        batch_size = 64

        train_generator = gen.flow_from_directory(
            self.train_path,
            shuffle = True,
            target_size = self.IMAGE_SIZE,
            batch_size = batch_size
        )

        test_generator = gen.flow_from_directory(
            self.test_path,
            target_size = self.IMAGE_SIZE,
            batch_size = batch_size
        )

        model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

        self.model_file = os.path.join(self.model_folder, "trained_model.h5")

        try:
            os.mkdir(self.model_folder)
        except:
            pass

        myCall = myCallback()
        
        r = model.fit(
                train_generator,
                validation_data = test_generator,
                epochs = 8,
                steps_per_epoch = int(np.ceil(len(self.train_image_files)/batch_size)),
                validation_steps = int(np.ceil(len(self.test_image_files)/batch_size)),
                callbacks=[myCall]
        )

        model.save(self.model_file)
        print(f"Saving model at {self.model_file}")

