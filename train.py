import sys
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# These three lines of code I have to add in order for tensorflow-gpu
# to run on my computer.
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def main():

    model_path = "./model.h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min',
                                 save_best_only='True', verbose=1)

    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu',
                   kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    model = Model(inputs=model.inputs, outputs=output)

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.mean = [123.68, 116.779, 103.939]

    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                           class_mode='binary', batch_size=16,
                                           target_size=(224, 224))
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                          class_mode='binary', batch_size=16,
                                          target_size=(224, 224))


    model.fit_generator(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=10, callbacks=[checkpoint], verbose=1)

    user_input = input("Do you want to evaluate the model on the test set?\
                       (y/n)")

    if user_input == 'y':
        score = model.evaluate_generator(test_it, steps=len(test_it),
                                         verbose=1)
        print("loss: ", score[0])
        print('accuracy: ', score[1])


if __name__ == '__main__':
    main()
