from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pandas as pd
import numpy as np

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datagen = ImageDataGenerator(featurewise_center=True)
datagen.mean = [123.68, 116.779, 103.939]

batch_size = 10
iterator = datagen.flow_from_directory('test', class_mode=None,
                                       batch_size=batch_size, target_size=(224, 224),
                                       shuffle=False)

model = load_model('model.h5')
results = model.predict_generator(iterator, steps=1, verbose=1)

results_string = list()
for result in results:
    if result == 0.:
        results_string.append('cat')
    else:
        results_string.append('dog')

filenames = list()
for file in iterator.filenames[:batch_size]:
    file = np.char.replace(file, 'test1/', '' , count=1)
    filenames.append(file)

df = pd.DataFrame({'Filename':filenames,
                  'Prediction': results_string})

print(df)
