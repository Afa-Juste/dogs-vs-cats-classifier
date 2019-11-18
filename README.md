# dogs-vs-cats-classifier

This is a binary classifier CNN that uses a pretrained VGG16 feature extractor
net. it distinguishes between images of dogs and cats with an accuracy of 97%.

The code is largely based on a tutorial from machinelearningmastery.com.


## Instructions:

1. In order to use train the model and make inferences you have to first download
the dogs-vs-cats dataset from Kaggle.com, unzip them, and put the train/  
directory in the same directory as the scripts in the repository.

2. Create an empty test/ directory in the same directory as the scripts in
the repo, and put the unzipped test1/ directory inside it. This is required in
order to properly do inference with the inference.py script.

3. Run the directory_preparer.py script to organize the dataset in the
way the ImageDataGenerator from tensorflow.keras class requires.

4. Run the train.py script to train the model.

5. Run the inference.py script.
