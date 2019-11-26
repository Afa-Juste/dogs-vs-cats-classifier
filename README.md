# dogs-vs-cats-classifier

This is a binary classifier Convolutional Neural Network that uses a pretrained VGG16 feature extractor
net. It distinguishes between images of dogs and cats with an accuracy of 97%.


## Instructions:

1. In order to use train the model and make inferences you have to first download
the dogs-vs-cats dataset from kaggle.com, unzip it, and put the resulting unzipped **train/** directory 
in the same directory as the scripts in the repo.

2. Create an empty **test/** directory in the same directory as the scripts in
the repo, and put the unzipped **test1/** directory inside it. This is required in
order to properly do inference with the inference.py script.

3. Run the **directory_preparer.py** script to organize the dataset in the
way the ImageDataGenerator from tensorflow.keras class requires.

4. Run the **train.py** script to train the model.

5. Run the **inference.py** script.
