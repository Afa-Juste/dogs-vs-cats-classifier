from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

dataset_home ='dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']

for subdir in subdirs:
    labeldirs = ['cats/', 'dogs/']

    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

seed(1990)
val_ratio = 0.25

src_directory = 'train/'
for file in listdir(src_directory):
    src = src_directory + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, dst)
