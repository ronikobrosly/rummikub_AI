from os import listdir
from os.path import expanduser

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from tqdm import tqdm


NEW_TRAINING_IMAGES = 1500
NEW_TEST_IMAGES = 200

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')



# Get directory contents
raw_sep_images = listdir(expanduser('~/Desktop/rummikub/images/separate_initial_images/'))

raw_sep_image_names = [i.split('.')[0] for i in raw_sep_images]


train_label_list = []
test_label_list = []


# Augment the images I have
for raw_image in tqdm(raw_sep_image_names):

    # Load the image, convert into 4d numpy array of shape (1, 170, 120, 3)
    img = load_img(expanduser(f'~/Desktop/rummikub/images/separate_initial_images/{raw_image}.png'))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Now use `datagen` to create new images for training
    i = 0
    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=expanduser('~/Desktop/rummikub/images/training/'),
        save_prefix=raw_image,
        save_format='jpeg'
    ):
        i += 1
        train_label_list.append(raw_image)

        if i > NEW_TRAINING_IMAGES:
            break



    # Do the same for testing
    i = 0
    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=expanduser('~/Desktop/rummikub/images/testing/'),
        save_prefix=raw_image,
        save_format='jpeg'
    ):
        i += 1
        test_label_list.append(raw_image)
        if i > NEW_TEST_IMAGES:
            break
