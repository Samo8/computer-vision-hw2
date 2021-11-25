import tensorflow as tf
import tensorflow_datasets as tfds

import prepare_masks as prep
import utils

dataset = prep.loadImagesAndMasks()


# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train_images = dataset[:5].map(utils.load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset[4:].map(utils.load_image, num_parallel_calls=tf.data.AUTOTUNE)

print(train_images)

# print(info)
# print(dataset)