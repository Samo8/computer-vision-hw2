import tensorflow as tf
import tensorflow_datasets as tfds

import prepare_masks as prep
import utils
import glob
from matplotlib import pyplot as plt
from functools import partial

import IPython.display as display


# data = prep.loadImagesAndMasks()

# https://keras.io/examples/keras_recipes/tfrecord/

IMAGE_SIZE = [1024, 1024]
BATCH_SIZE = 4
AUTOTUNE = tf.data.AUTOTUNE

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames)
    # print(list(dataset.as_numpy_iterator()))
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    for batch in dataset:
        print("x = {x:.4f},  y = {y:.4f}".format(**batch))
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

# dataset = tf.data.TFRecordDataset('task_taltech_cv-2021_11_24_20_25_56-tfrecord 1.0/default.tfrecord')


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


# train_dataset = get_dataset('task_taltech_cv-2021_11_24_20_25_56-tfrecord 1.0/default.tfrecord')
filenames = ['task_taltech_cv-2021_11_24_20_25_56-tfrecord 1.0/default.tfrecord']
raw_dataset = tf.data.TFRecordDataset(filenames)
# print(raw_dataset)
for raw_record in raw_dataset.take(2):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)

image_feature_description = {
    # 'height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_dataset.map(_parse_image_function)
print(parsed_image_dataset)

# for image_features in parsed_image_dataset:
#   image_raw = image_features['image'].numpy()
#   print(image_raw)
#   display.display(display.Image(data=image_raw))

    # for raw_record in image_features:
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     print(example)
# parsed_image_dataset
# for raw_record in raw_dataset.take(2):
#   example = tf.train.Example()
#   example.ParseFromString(raw_record.numpy())
#   print(example)
# valid_dataset = get_dataset(VALID_FILENAMES)
# test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

# image_batch, label_batch = next(iter(train_dataset))


# def show_batch(image_batch, label_batch):
#     plt.figure(figsize=(10, 10))
#     for n in range(25):
#         ax = plt.subplot(5, 5, n + 1)
#         plt.imshow(image_batch[n] / 255.0)
#         if label_batch[n]:
#             plt.title("MALIGNANT")
#         else:
#             plt.title("BENIGN")
#         plt.axis("off")


# show_batch(image_batch.numpy(), label_batch.numpy())

# mobile = tf.keras.applications.mobilenet.MobileNet()
# print(mobile.summary())

# for raw_record in dataset.take(1):
#    print('RAW')
#    print(raw_record)
#    example = tf.train.Example()
#    example.ParseFromString(raw_record.numpy())
# print(example)



# train_dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.RandomFlip('horizontal'),
#   tf.keras.layers.RandomRotation(0.2),
# ])

# for image in train_dataset.take(1).tf.train.Example():
#   plt.figure(figsize=(10, 10))
#   first_image = image[0]
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#     plt.imshow(augmented_image[0] / 255)
#     plt.axis('off')


# class_names = dataset.class_names

# plt.figure(figsize=(10, 10))
# for images, labels in dataset.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
# train_images = dataset[:5].map(utils.load_image, num_parallel_calls=tf.data.AUTOTUNE)
# test_images = dataset[4:].map(utils.load_image, num_parallel_calls=tf.data.AUTOTUNE)

# print(train_images)

# print(info)
# print(dataset)