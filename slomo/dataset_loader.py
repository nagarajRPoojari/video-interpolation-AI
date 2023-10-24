import pathlib
import numpy as np
import tensorflow as tf

import os


def load_dataset(
    data_dir: pathlib.Path,
    batch_size: int = 32,
    buffer_size: int = 1000,
    cache: bool = False,
    train: bool = True,
):

    autotune = tf.data.experimental.AUTOTUNE
    folders= os.listdir(data_dir)
    for i in range(len(folders)):
        folders[i]= str(data_dir) + '/' + folders[i]
    ds = tf.data.Dataset.from_tensor_slices(folders)
    ds = ds.map(lambda x: load_frames(x, train), num_parallel_calls=autotune)

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if train:
        ds = ds.shuffle(buffer_size=buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(autotune)
    return ds


def load_frames(folder_path: str, train: bool):

    files = tf.io.matching_files(folder_path + "/*.jpg")
    sampled_indices = tf.random.shuffle(tf.range(12))[:3]
    flip_sequence = tf.random.uniform([], maxval=1, dtype=tf.int32)
    sampled_indices = tf.where(
        flip_sequence == 1 and train,
        tf.sort(sampled_indices, direction="DESCENDING"),
        tf.sort(sampled_indices)
    )
    sampled_indices = tf.sort(sampled_indices)
    sampled_files = tf.gather(files, sampled_indices)

    frame_0 = decode_img(sampled_files[0])
    frame_1 = decode_img(sampled_files[2])
    frame_t = decode_img(sampled_files[1])
    
    if train:
        frames = data_augment(tf.concat([frame_0, frame_1, frame_t], axis=2))
        frame_0, frame_1, frame_t = frames[:, :, :3], frames[:, :, 3:6], frames[:, :, 6:9]
    return (frame_0, frame_1, sampled_indices[1]), frame_t


def data_augment(image):

    image = tf.image.resize(image, [360, 360])

    image = tf.image.random_crop(image, size=[352, 352, 9])

    image = tf.image.random_flip_left_right(image)

    return image


def decode_img(image: str):

    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


