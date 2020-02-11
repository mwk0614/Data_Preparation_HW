import tensorflow as tf
import numpy as np
import glob
import random
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded': _bytes_feature(img),
               'label': _float_feature(lab),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()

def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    imagefiles = glob.glob(imagedir)
    random.shuffle(imagefiles)
    writer = tf.io.TFRecordWriter(datadir)

    for i in range(len(imagefiles)):
        imagefile = imagefiles[i]
        label = float(imagefile[-11])
        image_data = _image_as_bytes(imagefile)
        example = make_example(image_data,label)
        writer.write(example)
    writer.close()

def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """

    # filename queue
    filenames = glob.glob(folder + '/*')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    # read serialized examples
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    # parse examples into features, each
    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                      'label': tf.FixedLenFeature([], tf.float32, default_value=0)}
    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    # decode data
    img = tf.decode_raw(features['encoded'], tf.uint8)
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.reshape(img,[28,28,1])
    img = tf.math.divide(img, 255.0)
    lab = tf.cast(features['label'], tf.float32)
    # mini-batch examples queue
    min_after_dequeue = 1000

    img_batch, lab_batch = tf.train.shuffle_batch([img, lab], batch_size=batch,
                                                  capacity=min_after_dequeue + 3 * batch,
                                                  num_threads=5, min_after_dequeue=min_after_dequeue)

    lab_batch = tf.cast(lab_batch, tf.int64)
    lab_10 = tf.one_hot(lab_batch, depth=10)
    lab = tf.reshape(lab_10, shape=[-1, 10])

    return img_batch, lab

