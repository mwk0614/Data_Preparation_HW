# ISPL_Freshman_practice

<MNIST classification>

1)

download all files including mnist.zip

DO NOT use tensorflow.examples.tutorials.mnist library

2)

fill in the TODOs

you should modify main.py code

3)

upload your modified codes and performance-written text file(result.txt)

in your own repository

usage: 

    python main.py --process=write --imagedir=./mnist/train --datadir=./mnist/train_tfrecord

    or

    python main.py --process=train --datadir=./mnist/train_tfrecord --val_datadir=./mnist/val_tfrecord --epoch=1 --lr=1e-3 --ckptdir=./ckpt --batch=100 --restore=False
