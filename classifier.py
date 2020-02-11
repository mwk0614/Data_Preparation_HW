import tensorflow as tf
from preprocess import read_tfrecord


class Classifier(object):
    def __init__(self, FLAGS):
        self.args = FLAGS

    def build(self, x, reuse=None):
        """ TODO: define your model (2 conv layers and 2 fc layers?)
        x: input image
        logit: network output w/o softmax """
        with tf.variable_scope('model', reuse=reuse):
            x_reshape = tf.reshape(x, [-1, 28, 28, 1])

            W1 = tf.Variable(tf.random_normal([3, 3, 1, 128]))
            W2 = tf.Variable(tf.random_normal([3, 3, 128, 64]))
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 32]))
            W4 = tf.Variable(tf.random_normal([3, 3, 32, 16]))
            FC1 = tf.Variable(tf.random_normal([7 * 7 * 16, 100]))
            FC2 = tf.Variable(tf.random_normal([100, 10]))

            x1 = tf.nn.conv2d(x_reshape, W1, strides=[1, 1, 1, 1], padding="SAME")
            x1_relu = tf.nn.relu(x1)
            x1_pool = tf.nn.max_pool(x1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            x2 = tf.nn.conv2d(x1_pool, W2, strides=[1, 1, 1, 1], padding="SAME")
            x2_relu = tf.nn.relu(x2)

            x3 = tf.nn.conv2d(x2_relu, W3, strides=[1, 1, 1, 1], padding="SAME")
            x3_relu = tf.nn.relu(x3)
            x3_pool = tf.nn.max_pool(x3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            x4 = tf.nn.conv2d(x3_pool, W4, strides=[1, 1, 1, 1], padding="SAME")
            x4_relu = tf.nn.relu(x4)

            x5 = tf.layers.flatten(x4_relu)
            x5_fc1 = tf.matmul(x5, FC1)
            x5_fc2 = tf.matmul(x5_fc1, FC2)
            logit = x5_fc2

        return logit

    def accuracy(self, label_onehot, logit):
        """ accuracy between one-hot label and logit """
        softmax = tf.nn.softmax(logit, -1)
        prediction = tf.argmax(softmax, -1)
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label_onehot, -1), prediction), tf.float32))

    def train(self):
        """ train 10-class MNIST classifier """

        # load data
        tr_img, tr_lab = read_tfrecord(self.args.train_datadir, self.args.batch, self.args.epoch)
        val_img, val_lab = read_tfrecord(self.args.valid_datadir, self.args.batch, self.args.epoch)

        # graph
        tr_logit = self.build(tr_img)
        val_logit = self.build(val_img, True)

        step = tf.Variable(0, trainable=False)
        increment_step = tf.assign_add(step, 1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tr_lab, logits=tr_logit))
        optimizer = tf.train.AdamOptimizer(self.args.lr).minimize(loss, global_step=step)

        tr_accuracy = self.accuracy(tr_lab, tr_logit)
        val_accuracy = self.accuracy(val_lab, val_logit)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        saver = tf.train.Saver(max_to_keep=2, var_list = var_list)
        # session
        with tf.Session() as sess:
            if self.args.restore:
                saver.restore(sess, tf.train.latest_checkpoint(self.args.ckptdir))
            else:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                min_val_acc = 10000.
                while not coord.should_stop():
                    global_step = sess.run(step)
                    print(global_step)
                    batch_loss, batch_acc, _ = sess.run([loss, tr_accuracy, optimizer])
                    if global_step%1000 == 0:
                        print('step:: %d, loss= %.3f, accuracy= %.3f' % (global_step, batch_loss, batch_acc))

                    if global_step%3000 == 0:
                        val_acc = sess.run(val_accuracy)
                        print('val accuracy= %.3f' % val_acc)
                        if val_acc < min_val_acc:
                            min_val_acc = val_acc
                            save_path = saver.save(sess, self.args.ckptdir+'/model_%.3f.ckpt' % val_acc, global_step = step)
                            print('model saved in file: %s' % save_path)

                    sess.run(increment_step)

            except KeyboardInterrupt:
                print('keyboard interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, self.args.ckptdir+'/model.ckpt', global_step = step)
                print('model saved in file : %s' % save_path)
                coord.request_stop()
                coord.join(threads)

    def test(self):
        # load data
        ts_img, ts_lab = read_tfrecord(self.args.test_datadir, self.args.batch, None)

        # graph
        ts_logit = self.build(ts_img)

        step = tf.Variable(0, trainable=False)

        ts_accuracy = self.accuracy(ts_lab, ts_logit)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        saver = tf.train.Saver(var_list=var_list)

        # session
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.args.ckptdir))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            total_acc = 0.
            steps = 0
            while steps < 10000/self.args.batch:
                batch_acc = sess.run(ts_accuracy)
                total_acc += batch_acc
                steps += 1

            total_acc /= steps
            print('number: %d, total acc: %.1f' % (steps, total_acc*100)+'%')

            coord.request_stop()
            coord.join(threads)

