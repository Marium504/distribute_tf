#coding = utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000

MODEL_SAVE_PATH = "./model_dis/"
DATA_PATH = "./data/"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', 'worker', '"ps" or "worker"')
tf.app.flags.DEFINE_string(
    'ps_hosts', 'tf-ps0:2222, tf-ps1:1111',
    'Comma-separated list of hostname: port for the parameter server jobs.'
    'e.g. "tf-ps0:2222, tf-ps1:1111" ')
tf.app.flags.DEFINE_string(
    'worker_hosts', 'tf-worker0:2222, tf-worker1:1111',
    'Comma-separated list of hostname: port for the worker jobs. '
    'e.g. "tf-worker0:2222, tf-worker1:1111" ')
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')


def build_mode(x, y_):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000/BATCH_SIZE, LEARNING_RATE_DECAY)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return global_step, loss, train_op


def main(unused_argv):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id,
                                                  cluster=cluster)):

        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name= 'x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')
        global_step, loss, train_op = build_mode(x, y_)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        # init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief = is_chief, logdir= MODEL_SAVE_PATH, init_op = init_op, recovery_wait_secs=1,
                                 global_step=global_step)
            # summary_op = summary_op,
            # saver = saver,
            # global_step = global_step,
            # save_model_secs= 60,
            # save_summaries_secs= 60 )
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        sess = sv.prepare_or_wait_for_session(server.target,)# config=sess_config

        step = 0
        start_time = time.time()

        # while not sv.should_stop() and step< TRAINING_STEPS:
        while True:
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step_value = sess.run(
                [train_op, loss, global_step], feed_dict = {x:xs, y_:ys} )
            # print("loss_value:", loss_value, "global_step_value:", global_step_value)

            if global_step_value >= TRAINING_STEPS:
                print("break")
                break

            if step > 0 and step %100 ==0:
                duration = time.time() - start_time
                sec_per_batch = duration / global_step_value
                format_str = ("After %d training steps (%d global steps), "
                              "loss on training batch is %g.   "
                              "(%.3f sec/batch)")
                print(format_str % (step, global_step_value, loss_value, sec_per_batch))
            step += 1
    sess.close()

    # sv.stop()

if __name__ == "__main__":
    tf.app.run()


