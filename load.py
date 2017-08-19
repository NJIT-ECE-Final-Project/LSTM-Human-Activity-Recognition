import tensorflow as tf


sess = tf.Session()
new_saver = tf.train.import_meta_graph('/home/david/Code/Senior Project/Activity Recognition/LSTM-Human-Activity-Recognition/53/model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('/home/david/Code/Senior Project/Activity Recognition/LSTM-Human-Activity-Recognition/53/'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)