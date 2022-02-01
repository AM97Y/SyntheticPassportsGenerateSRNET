import tensorflow as tf
import os


sess = tf.Session()
# path = './model'
path = './model_logs/checkpoints/20211025202708/iter/test/'
model = 'iter-12000'
#model = 'final'
#new_saver = tf.train.import_meta_graph(f'{path}/{model}.meta')
#new_saver.restore(sess, f'{path}/{model}.data-00000-of-00001')

#new_saver = tf.train.import_meta_graph(f'/media/monster/My Passport/work/DataGen/SRNetTF/SRNet/model_logs/checkpoints/20211025202708/test/iter-12000.meta')
#new_saver.restore(sess,tf.train.latest_checkpoint('/media/monster/My Passport/work/DataGen/SRNetTF/SRNet/model_logs/checkpoints/20211025202708/'))

new_saver = tf.train.import_meta_graph(f'/media/monster/My Passport/work/DataGen/SRNetTF/SRNet/model/final.meta')
new_saver.restore(sess,tf.train.latest_checkpoint('/media/monster/My Passport/work/DataGen/SRNetTF/SRNet/model/'))
all_vars = tf.compat.v1.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)
    
print('Good')
