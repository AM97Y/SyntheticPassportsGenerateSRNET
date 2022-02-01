"""
SRNet - Editing Text in the Wild
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

# device
gpu = 0

# pretrained vgg
vgg19_weights = 'SRNet/model_logs/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.pb'

# model parameters
lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
learning_rate = 1e-4 # default 1e-3
decay_rate = 0.9
decay_steps = 1000
staircase = False
beta1 = 0.9 # default 0.9
beta2 = 0.999 # default 0.999
max_iter = 10000
show_loss_interval = 50
write_log_interval = 50
save_ckpt_interval = 1000
gen_example_interval = 1000
checkpoint_savedir = 'model_logs/checkpoints'
tensorboard_dir = 'model_logs/train_logs'
pretrained_ckpt_path = 'model_logs/checkpoints/20211109153223/iter-6000'# '/media/monster/My Passport/work/DataGen/SRNetTF/SRNet/model/final'
train_name = None # used for name examples and tensorboard logdirs, set None to use time

# data
batch_size = 8
data_shape = [64, None]
data_dir = '/media/monster/My Passport/work/DataGen/SRNet/train'
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
example_data_dir = 'examples/labels'
example_result_dir = 'examples/gen_logs'

# predict
predict_ckpt_path = '/media/monster/My Passport/work/DataGen/SRNetTF/SRNet/model/final'
predict_data_dir = 'predict'
predict_result_dir = 'result_predict'
