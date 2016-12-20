import tensorflow as tf
from src.architecture import MSOEPyramid

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = True
# config.intra_op_parallelism_threads = 1
my_config = {}
my_config['dataset_path'] = '/home/mtesfald/UCF-101-gt/'
my_config['batch_size'] = 10
my_config['temporal_extent'] = 2
my_config['iterations'] = 200000
my_config['snapshot_frequency'] = 500
my_config['print_frequency'] = 1
my_config['base_lr'] = 3e-3
my_config['lr_gamma'] = 1.0
my_config['lr_stepsize'] = 300
my_config['lr_policy_start'] = 0
my_config['num_threads'] = 6
my_config['num_scales'] = 3

with tf.device('/gpu:1'):
    net = MSOEPyramid(config={'tf': config_proto,
                              'user': my_config})
    net.run_train()
