import tensorflow as tf
from src.MSOEmultiscale import MSOEmultiscale

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
# config.intra_op_parallelism_threads = 1
my_config = {}
my_config['train_filename'] = \
    '/local/ssd/mtesfald/UCF-101-withflow'
my_config['batch_size'] = 10
my_config['iterations'] = 600000
my_config['snapshot_frequency'] = 10000
my_config['print_frequency'] = 10
my_config['validation_frequency'] = 500
my_config['lr'] = 6e-3
my_config['num_threads'] = 6
my_config['num_scales'] = 5
my_config['gpu'] = 3
my_config['run_id'] = 'no_gating_deeperMSOEnet_2_ucf101'

net = MSOEmultiscale(config={'tf': config_proto,
                             'user': my_config})
net.run_train()
