import tensorflow as tf
from src.MSOEmultiscale import MSOEmultiscale

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
# config_proto.intra_op_parallelism_threads = 1
# config_proto.inter_op_parallelism_threads = 1
my_config = {}
my_config['train_filename'] = \
    '/local/ssd/mtesfald/FlyingChairs/data'
# my_config['train_filename'] = \
#     '/data/cv/UCF-101-train01-withflow'
# my_config['train_filename'] = \
#     '/data/cv/UCF-101-test01-withflow'
my_config['batch_size'] = 4
# my_config['batch_size'] = 10
my_config['augment_data'] = True
my_config['iterations'] = 600000
my_config['snapshot_frequency'] = 10000
my_config['print_frequency'] = 10
my_config['validation_frequency'] = 500
# my_config['lr'] = 6e-4
my_config['lr'] = 6e-3
my_config['num_threads'] = 6
my_config['num_scales'] = 5
my_config['gpu'] = 0
my_config['run_id'] = 'MSOEnet_chairs_6e-3_allaug_exceptscale_randorder_intlosses'

net = MSOEmultiscale(config={'tf': config_proto,
                             'user': my_config})
net.run_train()
