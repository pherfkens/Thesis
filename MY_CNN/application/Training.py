import random
import sys, os, argparse

gpu_id = 0 # ADDED BY ME
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")



sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters
from framework.mel_loudness_model_hop320 import *


def main(argv):
    batch_size = 32 # ADDED BY ME

    monitor = 'pleasant'                                       # INPUT!!!
    sys_name = 'system'

    system_path = os.path.join(os.getcwd(), sys_name)

    models_dir = os.path.join(system_path, 'model')

    using_model = AD_CNN_dense_layer_hop_combined                  # INPUT!!!    [AD_CNN_decreased_conv_layers, AD_CNN_linear_layer, AD_CNN_hop_length, AD_CNN_harder_max_pooling]  AD_CNN_hop_length_with_loudness AD_CNN_dense_layer_hop_combined

    model_name = using_model.__name__ + '_monitor_' + monitor

    model = using_model()
    print(f"Model: {model_name}")
    print(f"Parameters: {count_parameters(model)/1e3:.1f} K")

    if config.cuda and torch.cuda.is_available():
        model.cuda()

    Dataset_path = os.path.join(os.getcwd(), 'Dataset')                           # INPUT!!!
    generator = DataGenerator_Mel_loudness_no_graph(Dataset_path, using_mel=True, using_loudness=False, overwrite=False) # Set overwrite TRUE to Generate normalization file when training with HOP adjusted spectrograms

    Training_early_stopping(generator, model, monitor, models_dir, batch_size, 
                       using_mel=True, using_loudness=False, model_name=model_name)

    print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















