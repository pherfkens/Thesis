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


def main(argv):
    batch_size = 64 # ADDED BY ME

    monitor = 'ISOPls' 
    sys_name = 'system'

    system_path = os.path.join(os.getcwd(), sys_name)

    models_dir = os.path.join(system_path, 'model')

    using_model = AD_CNN

    model = using_model()

    if config.cuda and torch.cuda.is_available():
        model.cuda()

    Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_Mel_loudness_no_graph(Dataset_path)

    Training_early_stopping(generator, model, monitor, models_dir, batch_size)

    print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















