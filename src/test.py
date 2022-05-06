from absl import flags
from absl import app
from tqdm import tqdm

from torch.utils import tensorboard
import torchvision
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
import os
import sys
import logging

from data import preprocess
# from data import loader
from data.mstar import MSTAR
from train import load_dataset, test
from utils import common

from model.aconvnet import AConvNet

flags.DEFINE_string('model_path', 'SOC', help='please input the model-path')
FLAGS = flags.FLAGS

common.set_random_seed(12321)

def main(_):
    model_path = os.path.join(common.project_root,'experiments/model',FLAGS.model_path)
    config_name = os.path.join(model_path, 'config.json')
    ckpt_path = os.path.join(model_path, 'model.pth')

    config = common.load_config(config_name)

    dataset_name = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
    epochs = config['epochs']
    batch_size = config['batch_size']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']
    model_name = config['model_name']

    logging.root.handlers = []
    logging.basicConfig(
                format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
                datefmt='%m/%d %H:%M:%S',
                handlers=[
                    # logging.FileHandler(os.path.join(model_path, 'test.log')),
                    logging.StreamHandler(),
                    ],
                level=logging.INFO)
    logging.error('Start')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_loader = load_dataset(dataset_name, True, batch_size)
    val_loader = load_dataset(dataset_name, False, batch_size)

    model = AConvNet(channels, num_class=10, dropout_rate=0.1)
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    test(model, val_loader, device)
    import ipdb; ipdb.set_trace()

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
