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
from data import loader
from data.mstar import MSTAR
from utils import common

from model.aconvnet import AConvNet

flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', 'config/AConvNet-SOC.json', help='')
FLAGS = flags.FLAGS

common.set_random_seed(12321)

def load_dataset(name, is_train, batch_size):
    transform = [preprocess.CenterCrop(88), torchvision.transforms.ToTensor()]
    if is_train:
        pass
        # transform = [preprocess.RandomCrop(88), torchvision.transforms.ToTensor()]
    _dataset = MSTAR(
        name=name, is_train=is_train,
        transform=torchvision.transforms.Compose(transform)
        )
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=1
    )
    return data_loader

def test(model, loader, device):
    num_data = 0
    corrects = 0
    _loss = []
    criterion = torch.nn.CrossEntropyLoss()

    # Test loop
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        images, labels, _ = data
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        pred = F.softmax(logits, dim=-1)

        _, pred = torch.max(pred.data, 1)
        _loss.append(loss.item())
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (pred == labels.to(device)).sum().item()

    accuracy = 100 * corrects / num_data
    logging.info(f'Epoch: | val_loss={np.mean(_loss):.4f} | val_accuracy={accuracy:.2f}')



'''

TODO: replace the fuckin stupid model with normal settiings

- faster data loading
- new-log & proper cfg setup
- setup training and iterface [clear]

'''

def main(_):
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

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

    model_path = os.path.join(experiments_path, f'model/{model_name}')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    # setup logger
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(model_path, './model.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.basicConfig(
                format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
                datefmt='%m/%d %H:%M:%S',
                handlers=[ch, file_handler])

    logging.info('Start')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = load_dataset(dataset_name, True, batch_size)
    val_loader = load_dataset(dataset_name, False, batch_size)


    model = AConvNet(channels, num_class=10, dropout_rate=0.1)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1.e-5)

    # m = model.Model(
    #     classes=classes, dropout_rate=dropout_rate, channels=channels,
    #     lr=lr, lr_step=lr_step, lr_decay=lr_decay,
    #     weight_decay=weight_decay
    # )

    
    history = {
        'loss': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        _loss = []
        num_data = 0
        corrects = 0

        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            pred =F. softmax(logits, dim=-1)
            _, pred = torch.max(pred.data, 1)
            labels = labels.type(torch.LongTensor)
            num_data += labels.size(0)
            corrects += (pred == labels.to(device)).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.item())

            cur_lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

        accuracy = 100 * corrects / num_data

        logging.info(
            f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | accuracy={accuracy:.2f}'
        )

        history['loss'].append(np.mean(_loss))
        history['accuracy'].append(accuracy)

        # if experiments_path:
            # m.save(os.path.join(model_path, f'model-{epoch + 1:03d}.pth'))

        # with torch.no_grad():
            # num_data = 0
            # corrects = 0

            # # Test loop
            # model.eval()
            # for i, data in enumerate(tqdm(val_loader)):
                # images, labels, _ = data
                # images, labels = images.to(device), labels.to(device)

                # logits = model(images)
                # pred = F.softmax(logits, dim=-1)

                # _, pred = torch.max(pred.data, 1)
                # labels = labels.type(torch.LongTensor)
                # num_data += labels.size(0)
                # corrects += (pred == labels.to(device)).sum().item()

        # accuracy = 100 * corrects / num_data
        # f'Epoch: {epoch + 1:03d}/{epochs:03d} | val_loss={np.mean(_loss):.4f} | lr={cur_lr} | val_accuracy={accuracy:.2f}'
        # test on valid set
        test(model, val_loader, device)

    with open(os.path.join(history_path, f'history-{model_name}.json'), mode='w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
