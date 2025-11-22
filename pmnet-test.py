from __future__ import print_function, division
import os
import time
#from tkinter import W
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from methods import Tent
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
torch.set_default_dtype(torch.float32)

torch.backends.cudnn.enabled
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import sys

import argparse
import importlib
import json
import random
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import methods



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(13)

# RESULT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension_Result'
RESULT_FOLDER = '"F:/doc/code/pmnet-main'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'

def L1_loss(pred, target):
  loss = nn.L1Loss()(pred, target)
  return loss

def MSE(pred, target):
  loss = nn.MSELoss()(pred, target)
  return loss

def RMSE(pred, target, metrics=None):
  loss = (((pred-target)**2).mean())**0.5
  return loss

def eval_model(model, test_loader, cfg=None, infer_img_path=''):

    # 设置模型为评估模式
    model.eval()
    TTA = getattr(methods, 'Tent').setup(model, args)
    # 初始化指标
    n_samples = 0
    total_MAE = 0
    total_MSE = 0
    total_RMSE = 0
    total_NMSE = 0

    pred_cnt = 1  # 从 1 开始计数

    for inputs, targets in tqdm(test_loader):
        iteration = 0
        iteration += 1
        inputs = inputs.cuda()
        targets = targets.cuda()

        # 对输入应用噪声和随机黑块
        # inputs = add_noise(inputs, noise_factor=0.05)
        # inputs = add_white_patches(inputs, horizontal_size=(2, 5), vertical_size=(5, 2), num_patches=30)

        # if infer_img_path != '':
        #     for i in range(len(inputs)):
        #         # 将输入张量转换为 NumPy 数组
        #         input_img = inputs[i][0].cpu().detach().numpy()
        #
        #         # 重新缩放到 [0, 255]
        #         input_img_rescaled = (input_img * 255).astype(np.uint8)
        #
        #         # 保存输入图像
        #         img_name = os.path.join(infer_img_path, 'input_images', f'{pred_cnt}.png')
        #         cv2.imwrite(img_name, input_img_rescaled)
        #         pred_cnt += 1
        #         if pred_cnt % 100 == 0:
        #             print(f'{img_name} saved')

        with torch.no_grad():
            # preds = model(inputs)
            preds = TTA.forward(inputs,targets)

            #
            preds = torch.clamp(preds, 0, 1)
            inference_images_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'inference_images')
            input_images_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'input_images')
            os.makedirs(inference_images_path, exist_ok=True)
            os.makedirs(input_images_path, exist_ok=True)


            # 计算指标
            MAE = nn.L1Loss()(preds, targets).item()
            MSE = nn.MSELoss()(preds, targets).item()
            RMSE = torch.sqrt(((preds - targets) ** 2).mean()).item()
            NMSE = MSE / (nn.MSELoss()(targets, torch.zeros_like(targets)).item() + 1e-7)



            # 累加指标
            total_MAE += MAE * inputs.size(0)
            total_MSE += MSE * inputs.size(0)
            total_RMSE += RMSE * inputs.size(0)
            total_NMSE += NMSE * inputs.size(0)
            n_samples += inputs.size(0)

            # 保存预测和标签图像
            if infer_img_path != '':
                for i in range(len(preds)):
                    # 将预测和标签转换为 numpy 数组
                    pred_img = preds[i][0].cpu().detach().numpy()
                    target_img = targets[i][0].cpu().detach().numpy()

                    # 重新缩放到 [0, 255]
                    pred_img_rescaled = (pred_img * 255).astype(np.uint8)
                    target_img_rescaled = (target_img * 255).astype(np.uint8)

                    # 将预测和标签图像并排组合
                    combined_img = np.hstack((pred_img_rescaled, target_img_rescaled))

                    # 保存组合后的图像
                    img_name = os.path.join(infer_img_path, 'inference_images', f'{pred_cnt}.png')
                    cv2.imwrite(img_name, combined_img)
                    pred_cnt += 1
                    if pred_cnt % 100 == 0:
                        print(f'{img_name} saved')

    # 计算平均指标
    avg_MAE = total_MAE / n_samples
    avg_MSE = total_MSE / n_samples
    avg_RMSE = total_RMSE / n_samples
    avg_NMSE = total_NMSE / n_samples

    # 打印总体指标
    print(f"Overall testing MAE: {avg_MAE}")
    print(f"Overall testing MSE: {avg_MSE}")
    print(f"Overall testing NMSE: {avg_NMSE}")
    print(f"Overall testing RMSE: {avg_RMSE}")


    # 返回指标字典
    results = {
        'MAE': avg_MAE,
        'MSE': avg_MSE,
        'RMSE': avg_RMSE,
        'NMSE': avg_NMSE
    }

    return results


def load_config_module(module_name, class_name):
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        return config_class()


if __name__ == "__main__":

    # 手动设置参数，模拟命令行传递的方式
    class Args:
        data_root = 'F:/doc/code/pmnet-main/data/USC/'  # 数据根目录
        network = 'pmnet_v3'  # 网络类型，pmnet_v1 或 pmnet_v3
        model_to_eval = 'checkpoints/USC_8H_8W.pt'  # 预训练模型路径
        config = 'config_USC_pmnetV3_V2'  # 配置类名


    args = Args()

    # 剩下的代码不变
    print('start')
    cfg = load_config_module(f'config.{args.config}', args.config)
    print(cfg.get_train_parameters())
    cfg.now = datetime.today().strftime("%Y%m%d%H%M")  # YYYYmmddHHMM

    # Load dataset
    if cfg.sampling == 'exclusive':
        csv_file = os.path.join(args.data_root, 'Data_coarse_train.csv')

        data_train = None
        if 'usc' in args.config.lower():
            from dataloader.loader_USC import PMnet_usc

            num_of_maps = 19016
            ddf = pd.DataFrame(np.arange(1, num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_usc(csv_file=csv_file, dir_dataset=args.data_root)
        elif 'ucla' in args.config.lower():
            from dataloader.loader_UCLA import PMnet_ucla

            num_of_maps = 3776
            ddf = pd.DataFrame(np.arange(1, num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_ucla(csv_file=csv_file, dir_dataset=args.data_root)
        elif 'boston' in args.config.lower():
            from dataloader.loader_Boston import PMnet_boston

            num_of_maps = 3143
            ddf = pd.DataFrame(np.arange(1, num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_boston(csv_file=csv_file, dir_dataset=args.data_root)

        dataset_size = len(data_train)

        train_size = int(dataset_size * cfg.train_ratio)
        # validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(data_train, [train_size, test_size],
                                                   generator=torch.Generator(device='cpu')) # 改为 cpu

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8) # 删掉 generator 参数即可
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8) # 删掉 generator 参数即可
        

    elif cfg.sampling == 'random':
        pass

    # Initialize PMNet and Load pre-trained weights if given.
    if 'pmnet_v1' == args.network:
        from models.pmnet_v1 import PMNet as Model

        # init model
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16, )

        model.cuda()
    elif 'pmnet_v3' == args.network:
        from models.pmnet_v3 import PMNet as Model

        # init model
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8, )

        model.cuda()

    # Load pre-trained weights to evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_to_eval))
    model.to(device)

    # create inference images directory if not exist
    # os.makedirs(os.path.join(os.path.split(args.model_to_eval)[-2], 'inference_images'), exist_ok=True)

    result = eval_model(model, test_loader, cfg=None,infer_img_path=os.path.split(args.model_to_eval)[-2])
    result_json_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'result.json')
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print('Evaluation score: ', result)

