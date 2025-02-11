from __future__ import print_function, division
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
import importlib
import json
import random
import matplotlib

matplotlib.use('Agg')
import cv2
import method


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(13)


def eval_model(model, test_loader, args, infer_img_path=''):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The test data loader.
        args (object): Arguments object.
        infer_img_path (str): Path to save inference images.

    Returns:
        dict: A dictionary containing the evaluation metrics (MAE, MSE, RMSE, NMSE).
    """
    # Set model to evaluation mode
    model.eval()
    TTA = getattr(method, 'pmnettta').setup(model, args)
    # Initialize metrics
    n_samples = 0
    total_MAE = 0
    total_MSE = 0
    total_RMSE = 0
    total_NMSE = 0

    pred_cnt = 1  # Start counting from 1

    for inputs, targets in tqdm(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            preds = TTA.forward(inputs, targets)

            # Clamp predictions to [0, 1]
            preds = torch.clamp(preds, 0, 1)

            # Create directories for saving images
            inference_images_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'inference_images')
            input_images_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'input_images')
            os.makedirs(inference_images_path, exist_ok=True)
            os.makedirs(input_images_path, exist_ok=True)

            # Calculate metrics
            MAE = nn.L1Loss()(preds, targets).item()
            MSE = nn.MSELoss()(preds, targets).item()
            RMSE = torch.sqrt(((preds - targets) ** 2).mean()).item()
            NMSE = MSE / (nn.MSELoss()(targets, torch.zeros_like(targets)).item() + 1e-7)

            # Accumulate metrics
            total_MAE += MAE * inputs.size(0)
            total_MSE += MSE * inputs.size(0)
            total_RMSE += RMSE * inputs.size(0)
            total_NMSE += NMSE * inputs.size(0)
            n_samples += inputs.size(0)

            # Save prediction and label images
            if infer_img_path != '':
                for i in range(len(preds)):
                    # Convert prediction and label to numpy arrays
                    pred_img = preds[i][0].cpu().detach().numpy()
                    target_img = targets[i][0].cpu().detach().numpy()

                    # Rescale to [0, 255]
                    pred_img_rescaled = (pred_img * 255).astype(np.uint8)
                    target_img_rescaled = (target_img * 255).astype(np.uint8)

                    # Combine prediction and label images side by side
                    combined_img = np.hstack((pred_img_rescaled, target_img_rescaled))

                    # Save combined image
                    img_name = os.path.join(infer_img_path, 'inference_images', f'{pred_cnt}.png')
                    cv2.imwrite(img_name, combined_img)
                    pred_cnt += 1
                    if pred_cnt % 100 == 0:
                        print(f'{img_name} saved')

    # Calculate average metrics
    avg_MAE = total_MAE / n_samples
    avg_MSE = total_MSE / n_samples
    avg_RMSE = total_RMSE / n_samples
    avg_NMSE = total_NMSE / n_samples

    # Print overall metrics
    print(f"Overall testing MAE: {avg_MAE}")
    print(f"Overall testing MSE: {avg_MSE}")
    print(f"Overall testing NMSE: {avg_NMSE}")
    print(f"Overall testing RMSE: {avg_RMSE}")

    # Return metric dictionary
    results = {
        'MAE': avg_MAE,
        'MSE': avg_MSE,
        'RMSE': avg_RMSE,
        'NMSE': avg_NMSE
    }

    return results


def load_config_module(module_name, class_name):
    """Loads a configuration class from a module.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the configuration class.

    Returns:
        object: An instance of the configuration class.
    """
    module = importlib.import_module(module_name)
    config_class = getattr(module, class_name)
    return config_class()


if __name__ == "__main__":
    # Define arguments
    class Args:
        data_root = 'C:/Users/86156/Desktop/radio map/data/USC/'  # Data root directory
        network = 'pmnet_v3'  # Network type: pmnet_v1 or pmnet_v3
        model_to_eval = 'checkpoints/pmnet/USC_8H_8W.pt'  # Pre-trained model path
        config = 'config_USC_pmnetV3_V2'  # Configuration class name

    args = Args()

    print('start')
    cfg = load_config_module(f'config.{args.config}', args.config)
    print(cfg.get_train_parameters())
    cfg.now = datetime.today().strftime("%Y%m%d%H%M")  # YYYYmmddHHMM

    # Load dataset
    if cfg.sampling == 'exclusive':
        csv_file = os.path.join(args.data_root, 'Data_coarse_train.csv')

        data_train = None
        if 'usc' in args.config.lower():
            from dataloader.loaders import PMnet_usc

            num_of_maps = 19016
            ddf = pd.DataFrame(np.arange(1, num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_usc(csv_file=csv_file, dir_dataset=args.data_root)
        elif 'ucla' in args.config.lower():
            from dataloader.loaders import PMnet_ucla

            num_of_maps = 3776
            ddf = pd.DataFrame(np.arange(1, num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_ucla(csv_file=csv_file, dir_dataset=args.data_root)
        elif 'boston' in args.config.lower():
            from dataloader.loaders import PMnet_boston

            num_of_maps = 3143
            ddf = pd.DataFrame(np.arange(1, num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_boston(csv_file=csv_file, dir_dataset=args.data_root)

        dataset_size = len(data_train)

        train_size = int(dataset_size * cfg.train_ratio)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(data_train, [train_size, test_size],
                                                   generator=torch.Generator(device='cuda'))

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8,
                                  generator=torch.Generator(device='cuda'))
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8,
                                 generator=torch.Generator(device='cuda'))
    elif cfg.sampling == 'random':
        pass

    # Initialize PMNet
    if 'pmnet_v1' == args.network:
        from model.pmnet.pmnet_v1 import PMNet as Model

        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16, )

    elif 'pmnet_v3' == args.network:
        from model.pmnet.pmnet_v3 import PMNet as Model

        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8, )
    model.cuda()
    # Load pre-trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_to_eval))
    model.to(device)

    # Evaluate the model
    result = eval_model(model, test_loader, args, infer_img_path=os.path.split(args.model_to_eval)[-2])
    result_json_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'result.json')
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print('Evaluation score: ', result)