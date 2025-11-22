import os
import torch
import importlib
import numpy as np
import random
import shutil
import time
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import method
from dataloader import loaders
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from tqdm import tqdm
import argparse
from method import TTA,RMEGANTTA,pmnettta


def set_seed(seed=13):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_directory(directory):
    """Clears the specified directory of all files and subdirectories."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def compute_losses(prob, tgt, loss_functions):
    """Computes MAE, MSE, NMSE, and RMSE."""
    MAE_list, MSE_list, NMSE_list, RMSE_list = [], [], [], []

    # Normalization (ensure prob and tgt are in the same range)
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-10)
    tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min() + 1e-10)

    for p in prob:
        _MAE = loss_functions['MAE'](p, tgt).cpu().numpy()
        _MSE = loss_functions['MSE'](p, tgt).cpu().numpy()
        _RMSE = np.sqrt(_MSE)  # RMSE is the square root of MSE
        _NMSE = _MSE / (tgt.var().cpu().numpy() + 1e-10)  # NMSE normalized by variance

        MAE_list.append(_MAE)
        MSE_list.append(_MSE)
        NMSE_list.append(_NMSE)
        RMSE_list.append(_RMSE)

    return MAE_list, MSE_list, NMSE_list, RMSE_list


def save_image_direct(tensor, iteration, idx, output_dir="output_images"):
    """Directly save the tensor as an image using PyTorch's save_image function."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{iteration}_{idx}.png")
    save_image(tensor, file_path, normalize=True)


def test_actgan(checkpoint_path="checkpoints/ACT-GAN/G0470000.pt", batch_size=11, max_iterations=8000):
    """Tests the ACT-GAN model."""
    set_seed()
    output_dir = "output_images_actgan"
    clear_directory(output_dir)

    net = importlib.import_module('model.ACTGAN.au_gan')
    model = net.Generator().cuda()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    model.eval()

    loss_functions = {'MAE': nn.L1Loss(), 'MSE': nn.MSELoss()}
    OA = getattr(method, 'OA').setup(model)

    test_data = loaders.AUGAN_scene1(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=4,
                                 drop_last=False)

    iteration = 0
    for build, antenna, target, img_names in tqdm(test_dataloader, desc="Testing ACT-GAN"):
        iteration += 1
        build, antenna, target = build.cuda(), antenna.cuda(), target.cuda()

        with torch.no_grad():
            predict_imgs = OA.forward(build, antenna, target)  # Use TTA module

            for idx, (prob, tgt) in enumerate(zip(predict_imgs, target)):
                _MAE, _MSE, _NMSE, _RMSE = compute_losses(prob, tgt, loss_functions)

                print(f"ACT-GAN testing MAE {np.mean(_MAE)} || MSE {np.mean(_MSE)} || "
                      f"_NMSE {np.mean(_NMSE)} || _RMSE {np.mean(_RMSE)}")

                # save_image_direct(prob, iteration, f'predict_{idx}', output_dir)
                # save_image_direct(tgt, iteration, f'target_{idx}', output_dir)

            if iteration >= max_iterations:
                break



def test_rmegan(checkpoint_path="checkpoints/RME-GAN/Trained_ModelMSE_G.pt", batch_size=15, max_iterations=8000):
    """Tests the RME-GAN model."""
    set_seed(13)
    output_dir = "output_images_rmegan"
    clear_directory(output_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_setup = 1  # Index of training setup
    test_setup = 1  # Index of testing setup
    setups = ['uniform', 'twoside', 'nonuniform']
    test_loader = None

    if test_setup == 1:
        Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300)
        test_loader = DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=4)

    elif test_setup == 2:
        Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655 * 10)
        test_loader = DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=4)

    else:
        Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655 * 10)
        test_loader = DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=4)
    if test_loader is None:
        raise ValueError("Invalid test_setup value.")

    loss_functions = {'MAE': nn.L1Loss(), 'MSE': nn.MSELoss()}
    net = importlib.import_module('model.RMEGAN.modules')
    model = net.RadioWNet(phase="firstU")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    OA = getattr(method, 'OA').setup(model)

    iteration = 0
    start_time = time.time()
    for inputs, targets in tqdm(test_loader, desc="Testing RME-GAN"):  # Use tqdm for progress bar
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs[:, :3, :, :]  # Take only the first three channels

        with torch.no_grad():
            outputs = OA.forward(inputs, targets)  

            for idx, (prob, tgt) in enumerate(zip(outputs, targets)):
                _MAE, _MSE, _NMSE, _RMSE = compute_losses(prob, tgt, loss_functions)

                print(f"testing MAE {np.mean(_MAE)} || MSE {np.mean(_MSE)} || "
                      f"_NMSE {np.mean(_NMSE)} || _RMSE {np.mean(_RMSE)}")


            # for idx, (prob, tgt) in enumerate(zip(outputs, targets)):
            #     save_image_direct(prob, iteration, f'predict_{idx}', output_dir)
            #     save_image_direct(tgt, iteration, f'target_{idx}', output_dir)


        if iteration >= max_iterations:
            break


    time_elapsed = time.time() - start_time
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def test_radiounet(checkpoint_path="C:/Users//86156/Desktop/radio map/checkpoints/RadioUNet/Trained_Model_FirstU.pt", batch_size=16, max_iterations=8000):
    """Tests the RadioUNet model."""
    set_seed()
    output_dir = "output_images_radiounet"
    clear_directory(output_dir)

    net = importlib.import_module('model.RadioUNet.modules')
    model = net.RadioWNet().cuda()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    model.eval()

    loss_functions = {'MAE': nn.L1Loss(), 'MSE': nn.MSELoss()}
    OA = getattr(method, 'OA').setup(model)

    test_data = loaders.RadioUNet_c(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=4)

    iteration = 0
    for build, antenna, target, img_names in tqdm(test_dataloader, desc="Testing RadioUNet"):
        iteration += 1
        build, antenna, target = build.cuda(), antenna.cuda(), target.cuda()

        with torch.no_grad():
            predict_img = OA.forward(build, antenna, target)

            for idx, (prob, tgt) in enumerate(zip(predict_img, target)):
                _MAE, _MSE, _NMSE, _RMSE = compute_losses(prob, tgt, loss_functions)

                print(f"RadioUNet testing MAE {np.mean(_MAE)} || MSE {np.mean(_MSE)} || "
                      f"_NMSE {np.mean(_NMSE)} || _RMSE {np.mean(_RMSE)}")

                # save_image_direct(prob, iteration, f'predict_{idx}', output_dir)
                # save_image_direct(tgt, iteration, f'target_{idx}', output_dir)

            if iteration >= max_iterations:
                break

def test_remnet(checkpoint_path="checkpoints/REM-NET+/Deep_AE.pt", batch_size=12, max_iterations=8000):
    """Tests the REM-NET+ model."""
    set_seed()
    output_dir = "output_images_remnet"
    clear_directory(output_dir)

    loss_functions = {'MAE': nn.L1Loss(), 'MSE': nn.MSELoss()}
    net = importlib.import_module('model.REMNET.modules')
    model = net.REM_Net().cuda()

    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    model.eval()

    OA = getattr(method, 'OA').setup(model)

    # Create test data loader
    test_data = loaders.loader_3D(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=4)

    iteration = 0
    for build, antenna, target, img_names in tqdm(test_dataloader, desc="Testing REM-NET+"):
        iteration += 1
        build, antenna, target = build.cuda(), antenna.cuda(), target.cuda()

        # Make prediction
        with torch.no_grad():
            predict_img = OA.forward(build, antenna, target)

            for idx, (prob, tgt) in enumerate(zip(predict_img, target)):
                _MAE, _MSE, _NMSE, _RMSE = compute_losses(prob, tgt, loss_functions)
                print(f"REM-NET+ testing MAE {np.mean(_MAE)} || MSE {np.mean(_MSE)} || "
                      f"NMSE {np.mean(_NMSE)} || RMSE {np.mean(_RMSE)}")

                save_image_direct(prob, iteration, f'predict_{idx}', output_dir)
                save_image_direct(tgt, iteration, f'target_{idx}', output_dir)

            if iteration >= max_iterations:
                break
    print(f"REM-NET+ testing completed")


def main():
    """Main function to select and run the test for a specific model."""
    set_seed()

    # Use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Select the model to run tests on.")
    parser.add_argument("--model", choices=["actgan", "rmegan", "radiounet", "remnet"],
                        default="radiounet", help="Name of the model to test")
    args = parser.parse_args()

    model_to_test = args.model

    if model_to_test == "actgan":
        test_actgan()
    elif model_to_test == "rmegan":
        test_rmegan()
    elif model_to_test == "radiounet":
        test_radiounet()
    elif model_to_test == "remnet":
        test_remnet()
    else:
        print("Invalid model selection. Please choose 'actgan', 'rmegan', 'radiounet', or 'remnet'.")


if __name__ == "__main__":
    main()
