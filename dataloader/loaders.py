from __future__ import print_function, division
import re
import os
import math
import torch
import random
import warnings
import numpy as np
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")


def extract_and_combine_numbers(name):
    """
    Extracts all numbers from a string and concatenates them.

    Args:
        name (str): The input string.

    Returns:
        str: A string containing the combined numbers.
    """
    numbers = re.findall(r'\d+', name)
    combined_numbers = ''.join(numbers)
    return combined_numbers


class AUGAN_scene1(Dataset):
    """
    Dataset class for AUGAN scene 1.
    """

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/Radiomapseer",
                 numTx=80,
                 thresh=0.2,
                 transform=transforms.ToTensor()):
        """
        Initializes the AUGAN_scene1 dataset.

        Args:
            maps (numpy.ndarray, optional):  Array of map indices. Defaults to np.zeros(1).
            phase (str, optional):  'train', 'val', or 'test'. Defaults to 'train'.
            num1 (int, optional): Start index for data loading. Defaults to 0.
            num2 (int, optional): End index for data loading. Defaults to 0.
            data (str, optional): Path to the dataset. Defaults to "data/Radiomapseer".
            numTx (int, optional): Number of transmitters. Defaults to 80.
            thresh (float, optional): Threshold for target values. Defaults to 0.2.
            transform (callable, optional):  Transform to apply to the images. Defaults to transforms.ToTensor().
        """

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            # np.random.seed(42)
            # np.random.shuffle(self.maps)
        else:
            self.maps = maps

        print('当前输入场景：AUGAN_scene1')
        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 128
        self.width = 128
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = os.path.join(self.data, "image")
        # self.build = os.path.join(self.data, "png", "buildings_complete")  # complete
        self.build = self.data + "/png/car/60"  # car obstruction
        # self.build = self.data + "/png/noise/0.1"  # noise
        # self.build = self.data + "/png/missing building/1"  # building missing
        self.antenna = os.path.join(self.data, "antenna")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:  A tuple containing:
                - arr_builds (torch.Tensor):  Building image tensor.
                - arr_antennas (torch.Tensor): Antenna image tensor.
                - arr_targets (torch.Tensor): Target image tensor.
                - name2 (str): Name of the antenna file.
        """
        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # Loading building
        builds = os.path.join(self.build, name1)
        arr_build = np.asarray(io.imread(builds))

        # Loading antenna
        antennas = os.path.join(self.antenna, name2)
        arr_antenna = np.asarray(io.imread(antennas))

        # loading target
        target = os.path.join(self.simulation, name2)
        arr_target = np.asarray(io.imread(target))

        # threshold setting
        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # transfer tensor
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2


class PMnet_usc(Dataset):
    """
    Dataset class for PMnet USC data.
    """

    def __init__(self, csv_file,
                 dir_dataset="data/USC/",
                 transform=transforms.ToTensor()):
        """
        Initializes the PMnet_usc dataset.

        Args:
            csv_file (str): Path to the CSV file containing data indices.
            dir_dataset (str, optional): Path to the dataset directory. Defaults to "data/USC/".
            transform (callable, optional): Transform to apply to the images. Defaults to transforms.ToTensor().
        """
        self.ind_val = pd.read_csv(csv_file)
        self.dir_dataset = dir_dataset
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.ind_val)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            list: A list containing:
                - inputs (torch.Tensor): Stacked input image tensor (buildings and Tx).
                - power (torch.Tensor): Power image tensor.
        """
        # Load city map
        # self.dir_buildings = self.dir_dataset+ "map/"
        # noise
        # self.dir_buildings = self.dir_dataset+ "noise/0.1/"
        # building missing
        # self.dir_buildings = self.dir_dataset+ "missing building/1"
        # car obstruction
        self.dir_buildings = os.path.join(self.dir_dataset, "car", "30")

        img_name_buildings = os.path.join(self.dir_buildings, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Tx (transmitter):
        self.dir_Tx = os.path.join(self.dir_dataset, "Tx")
        img_name_Tx = os.path.join(self.dir_Tx, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_Tx = np.asarray(io.imread(img_name_Tx))

        # Load Rx (reciever): (not used in our training)
        self.dir_Rx = os.path.join(self.dir_dataset, "Rx")
        img_name_Rx = os.path.join(self.dir_Rx, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_Rx = np.asarray(io.imread(img_name_Rx))

        # Load Power:
        self.dir_power = os.path.join(self.dir_dataset, "pmap")
        img_name_power = os.path.join(self.dir_power, str(self.ind_val.iloc[idx, 0])) + ".png"
        image_power = np.asarray(io.imread(img_name_power))

        inputs = np.stack([image_buildings, image_Tx], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            power = self.transform(image_power).type(torch.float32)

        return [inputs, power]


# RadioUNet
class RadioUNet_c(Dataset):
    """
    Dataset class for RadioUNet.
    """

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/Radiomapseer",
                 numTx=80,
                 thresh=0.2,
                 transform=transforms.ToTensor()):
        """
        Initializes the RadioUNet dataset.

        Args:
            maps (numpy.ndarray, optional): Array of map indices. Defaults to np.zeros(1).
            phase (str, optional): 'train', 'val', or 'test'. Defaults to 'train'.
            num1 (int, optional): Start index for data loading. Defaults to 0.
            num2 (int, optional): End index for data loading. Defaults to 0.
            data (str, optional): Path to the dataset. Defaults to "data/Radiomapseer".
            numTx (int, optional): Number of transmitters. Defaults to 80.
            thresh (float, optional): Threshold for target values. Defaults to 0.2.
            transform (callable, optional): Transform to apply to the images. Defaults to transforms.ToTensor().
        """
        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            # np.random.seed(42)
            # np.random.shuffle(self.maps)
        else:
            self.maps = maps

        print('当前输入场景：AUGAN_scene1')
        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 128
        self.width = 128
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = os.path.join(self.data, "image")
        # self.build = os.path.join(self.data, "png", "buildings_complete")
        self.build = self.data + "/png/car/60"  # car obstruction
        # self.build = self.data + "/png/noise/0.1"  # noise
        # self.build = self.data + "/png/missing building/1"  # building missing
        self.antenna = os.path.join(self.data, "antenna")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - arr_builds (torch.Tensor): Building image tensor.
                - arr_antennas (torch.Tensor): Antenna image tensor.
                - arr_targets (torch.Tensor): Target image tensor.
                - name2 (str): Name of the antenna file.
        """
        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # Loading building
        builds = os.path.join(self.build, name1)
        arr_build = np.asarray(io.imread(builds))

        # Loading antenna
        antennas = os.path.join(self.antenna, name2)
        arr_antenna = np.asarray(io.imread(antennas))

        # loading target
        target = os.path.join(self.simulation, name2)
        arr_target = np.asarray(io.imread(target))

        # threshold setting
        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # transfer tensor
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2


# REM-NET+
class loader_3D(Dataset):
    """
    Dataset class for REM-NET+ (3D).
    """

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/Radiomapseer3d",
                 numTx=80,
                 thresh=0.,
                 transform=transforms.ToTensor()):
        """
        Initializes the loader_3D dataset.

        Args:
            maps (numpy.ndarray, optional): Array of map indices. Defaults to np.zeros(1).
            phase (str, optional): 'train', 'val', or 'test'. Defaults to 'train'.
            num1 (int, optional): Start index for data loading. Defaults to 0.
            num2 (int, optional): End index for data loading. Defaults to 0.
            data (str, optional): Path to the dataset. Defaults to "data/Radiomapseer3d".
            numTx (int, optional): Number of transmitters. Defaults to 80.
            thresh (float, optional): Threshold for target values. Defaults to 0.0.
            transform (callable, optional): Transform to apply to the images. Defaults to transforms.ToTensor().
        """
        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(2024)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 256
        self.width = 256
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = os.path.join(self.data, "gain")
        # self.build = self.data + "/png/car/30"
        self.build = os.path.join(self.data, "png", "noise", "0.2")
        # self.build = self.data + "/png/missing building/1"
        # self.antenna = self.data + "/png/buildingsWHeight"
        self.antenna = os.path.join(self.data, "png", "antennasWHeight")
        self.free_pro = os.path.join(self.data, "png", "free_propagation")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - arr_builds (torch.Tensor): Building image tensor.
                - arr_antennas (torch.Tensor): Antenna image tensor.
                - arr_targets (torch.Tensor): Target image tensor.
                - name2 (str): Name of the antenna file.
        """
        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # Loading Build
        builds = os.path.join(self.build, name1)
        arr_build = np.asarray(io.imread(builds))

        # Loading Antenna
        antennas = os.path.join(self.antenna, name2)
        arr_antenna = np.asarray(io.imread(antennas))

        # Loading Path Loss Maps
        free_pro = os.path.join(self.free_pro, name2)
        arr_free_pro = np.asarray(io.imread(free_pro))

        # Loading Target
        target = os.path.join(self.simulation, name2)
        arr_target = np.asarray(io.imread(target))

        # Threshold Transfer
        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # 转张量 (Convert to Tensor)
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_free_pros = self.transform(arr_free_pro).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2


# RME-GAN
class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)
    And we assume a fixed sample size of 1% of 256x256"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="data/Radiomapseer/",  # path to dataset
                 numTx=80,
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=655 * 10,
                 num_samples_low=655,
                 num_samples_high=655 * 10,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10.
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The RadioUNet inputs.
            image_gain

        """

        # self.phase=phase

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        self.arr = np.arange(256)
        self.one = np.ones(256)
        self.img = np.outer(self.arr, self.one)

        if simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "act-gan image/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsDPM/"

        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT2/"

        elif simulation == "rand":
            if carsSimul == "no":
                self.dir_gainDPM = self.dir_dataset + "gain/DPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gainDPM = self.dir_dataset + "gain/carsDPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/carsIRT2/"

        self.IRT2maxW = IRT2maxW

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":

            # self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
            # self.dir_buildings=self.dir_dataset+"png/missing build1/"
            # self.dir_buildings=self.dir_dataset+"png/missing build2/"
            # self.dir_buildings=self.dir_dataset+"png/missing build4/"
            # self.dir_buildings=self.dir_dataset+"png/build2-100/"
            # self.dir_buildings=self.dir_dataset+"png/build3-30/"
            self.dir_buildings = self.dir_dataset + "png/car/60"
            # self.dir_buildings=self.dir_dataset+"png/2d 0.05/"
            # self.dir_buildings=self.dir_dataset+"png/2d 0.1/"
            # self.dir_buildings=self.dir_dataset+"png/2d 0.2/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.fix_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high

        self.transform = transform

        self.dir_Tx = self.dir_dataset + "antenna"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings)) / 256

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx)) / 256

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 256
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w = np.random.uniform(0, self.IRT2maxW)  # IRT2 weight of random average
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        image_gain = image_gain * 256  # we use this normalization so all RadioUNet methods can have the same learning rate.
        # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
        # Important: when evaluating the accuracy, remember to devide the errors by 256!

        # input measurements
        image_samples = np.zeros((256, 256))
        if self.fix_samples == 0:
            num_samples = np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_samples = np.floor(self.fix_samples).astype(int)
        x_samples = np.random.randint(0, 255, size=num_samples)
        y_samples = np.random.randint(0, 255, size=num_samples)

        if self.fix_samples == 1:
            side = np.random.randint(0, 2)
            if side == 1:
                x_samples = np.append(np.random.randint(0, 128, size=6550), np.random.randint(128, 255, size=655))
                y_samples = np.random.randint(0, 255, size=6550 + 655)
            else:
                x_samples = np.append(np.random.randint(0, 128, size=655), np.random.randint(128, 255, size=6550))
                y_samples = np.random.randint(0, 255, size=6550 + 655)

        image_samples[x_samples, y_samples] = image_gain[x_samples, y_samples, 0]

        def objective(x, theta, c):
            return c - 10 * theta * x

        xk, yk = np.where(image_samples != 0)
        xk, yk = xk.reshape(xk.shape[0], 1), yk.reshape(yk.shape[0], 1)
        p, q = np.where(image_Tx != 0)
        # p,q = p.reshape(p.shape[0],1),p.reshape(p.shape[0],1)
        # print(p,q)
        x = np.log10(np.sqrt(np.square(xk - p) + np.square(yk - q)) + 1e-30).flatten()
        y = image_samples[xk, yk].flatten()

        # print(x.shape)
        # print(y.shape)
        pop, _ = curve_fit(objective, x, y)
        theta, c = pop
        genImg = c - 10 * theta * np.log10(np.sqrt(np.square(p - self.img) + np.square(q - self.img.T)) + 1e-30)

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx, image_samples, genImg], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_samples, genImg, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            # note that ToTensor moves the channel from the last asix to the first!

        return [inputs, image_gain]
