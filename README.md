# Robust Radio Map Prediction: An Online Adaptation Approach with Dynamic Learning Rates

## Overview

This project presents a novel approach to improve the accuracy of radio map prediction by leveraging Online Adaptation. Radio maps are vital for wireless network optimization and resource management. Addressing the limitations of traditional methods that rely on complex model innovation, we propose an Online Adaptation technique that enhances prediction precision without retraining the model, offering an efficient and easily integrable solution. This approach dynamically adjusts model parameters during inference, boosting the model's generalization ability and prediction accuracy.

As shown in the image below:
<img width="1600" height="900" alt="Framework Diagram" src="https://github.com/user-attachments/assets/c9691d75-c02f-4e1c-916c-6fa4aaf1a9c1" />

**This work is described in detail in our paper: *Robust Radio Map Prediction: An Online Adaptation Approach with Dynamic Learning Rates*.**

## Key Contributions

*   **Online Adaptation:** An efficient method to enhance radio map prediction accuracy *without* model retraining, improving generalization and prediction performance through dynamic parameter adjustment during inference.
*   **Novel Scenario Evaluation:** Introduction of three new, challenging scenarios for radio map prediction:
    *   Noise Scenarios
    *   Car Obstruction Scenarios
    *   Building Absence Scenarios
*   **Dynamic Learning Rate Adjustment:** A feature storage module-based approach for dynamic learning rate adjustment. This module stores feature representations and reconstructed radio signal maps from processed inputs.
*   **Feature Similarity-Based Adaptation:** By caching historical data, the method calculates the similarity between current inputs and historical samples to dynamically adjust the model's learning rate. This allows the model to adapt to environmental changes and improve prediction accuracy and generalization.

## Datasets

We sincerely acknowledge the contributions of the authors of the original datasets (RadioMapSeer, USC, etc.). Their high-fidelity simulations served as the foundation for our work. Building upon these benchmarks, we have constructed specific datasets for the challenging scenarios evaluated in our paper.

You can download the datasets for the three interference scenarios below:

### 1. Noise Scenario Dataset
*   **Description:** Building maps corrupted with varying levels of Gaussian noise to simulate sensor inaccuracies.
*   **Download Link:** [Baidu Netdisk (百度网盘)](https://pan.baidu.com/s/1v2HOKdTOV0kknXEMDx4PAw?pwd=1234)
*   **Password:** `1234`

### 2. Car Obstruction Scenario Dataset
*   **Description:** Scenarios simulating transient occlusions caused by vehicles.
*   **Download Link:** *[Link to be updated soon]*

### 3. Building Absence Scenario Dataset
*   **Description:** Scenarios simulating abrupt structural changes (missing buildings).
*   **Download Link:** *[Link to be updated soon]*

## Model

The project utilizes the [RME-GAN / RadioUNet / REMNET] architecture for radio map prediction.

## Usage

### 1. Model Weights
Since our Online Adaptation (OA) method operates strictly during the **inference (test) phase**, no special training is required for the backbone models. 

*   **Standard Weights:** You can use any model weights pre-trained on standard datasets (e.g., RadioMapSeer, USC) using the standard training protocols.
*   **Requesting Our Pre-trained Weights:** If you wish to use the exact pre-trained weights used in our experiments to ensure full reproducibility of the baseline results, please send a request to: **ljw199817@163.com**.

### 2. Data Preparation
*   Download the specific scenario datasets (Noise, Car Obstruction, Building Absence) from the links provided in the **Datasets** section above.
*   After extracting the files, organize them into the appropriate data directories (e.g., `./data/test/`).

### 3. Model Testing (Online Adaptation)
To run the evaluation with Online Adaptation and dynamic learning rate adjustment, use the following command. The script will load the pre-trained weights and perform adaptation on-the-fly for each test sample.

```bash
python test.py --model [model_name]
