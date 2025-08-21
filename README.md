# Robust Radio Map Prediction: An Online Adaptation Approach with Dynamic Learning Rates
## Overview

This project presents a novel approach to improve the accuracy of radio map prediction by leveraging Test-Time Adaptation (TTA). Radio maps are vital for wireless network optimization and resource management. Addressing the limitations of traditional methods that rely on complex model innovation, we propose a TTA technique that enhances prediction precision without retraining the model, offering an efficient and easily integrable solution. This approach dynamically adjusts model parameters during inference, boosting the model's generalization ability and prediction accuracy.

**This work is described in detail in our paper:Robust Radio Map Prediction: An Online Adaptation Approach with Dynamic Learning Rates.**

## Key Contributions

*   **Online Adaptation:** An efficient method to enhance radio map prediction accuracy *without* model retraining, improving generalization and prediction performance through dynamic parameter adjustment during inference.
*   **Novel Scenario Evaluation:** Introduction of three new, challenging scenarios for radio map prediction:
    *   Noise Scenarios
    *   Car Obstruction Scenarios
    *   Building Absence Scenarios
*   **Dynamic Learning Rate Adjustment:** A feature storage module-based approach for dynamic learning rate adjustment.  This module stores feature representations and reconstructed radio signal maps from processed inputs.
*   **Feature Similarity-Based Adaptation:** By caching historical data, the method calculates the similarity between current inputs and historical samples to dynamically adjust the model's learning rate. This allows the model to adapt to environmental changes and improve prediction accuracy and generalization.

## Model

The project utilizes the [RME-GAN or RadioUNet or REMNET]  architecture for radio map prediction. 

Usage
1. Requesting Model Weights and Dataset
Due to the size and potential licensing restrictions, the full dataset and pre-trained model weights are not directly available in this repository. To request access, please send an email to:

ljw199817@163.com

In your email, please briefly describe your intended use of the data and model. We will provide access to the dataset and pre-trained weights upon approval.

2. Data Preparation
After receiving the dataset, organize it according to the instructions provided in the access email.

3. Model Testing
To test the model with TTA and dynamic learning rate adjustment: python test.py --model [model_name]
