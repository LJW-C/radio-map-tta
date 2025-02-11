
"""
Configuration dictionary to store the Learning Rate (LR) and Tau values for different models across various scenarios.
This dictionary defines the default configurations for all models and scenarios. It can be imported and used in test scripts as needed.
This configuration is specifically designed to allow different models to utilize distinct Learning Rate and Tau values within the Test-Time Augmentation (TTA) algorithm.
"""

config = {
    "Uninterfered Scene": {
        "RadioUNet": {
            "LR": 0.0006,  # Learning Rate for RadioUNet in the Uninterfered Scene
            "tau": 0.9,    # Tau value for RadioUNet in the Uninterfered Scene
        },
        "RME-GAN": {
            "LR": 0.00002, # Learning Rate for RME-GAN in the Uninterfered Scene
            "tau": 0.01,   # Tau value for RME-GAN in the Uninterfered Scene
        },
        "ACT-GAN": {
            "LR": 0.000005, # Learning Rate for ACT-GAN in the Uninterfered Scene
            "tau": 0.5,    # Tau value for ACT-GAN in the Uninterfered Scene
        },
        "PMNet-v1": {
            "LR": 0.00002, # Learning Rate for PMNet-v1 in the Uninterfered Scene
            "tau": 0.1,    # Tau value for PMNet-v1 in the Uninterfered Scene
        },
        "PMNet-v3": {
            "LR": 0.00002, # Learning Rate for PMNet-v3 in the Uninterfered Scene
            "tau": 0.05,   # Tau value for PMNet-v3 in the Uninterfered Scene
        },
        "REM-NET+": {
            "LR": 0.002,  # Learning Rate for REM-NET+ in the Uninterfered Scene
            "tau": 0.4,    # Tau value for REM-NET+ in the Uninterfered Scene
        },
    },
    "Gaussian Noise": {
        "RadioUNet": {
            "LR": 0.0007,  # Learning Rate for RadioUNet in the Gaussian Noise scenario
            "tau": 0.9,    # Tau value for RadioUNet in the Gaussian Noise scenario
        },
        "RME-GAN": {
            "LR": 0.00002, # Learning Rate for RME-GAN in the Gaussian Noise scenario
            "tau": 0.01,   # Tau value for RME-GAN in the Gaussian Noise scenario
        },
        "ACT-GAN": {
            "LR": 0.00007, # Learning Rate for ACT-GAN in the Gaussian Noise scenario
            "tau": 0.5,    # Tau value for ACT-GAN in the Gaussian Noise scenario
        },
        "PMNet-v1": {
            "LR": 0.00002, # Learning Rate for PMNet-v1 in the Gaussian Noise scenario
            "tau": 0.01,   # Tau value for PMNet-v1 in the Gaussian Noise scenario
        },
        "PMNet-v3": {
            "LR": 0.00002, # Learning Rate for PMNet-v3 in the Gaussian Noise scenario
            "tau": 0.05,   # Tau value for PMNet-v3 in the Gaussian Noise scenario
        },
        "REM-NET+": {
            "LR": 0.0007,  # Learning Rate for REM-NET+ in the Gaussian Noise scenario
            "tau": 0.4,    # Tau value for REM-NET+ in the Gaussian Noise scenario
        },
    },
    "Car Occlusion": {
        "RadioUNet": {
            "LR": 0.00009,  # Learning Rate for RadioUNet in the Car Occlusion scenario
            "tau": 0.1,    # Tau value for RadioUNet in the Car Occlusion scenario
        },
        "RME-GAN": {
            "LR": 0.00002, # Learning Rate for RME-GAN in the Car Occlusion scenario
            "tau": 0.01,   # Tau value for RME-GAN in the Car Occlusion scenario
        },
        "ACT-GAN": {
            "LR": 0.000005, # Learning Rate for ACT-GAN in the Car Occlusion scenario
            "tau": 0.5,    # Tau value for ACT-GAN in the Car Occlusion scenario
        },
        "PMNet-v1": {
            "LR": 0.00002, # Learning Rate for PMNet-v1 in the Car Occlusion scenario
            "tau": 0.01,   # Tau value for PMNet-v1 in the Car Occlusion scenario
        },
        "PMNet-v3": {
            "LR": 0.00002, # Learning Rate for PMNet-v3 in the Car Occlusion scenario
            "tau": 0.05,   # Tau value for PMNet-v3 in the Car Occlusion scenario
        },
        "REM-NET+": {
            "LR": 0.00007,  # Learning Rate for REM-NET+ in the Car Occlusion scenario
            "tau": 0.4,    # Tau value for REM-NET+ in the Car Occlusion scenario
        },
    },
    "Building Missing": {
        "RadioUNet": {
            "LR": 0.00008,  # Learning Rate for RadioUNet in the Building Missing scenario
            "tau": 0.1,    # Tau value for RadioUNet in the Building Missing scenario
        },
        "RME-GAN": {
            "LR": 0.00002, # Learning Rate for RME-GAN in the Building Missing scenario
            "tau": 0.01,   # Tau value for RME-GAN in the Building Missing scenario
        },
        "ACT-GAN": {
            "LR": 0.000001, # Learning Rate for ACT-GAN in the Building Missing scenario
            "tau": 0.5,    # Tau value for ACT-GAN in the Building Missing scenario
        },
        "PMNet-v1": {
            "LR": 0.00002, # Learning Rate for PMNet-v1 in the Building Missing scenario
            "tau": 0.01,   # Tau value for PMNet-v1 in the Building Missing scenario
        },
        "PMNet-v3": {
            "LR": 0.00002, # Learning Rate for PMNet-v3 in the Building Missing scenario
            "tau": 0.05,   # Tau value for PMNet-v3 in the Building Missing scenario
        },
        "REM-NET+": {
            "LR": 0.0001,  # Learning Rate for REM-NET+ in the Building Missing scenario
            "tau": 0.4,    # Tau value for REM-NET+ in the Building Missing scenario
        },
    },
}
"""
Advantages:
- Clear structure, easy to understand and modify.
- You can easily add new scenarios and model configurations.
- Avoids the instantiation process when using classes, simplifying access.
- **Enables model-specific Learning Rate and Tau values within the TTA algorithm, allowing for fine-grained optimization of each model's performance.**
"""