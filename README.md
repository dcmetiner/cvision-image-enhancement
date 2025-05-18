This study consists of two models based on Deep-Learning: Zero-DCE and Retinex-Net

zero-dce.ipynb

- This file implements the Zero-DCE model for low-light image enhancement, with modifications to test different loss functions and datasets. The implementation is based on the paper:

Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (arXiv:2001.06826)

Features

- Implements Zero-DCE using TensorFlow/Keras.
- Utilizes LOL Dataset for training and evaluation.
- Introduces two additional augmented datasets to test model performance.
- Modifies the original Zero-DCE loss function with KL-Divergence Loss for experimentation.
- Integrates Weights & Biases (wandb) for experiment tracking.

Dataset

- This project uses the LOL Dataset https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

Dataset structure:

zero-dce-enhancement/
│── lol_dataset/
│   ├── our485/
│   │   ├── low/  (Low-light images)
│   │   ├── high/ (Normal-light images)

Place your dataset inside the lol_dataset/ directory before running the training script.

Usage:

IMAGE_SIZE = 256
MAX_TRAIN_IMAGES = 400
BATCH_SIZE = 16
TRAIN_VAL_IMAGE_DIR = "./lol_dataset/our485/low"
TEST_IMAGE_DIR = "./lol_dataset/eval15/low"
LEARNING_RATE = 1e-4
LOG_INTERVALS = 10
EPOCHS = 60

Play with these parameters to change your dataset, log interval or the number of epochs you want to train the model. Augmentation of the dataset is done automatically within the file without the need for any prior work.

Testing with different images:

- Test with images of your choice using the trained model following a similar structure:

test_images = ["test1.jpg", "test2.jpg"]

results = []

for img_path in test_images:
    enhanced_zero_dce = zero_dce_model(original)

retinex_net.ipynb

- This file contains an implementation of Retinex-Net for low-light image enhancement, based on deep learning techniques. The model follows a two-stage process involving a decomposition network and an enhancement network, as originally proposed in the Retinex theory.

Features

- Implements Retinex-Net for low-light image enhancement.
- Trains and evaluates the model on the LOL-dataset.
- Extends the model with additional augmentations and improvements.

Dataset

- This project uses the LOL Dataset https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

Dataset structure:

zero-dce-enhancement/
│── lol_dataset/
│   ├── our485/
│   │   ├── low/  (Low-light images)
│   │   ├── high/ (Normal-light images)

Place your dataset inside the lol_dataset/ directory before running the training script.

Usage:

IMAGE_SIZE = 256
MAX_TRAIN_IMAGES = 400
BATCH_SIZE = 16
TRAIN_VAL_IMAGE_DIR = "./lol_dataset/our485/low"
TEST_IMAGE_DIR = "./lol_dataset/eval15/low"
LEARNING_RATE = 1e-4
LOG_INTERVALS = 10
EPOCHS = 60

Play with these parameters to change your dataset, log interval or the number of epochs you want to train the model. Augmentation and separation of the dataset into two components are done conducted within the file without the need for any prior work.

Testing with different images:

- Test with images of your choice :

test_images = ["test1.jpg", "test2.jpg"]

results = []

for img_path in test_images:
    enhanced_retinex_net = retinex_net(original)
