# Binary Tree Recognizer using Deep Learning

This repository contains an implementation of a deep learning model for recognizing and converting hand-drawn binary tree images into data structures. The model is trained on a custom dataset of synthetically generated noisy binary tree images, and it learns to predict the row, column, and radius of the circle representing each node in the tree.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Generation](#dataset-generation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to explore the capabilities of deep learning in recognizing and reconstructing binary tree structures from hand-drawn images. The motivation behind this project is to provide a practical application of computer vision and deep learning techniques in the domain of data structure visualization and understanding.

The project consists of the following main components:

1. **Dataset Generation**: A script (`dataset.py`) to generate a synthetic dataset of noisy binary tree images and their corresponding labels (row, column, and radius of each node).
2. **Model Architecture**: A deep convolutional neural network (`network.py`) designed to take binary tree images as input and predict the parameters of the circles representing each node.
3. **Training**: A script (`train.py`) to train the model on the generated dataset.
4. **Evaluation**: A script (`validation.py`) to evaluate the trained model's performance on a separate validation set.
5. **Inference**: A script (`main.py`) to perform inference on new hand-drawn binary tree images using the trained model.

## Dataset Generation

The synthetic dataset is generated using the `dataset.py` script. The script creates a CSV file (`train_set.csv`) containing the paths to the generated images and their corresponding labels. The images are stored in the `datasets/train/` directory.

The dataset generation process involves the following steps:

1. Create a blank image of a fixed size (e.g., 200x200 pixels).
2. Draw a circle representing the root node at a random position with a random radius within a specified range.
3. Add random noise to the image to simulate hand-drawn imperfections.
4. Save the image and its corresponding label (row, column, and radius of the root node) to the dataset.

The level of noise and other parameters can be adjusted in the `dataset.py` script.

## Model Architecture

The deep learning model used in this project is a convolutional neural network (CNN) implemented in `network.py`. The model takes a binary tree image as input and predicts the row, column, and radius of the circle representing each node.

The CNN architecture consists of the following layers:

1. Convolutional layers with batch normalization and ReLU activation
2. Max pooling layers for spatial downsampling
3. Fully connected layers
4. Output layer with three units representing the row, column, and radius predictions

The model is trained using a combination of Mean Squared Error (MSE) and L1 loss functions to optimize the predictions.

## Usage

### Training

To train the model, run the `train.py` script with the desired configuration parameters. The script supports various command-line arguments, including:

- `--name`: Name of the experiment (default: `'v8'`)
- `--out_file`: Path to the output features file (default: `'new_out.txt'`)
- `-j`, `--workers`: Number of data loading workers (default: `8`)
- `-b`, `--batch_size`: Batch size for training (default: `256`)
- `--lr`, `--learning-rate`: Initial learning rate (default: `0.001`)
- `--resume`: Path to a checkpoint to resume training from (default: `''`)
- `--data`: Path to the image list file (default: `'train_set.csv'`)
- `--print_freq`: Print frequency during training (default: `50`)
- `--epochs`: Number of total epochs to run (default: `51`)
- `--start_epoch`: Manual epoch number to start from (default: `0`)
- `--save_freq`: Number of epochs to save the model after (default: `5`)

Example usage:

```
python train.py --name experiment_1 --epochs 100 --lr 0.0005
```

### Evaluation

The `validation.py` script is used to evaluate the trained model's performance on a validation set. It generates a set of noisy binary tree images, runs inference using the trained model, and calculates the Intersection over Union (IoU) between the predicted and ground truth circles for each node.

To run the evaluation script, execute:

```
python validation.py
```

The script will print the mean IoU over the validation set.

### Inference

The `main.py` script provides an example of how to use the trained model for inference on new hand-drawn binary tree images. It generates a small set of noisy binary tree images, runs inference using the trained model, and prints the predicted parameters (row, column, and radius) for each node.

To run the inference script, execute:

```
python main.py
```

## Results

The performance of the trained model can be evaluated using metrics such as the mean IoU or accuracy. The exact performance will depend on various factors, including the model architecture, training parameters, and the complexity of the input images.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).