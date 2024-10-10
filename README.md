### Project: ImageAuctus

GAN-powered Image Augmentation: Enhancing Data with Synthetic Visuals

### Overview

ImageAuctus is a deep learning project focused on data augmentation using Generative Adversarial Networks (GANs). This implementation leverages the CIFAR-10 dataset to generate synthetic images that resemble the original dataset, enhancing the diversity and quantity of training data for various machine learning tasks.

### Features

- **Data Augmentation:** Generate new images that closely resemble original images in the CIFAR-10 dataset.
- **GAN Architecture:** Utilizes a combination of a generator and discriminator to improve the quality of generated images.
- **Batch Normalization:** Incorporates batch normalization layers to stabilize training and improve convergence.
- **Visualization:** Visualize original and generated images during training to monitor the GAN's progress.

## Code Explanation

- **Data Loading and Preprocessing**: The CIFAR-10 dataset is loaded and normalized to a range of [-1, 1].
- **Model Architecture**:
  - **Generator**: A neural network that generates images from random noise. It consists of several layers, including Dense, Reshape, and Conv2DTranspose, with LeakyReLU activation functions and Batch Normalization for stability.
  - **Discriminator**: A neural network that classifies images as real or fake. It consists of Dense layers with LeakyReLU activations and Dropout for regularization.
- **Training Loop**: The GAN is trained for a specified number of epochs. In each epoch, the discriminator is trained on both real and generated images, and the generator is trained to improve its ability to create realistic images.

## Project Structure

```
ImageAuctus/
├── GANaugmentation.py  # Main script to train the GAN and generate images
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
```

### Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MohanVamsi183/ImageAuctus.git
   cd ImageAuctus
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the GAN augmentation script:

   ```bash
   python GANaugmentation.py
   ```

2. The script will load the CIFAR-10 dataset, train the GAN model, and generate synthetic images. Generated images will be displayed periodically during training.
   (Increase the number of epochs for better data augmentation)

3. Monitor generated images and training progress as specified in the code.

### Acknowledgments

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) for providing a benchmark dataset.
- TensorFlow for the powerful deep learning framework.
