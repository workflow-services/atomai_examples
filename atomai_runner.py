# -*- coding: utf-8 -*-
"""AtomicSemanticSegmention.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/AtomicSemanticSegmention.ipynb

# Semantic segmentation-based analysis of atomic images

Author: Maxim Ziatdinov

Email: ziatdinovmax@gmail.com

---


This notebook shows i) how to apply a UNet-like neural network for semantic segmentation of atomic images and to perform; ii) how to apply multivariate statistical analysis to the semantically-segmented output.


---

Install AtomAI:
"""


"""Imports:"""

import atomai as aoi
import numpy as np
import matplotlib.pyplot as plt
import torch

"""## Semantic segmentation

Download training/test data:
"""
# mkdir data && cd data
# wget -q 'https://www.dropbox.com/s/q961z099trfqioj/ferroics-exp-small-m.npz?dl=0' -O 'training_data.npy'
# wget -q 'https://www.dropbox.com/s/w7j12xchjd0at77/lbfo_expdata.npy?dl=0' -O 'validation_data.npy'
# cd ..

# Load train/test data (this is a simple dataset generated just from a single image)
dataset = np.load("data/training_data.npy")
images = dataset["X_train"]
labels = dataset["y_train"]
images_test = dataset["X_test"]
labels_test = dataset["y_test"]
# Load validation data (data from different experiment)
expdata = np.load("data/validation_data.npy")

print(images.shape, labels.shape)
print(images_test.shape, labels_test.shape)

"""The training data was generated from a single labeled experimental STEM image of Sm-doped BFO containing ~20,000 atomic unit cells (see arXiv:2002.04245 and the associated notebook). The "mother image" was ~3000 px x 3000 px. We randomly cropped about 2000 image-masks pairs of 256 x 256 resolution and then applied different image "distortions" (noise, blurring, zoom, etc.) to each cropped image, treating two atomic sublattices as two different classes.

The training/test images and masks represent 4 separate numpy arrays with the dimensions (n_images, n_channels=1, image_height, image_width) for training/test images, and (n_images, image_height, image_width) for the associated masks (aka ground truth). The reason that our images have 4 dimensions, while our labels have only 3 dimensions is because of how the cross-entropy loss is calculated in PyTorch (see [here](https://pytorch.org/docs/stable/nn.html#nllloss)). Briefly, if you have multiple channels corresponding to different classes in your labeled data, you'll need to map your target classes to tensor indices. Here, we already did this for our training and test data during the preparation stage and so everything is ready for training.

Let's plot some of the training data:
"""

n = 5  # number of images to plot

n = n + 1
fig = plt.figure(figsize=(30, 8))
for i in range(1, n):
    ax = fig.add_subplot(2, n, i)
    ax.imshow(images[i - 1, 0, :, :], cmap="gray")
    ax.set_title("Augmented image " + str(i))
    ax.grid(alpha=0.5)
    ax = fig.add_subplot(2, n, i + n)
    ax.imshow(labels[i - 1], interpolation="Gaussian", cmap="jet")
    ax.set_title("Ground truth " + str(i))
    ax.grid(alpha=0.75)

"""The neural network will be trained to take the images in the top row (see plot above) as the input and to output clean images of circular-shaped "blobs" on a uniform background in the bottom row, from which one can identify the xy centers of atoms.

We are going to use a [UNet](https://en.wikipedia.org/wiki/U-Net)-like neural network for semantic segmentation. In the semantic segmentation tasks we aim at categorizing every pixel in the image. This is different form a regular image-level classification tasks, where we categorize the image as whole (e.g. this image has a cat/dog, etc.). Here's a nice illustration from this [free book](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf):

<img src="https://drive.google.com/uc?export=view&id=18N4x3P0whH91OcpBOOkDprgWVo-36i34" width=800 px><br><br>

Initialize a nodel for semantic segmentation:
"""

model = aoi.models.Segmentor(nb_classes=3)

"""We can also "print" the neural network:"""

print(model.net)

"""Train the initialized model:

(Here the accuracy is calculated as [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index) (IoU) score, which is the area of overlap between the predicted segmentation and the ground truth divided by the area of union between them)
"""

model.fit(
    images,
    labels,
    images_test,
    labels_test,  # training data
    training_cycles=1,
    compute_accuracy=True,
    swa=False,  # training parameters
)

print("Finished training")

torch.save(model, "model.pth")
print("Saved model.pth")
