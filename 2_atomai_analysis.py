
import atomai as aoi
import numpy as np
import torch
import matplotlib.pyplot as plt


# download_link1 = 'https://drive.google.com/uc?id=1-4-IQ71m--OelQb1891GnbG1Ako1-DKh'
# download_link2 = 'https://drive.google.com/uc?id=18JK9GcMPMWHmHtwArujVQRLr6N4VIM_j'
# !gdown -q $download_link1 -O 'training_data.npy'
# !gdown -q $download_link2 -O 'validation_data.npy'

# Load train/test data (this is a simple dataset generated just from a single image)
# dataset = np.load('training_data.npy')
# images = dataset['X_train']
# labels = dataset['y_train']
# images_test = dataset['X_test']
# labels_test = dataset['y_test']
# Load validation data (data from different experiment)
expdata = np.load("data/validation_data.npy")

# print(images.shape, labels.shape)
# print(images_test.shape, labels_test.shape)

# model = aoi.models.Segmentor(nb_classes=3)
model = torch.load('model.pth')

print(model.net)

nn_output, coordinates = model.predict(expdata)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12))
ax1.imshow(expdata, cmap='gray', origin="lower")
ax2.imshow(nn_output.squeeze(), origin="lower")

plt.savefig('output_image.png') 

aoi.utils.plot_coord(expdata, coordinates[0], fsize=12)


# TODO: Will need to fix everything here and below. 
# imstack = aoi.stat.imlocal(nn_output, coordinates, window_size=32, coord_class=1)

# imstack.pca_scree_plot(plot_results=True);
# pca_results = imstack.imblock_pca(4, plot_results=True)
# ica_results = imstack.imblock_ica(4, plot_results=True)
# nmf_results = imstack.imblock_nmf(4, plot_results=True)
