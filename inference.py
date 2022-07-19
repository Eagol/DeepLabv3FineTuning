import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from os import listdir

from PIL import Image

# Load the trained model
model = torch.load('./output/weights.pt')
# Set the model to evaluate mode
model.eval()

pathImagesVal = './dataset/Val/Image'
# pathLabelsVal = './dataset/Test/Mask'

for fileName in listdir(pathImagesVal):

    print('Processando *****', fileName)

    baseFileName = fileName.split('.')[0]

    # Read  a sample image and mask from the data-set

    img = cv2.imread(f'{pathImagesVal}/{fileName}')

    h, w, c = img.shape
    
    img = cv2.imread(
        f'{pathImagesVal}/{fileName}').transpose(2, 0, 1).reshape(1, 3, w, h)
    # mask = cv2.imread(f'{pathLabelsVal}/{baseFileName}_label.png')
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)

    # Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
    plt.hist(a['out'].data.cpu().numpy().flatten())

    # Plot the input image, ground truth and the predicted output
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(img[0, ...].transpose(1, 2, 0))
    plt.title('Image')
    plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(mask)
    # plt.title('Ground Truth')
    # plt.axis('off')
    plt.subplot(133)
    plt.imshow(a['out'].cpu().detach().numpy()[0][0] > 0.2)
    plt.title('Segmentation Output')
    plt.axis('off')
    plt.savefig(f'./output/{baseFileName}_output.png', bbox_inches='tight')
