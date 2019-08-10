# Car-Sketch-to-Color-Pix2Pix
A Pytorch implementation of Outline sketches to color images generation using Pix2Pix.

The dataset used is from shapenet. Shapenet contains images of objects in many orientation. For now, only the car dataset has been used
The data generation process is quite complicated and hence I have made it simpler to use by uploading the dataset I created to google drive

Download the data set from https://drive.google.com/open?id=1Efu8cF3itOMW96xrub5Z6aFVan3pf4bp

## Code Organization
image_loader.py loads the image.
model.py defines the pix2pix architecture.
train.py trains the model.
Run on any image using colorize.py

The model was trained on aws p2 instance. If you want to train on the cpu just make use_gpu=False.



