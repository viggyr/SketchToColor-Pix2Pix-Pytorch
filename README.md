# Car-Sketch-to-Color-Pix2Pix
A Pytorch implementation of Sketch to color images generation using CGAN

The dataset used is from shapenet. Shapenet contains images of objects in many orientation. For now, only the car dataset has been used
The data generation process is quite complicated and hence I have made it simpler to use by uploading the dataset I created to my drive

Download the data set from https://drive.google.com/open?id=1Efu8cF3itOMW96xrub5Z6aFVan3pf4bp

The image loading is done by image_loader.py
The model definition is in model.py
The main file that trains the model is run_sketch2color.py

I have trained the model in aws p2 instance. If you want to train in cpu, remove .cuda() from wherever it has been used and also remove .cpu() inside visdom.images call.


