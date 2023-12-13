# CSCI467FinalProject
Final Project for CSCI467 -- cell segmentation

baseline_dapi algorithm is a basic thresholding algorithm

kmeans is a mini-batch k-means algorithm

machinealg is a convolutional neural network algorithm


Dataset: https://deepseas.org/datasets/
Note: The DeepSea dataset has a lot of images, but not all of them have a correct mask pair, so I only used the images with a paired mask (so my training dataset was smaller than what they provided). 

Commands to run for baseline thresholding algorithm, k-means algorithm, and CNN algorithm (respectively) -- Note: for testing with the dev sets I manually changed my code to do so: 

python baseline_dapi.py

python kmeans.py

python machinealg.py
