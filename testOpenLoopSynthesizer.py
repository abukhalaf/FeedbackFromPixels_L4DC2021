#!/usr/bin/env python

# Author: Murad Abu-Khalaf, MIT CSAIL.

"""
    Open-loop testing of the Synthesizer's ability to generate reference views.

"""

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np


import trainViewSynthesizerNNet

training_data =  trainViewSynthesizerNNet.getNumPyTrainingData()
net = trainViewSynthesizerNNet.net
device = trainViewSynthesizerNNet.device

###### Define the Test Methods #######

def showTrainingData():
    """
    Plots observed Views to serve as input to the synthesizer
    along with observed views at desired distances that will
    serve as ground truth for the output of the Synthesizer
    during training.

    This simply shows the training data. All views are Camera views, and
    non are synthesizer views.
    """
    print("Training Dataset Size:" + str(len(training_data)))
    for idx in range(len(training_data)):
        observed = training_data[idx][0]
        desired_spacing = training_data[idx][1][0,0,0]
        desired_view = training_data[idx][2]

        plt.figure(1, figsize=(9,5))
        plt.subplot(1,2,1)
        plt.title("Observed View (Input to Synthesizer)")
        plt.imshow(observed.transpose(1,2,0), cmap="viridis")
        plt.show(block = False)

        plt.subplot(1,2,2)
        plt.title("Ground Truth: Observed View for " +  '{:4.2f}'.format(desired_spacing))
        plt.imshow(desired_view.transpose(1,2,0), cmap="viridis")
        plt.show(block = False)
        plt.pause(0.1)


def generateReferenceViewFromObservation():
    """ 
    Generates a reference view for a FIXED reference distance and FIXED camera view.
    """
    idx = 747       # Picking an observation
    observed = training_data[idx][0]
    spacing  = training_data[idx][1]
    groundtruth = training_data[idx][2]

    plt.figure(2, figsize=(9,5))
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title("Observed View")
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")

    generated = net(torch.Tensor([observed]).to(device), torch.Tensor(spacing).to(device))
    generated = generated.to('cpu').detach().numpy()[0]
    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title("Generated View for " +  '{:4.2f}'.format(spacing.item()))
    plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title("Ground Truth View for " +  '{:4.2f}'.format(spacing.item()))
    plt.imshow(groundtruth.transpose(1,2,0)/255.0, cmap="viridis")

    plt.show(block = True)
    
    # Save the generated view for publication purposes
    plt.figure(2,frameon=False)
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('observed.png',bbox_inches='tight', pad_inches=0)

    plt.figure(2,frameon=False)
    plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('generated.png',bbox_inches='tight', pad_inches=0)


def generateReferenceViewsFromObservation():
    """ 
    Generates reference views for a VARYING reference distance and FIXED camera view.
    """
    idx = 150       # Picking an observation
    observed = training_data[idx][0]
    plt.figure(3, figsize=(9,5))
    plt.ion()
    plt.subplot(1,2,1)
    plt.title("Observed View")
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")
    
    plt.subplot(1,2,2)
    for s in tqdm(range(10,31,10)):
        generated = net(torch.Tensor([observed]).to(device), torch.Tensor([[[[s]]]]).to(device))
        generated = generated.to('cpu').detach().numpy()[0]
        plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")
        plt.title("Generated View for " + '{:4.2f}'.format(s))
        plt.pause(1.00)
        #plt.draw()

    input("Press [enter] to close.")

def generateReferenceViewFromObservations():
    """ 
    Generates a reference view for a FIXED reference distance and VARYING camera views.
    """
    s = 10        # Picking a desired spacing 

    fig = plt.figure(4, figsize=(9,5))
    plt.ion()
    sub1 = fig.add_subplot(1,2,1)
    sub1.set_title("Observed View")    
    sub2 = fig.add_subplot(1,2,2)
    sub2.set_title("Generated View for " + '{:4.2f}'.format(s))

    for idx in tqdm(range(0,len(training_data),1)):
        observed = training_data[idx][0]
        generated = net(torch.Tensor([observed]).to(device), torch.Tensor([[[[s]]]]).to(device))
        generated = generated.to('cpu').detach().numpy()[0]
        sub1.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")
        sub2.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")
        plt.pause(0.25)
        #plt.draw()

    input("Press [enter] to close.")


def testGeneralizationDataSet():
    """ 
    Generates reference views for a VARYING reference distance and Fixed camera view.
    """

    # Create NumPy tensors from images
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    TestingFolder = "CameraViewDistanceDataSet/TestingDataSet"
    LABELS = [f for f in os.listdir(TestingFolder) if not f.startswith('.')] # Use this to avoid hidden files
    LABELS.sort()

    test_data = []
    for label in tqdm(LABELS):
        try:
            path = os.path.join(TestingFolder, label)
            img = cv2.imread(path, cv2.IMREAD_COLOR)   # HxWxC
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.transpose(2,0,1) # HxWxC ==> CxHxW
            img = img[::-1,:,:]  # BGR ==> RGB
            test_data.append(np.array(img))
        except Exception as e:
            print(e)
            pass

    # Generate a reference view from an observation
    idx = 333       # Picking an observation
    observed = test_data[idx]
    plt.figure(6, figsize=(9,5))
    plt.ion()
    plt.subplot(1,2,1)
    plt.title("Observed Camera Feed")
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")
    
    plt.subplot(1,2,2)
    for s in tqdm(range(10,31,10)):
        generated = net(torch.Tensor([observed]).to(device), torch.Tensor([[[[s]]]]).to(device))
        generated = generated.to('cpu').detach().numpy()[0]
        plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")
        plt.title("Generated Scene View for Spacing " + '{:4.2f}'.format(s))
        plt.pause(1.00)

    input("Press [enter] to close.")

    # Save the generated view for publication purposes
    plt.figure(6,frameon=False)
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('observed.png',bbox_inches='tight', pad_inches=0)

    plt.figure(6,frameon=False)
    plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('generated.png',bbox_inches='tight', pad_inches=0)


###### Choose the desired test by speciying a number #######

testID = 4

if testID == 0:
    showTrainingData()
elif testID == 1:
    generateReferenceViewFromObservation()
elif testID == 2:
    generateReferenceViewsFromObservation()
elif testID == 3:
    generateReferenceViewFromObservations()
elif testID == 4:
    testGeneralizationDataSet()
