#!/usr/bin/env python

# Author: Murad Abu-Khalaf, MIT CSAIL.

"""
    This loads the training data, and trains the view synthesizer neural network.

    It assumes a data set is available in the form of raw images and a text file for distances. 
    
    Two flags:
        REBUILD_DATA -- prepares a tensor from the training data set (default False)
        TRAIN_NN -- trains the neural network and replaces existing trained one (default False)
"""

#from __future__ import division
import os

import random
import matplotlib.pyplot as plt
import copy

###### Prepare Training Data from Datasets #######
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False
TRAIN_NN = False

class carImages():
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    PARENT_FOLDER = "CameraViewDistanceDataSet/TrainingDataSet"

    TOWN_FOLDERS = [    "Town03_A",
                        "Town03_B",
                        "Town04_A",
                        "Town04_B",
                        "Town04_C",
                        "Town04_D",
                        "Town05_A",
                        "Town05_B" ]

    def make_training_data(self):
        # Process Dataset
        for townFolder in self.TOWN_FOLDERS:
            DISTANCES = np.loadtxt(self.PARENT_FOLDER + '/' + townFolder + '/distances.txt')[:,1]
            LABELS = [f for f in os.listdir(self.PARENT_FOLDER + "/" + townFolder) if f.endswith('.png')]
            LABELS.sort()

            # Throw away half of the recordings.
            DISTANCES_obs_ = DISTANCES[0::2]
            Observed_ = LABELS[0::2]
            cond = np.logical_and(DISTANCES_obs_>5.5, DISTANCES_obs_<=40)
            indices = np.where(cond)[0].tolist()
            ObservedViews = [e for idx, e in enumerate(Observed_) if idx in indices]

            # Select desired spacings of 10, 20, and 30
            cond1 = np.logical_and(DISTANCES>9.9, DISTANCES<=10.4)
            cond2 = np.logical_and(DISTANCES>19.9, DISTANCES<=20.4)
            cond3 = np.logical_and(DISTANCES>29.9, DISTANCES<=30.4)
            cond = np.logical_or(np.logical_or(cond1,cond2),cond3)
            DISTANCES_ref = DISTANCES[cond]
            indices = np.where(cond)[0].tolist()
            LABELS_ref = [e for idx, e in enumerate(LABELS) if idx in indices]

            DISTANCES_LABELS = []
            DISTANCES_LABELS.extend([[DISTANCES_ref[i], LABELS_ref[i]] for i in range(len(DISTANCES_ref))])

            training_data = []

            for distance_label in tqdm(DISTANCES_LABELS):
                try:
                    distance = distance_label[0]
                    label = distance_label[1]
                    path = os.path.join(self.PARENT_FOLDER + "/" + townFolder, label)
                    img_label = cv2.imread(path, cv2.IMREAD_COLOR)   # HxWxC
                    img_label = cv2.resize(img_label, (self.IMG_WIDTH, self.IMG_HEIGHT))
                    img_label = img_label.transpose(2,0,1) # HxWxC ==> CxHxW
                    img_label = img_label[::-1,:,:]  # BGR ==> RGB

                    for obs in ObservedViews:
                        path = os.path.join(self.PARENT_FOLDER + "/" + townFolder, obs)
                        img_obs = cv2.imread(path, cv2.IMREAD_COLOR)   # HxWxC
                        img_obs = cv2.resize(img_obs, (self.IMG_WIDTH, self.IMG_HEIGHT))
                        img_obs = img_obs.transpose(2,0,1) # HxWxC ==> CxHxW
                        img_obs = img_obs[::-1,:,:]  # BGR ==> RGB
                        training_data.append([np.array(img_obs), copy.deepcopy(np.array([[[distance]]])), copy.deepcopy(np.array(img_label))])

                except Exception as e:
                    print(path)
                    print(e)
                    pass
                        
            np.save("training_data_tensor_" + townFolder + ".npy", training_data)

acarImages = carImages()
if REBUILD_DATA:
    acarImages.make_training_data()

training_data = []
for townFolder in acarImages.TOWN_FOLDERS:
    training_data_Town = np.load("training_data_tensor_" + townFolder + ".npy", allow_pickle=True)
    if len(training_data) == 0:
        training_data = training_data_Town
        continue
    training_data = np.concatenate((training_data, training_data_Town), axis=0)


###### Build Neural Network #######
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary    # torch-summary 1.4.1

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
    
        # Register these tensors so they get pushed to the device as needed
        self.register_buffer('d', torch.linspace(-1, 1, 128))
        meshx, meshy = torch.meshgrid((self.d, self.d))
        self.register_buffer('meshx', meshx.clone()) 
        self.register_buffer('meshy', meshy.clone())
        # Addresses a loading state_dict error: "Please clone() the tensor before performing the operation"

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(  in_channels=3,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=16,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=1024,
                        out_channels=2048,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(True),
            nn.Flatten())

        # Desired spacing
        #self.transform = nn.Sequential(
        #    nn.Linear(1,2),
        #    nn.ReLU(True),
        #    nn.Flatten())


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048+1,
                               out_channels=1024,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=512,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.Conv2d(  in_channels=16,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.Tanh())

    def forward(self, xrgb, xphi):
        #print(xrgb.size())
        #print(xphi.size())

        z1 = self.encoder(xrgb)
        #print(z1.size())

        # Desired spacing
        #z2 = self.transform(xphi)
        #print(z2.size())

        #z3 = torch.cat((z1, xphi.view(-1,1), xphi.view(-1,1)), 1)
        z3 = torch.cat((z1, xphi.view(-1,1)), 1)

        #print(z3.size())

        z3r = z3.view(-1, 2048+1, 1, 1)
        
        z4 = self.decoder(z3r)
        
        #print(z6.size())
        gridFlow = torch.transpose(torch.transpose(z4, 1, 3),1,2)
        gxy = torch.stack((self.meshy, self.meshx), 2)
        gxy = gxy.unsqueeze(0) # add batch dim
        gxy = torch.cat(z4.size()[0]*[gxy])
        #print(gridFlow)
        #print(gridFlow.size())
        sampled = F.grid_sample(xrgb, gxy+gridFlow, align_corners=False)
        #print(sampled.size())
        return sampled
    

######## Instantiate Neural Network #######
net = Net()
print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)
print("Cuda device count is", torch.cuda.device_count())
net.to(device)


######## Check Neural Network Dimensions #######
#summary(net, [(3, 128, 128), (1,)])
#net(torch.randn(5,3,128,128).to(device),torch.randn(5,1,1,1).to(device))
#quit()


###### Train Neural Network #######
import torch.optim as optim

img_feed = torch.Tensor([i[0] for i in training_data])
s = torch.Tensor([i[1] for i in training_data]).view(-1,1,1,1)
img_syn = torch.Tensor([i[2] for i in training_data])

_ , idx  = s[:,0,0,0].sort()
s_sorted_A = s[idx]

validation_ratio = 0.1    # Validation percentage
validation_size = int(len(s) * validation_ratio)
validation_size = 1   # override for now

train_img0 = img_feed[:-validation_size]
train_s = s[:-validation_size]
train_img = img_syn[:-validation_size]

observed_dist_target = torch.utils.data.TensorDataset(train_img0, train_s, train_img)

validate_img0 = img_feed[-validation_size:]
validate_s = s[-validation_size:]
validate_img = img_syn[-validation_size:]

#train_img0 = torch.cat((train_img0_A, train_img0_B), 0)
#train_s    = torch.cat((train_s_A, train_s_B), 0)
#train_img  = torch.cat((train_img_A, train_img_B), 0)

BATCH_SIZE = 64
EPOCHS     = 10000
loader = torch.utils.data.DataLoader(observed_dist_target, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, drop_last = True)

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #loss_function = nn.MSELoss(reduction = 'mean')
    #loss_function = nn.SmoothL1Loss(reduction = 'mean')
    loss_function = nn.L1Loss(reduction = 'mean')
    for epoch in range(EPOCHS):
        sumLoss=0
        batchCnt = 0
        #for i in tqdm(range(0, len(train_s), BATCH_SIZE)):
        for batch_img0, batch_s, batch_img  in tqdm(loader):
            #batch_img0 = train_img0[i:i + BATCH_SIZE].to(device)
            #batch_s = train_s[i:i + BATCH_SIZE].to(device)
            #batch_img = train_img[i:i + BATCH_SIZE].to(device)
            batch_img0, batch_s, batch_img = batch_img0.to(device), batch_s.to(device), batch_img.to(device)
            net.zero_grad()
            outputs = net(batch_img0, batch_s)

            loss = loss_function(outputs, batch_img)
            batchCnt +=1
            sumLoss += float(loss)
            epochLoss = float(sumLoss/(batchCnt))

            #print(str(loss))
            loss.backward()
            optimizer.step()
            #torch.cuda.empty_cache()
            
        print("Epoch: " + str(epoch) + ". Batch Loss:" + str(loss) + ". Average Batch Loss:" + str(epochLoss))


if TRAIN_NN:
    train(net)
    torch.save(net.state_dict(), 'viewSynthesizerNNet.pth')
    #torch.onnx.export(net,(torch.randn(1,3,150,200).to(device),torch.randn(1,1,1,1).to(device)),"distance2DynamicView1.onnx")
    #net(torch.tensor([[[[1.0121]]]]))
else:
    net.load_state_dict(torch.load('viewSynthesizerNNet.pth', map_location=device))
    net.eval()


###### Test Training Result #######

def test0():
    # Shows the training data only, no generated views or use of the NN
    #idx = random.randint(0, len(training_data))
    print("Training Dataset Size:" + str(len(training_data)))
    #print(train_s)
    for idx in range(len(training_data)):
        observed = training_data[idx][0]
        desired_spacing = training_data[idx][1][0,0,0]
        desired_view = training_data[idx][2]

        plt.figure(1, figsize=(9,5))
        plt.subplot(1,2,1)
        plt.title("Observed View")
        plt.imshow(observed.transpose(1,2,0), cmap="viridis")
        plt.show(block = False)

        plt.subplot(1,2,2)
        plt.title("Ground Truth View for " +  '{:4.2f}'.format(desired_spacing))
        plt.imshow(desired_view.transpose(1,2,0), cmap="viridis")
        plt.show(block = False)
        plt.pause(0.1)


def test1(net):
    # Generates a reference view: FIXED reference distance with FIXED camera view.
    idx_ref = 80    # Picking a desired spacing 
    idx = 747       # Picking an observation
    print(len(train_img0))
    print(len(training_data))
    observed = train_img0[idx].to('cpu').numpy()
    plt.figure(2, figsize=(9,5))
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title("Observed View")
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")

    generated = net(train_img0[idx:idx+1].to(device), train_s[idx_ref:idx_ref+1].to(device))
    generated = generated.to('cpu').detach().numpy()[0]
    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title("Generated View for " +  '{:4.2f}'.format(train_s[idx_ref:idx_ref+1].item()))
    plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")

    groundtruth = train_img[idx_ref].to('cpu').numpy()
    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title("Ground Truth View for " +  '{:4.2f}'.format(train_s[idx_ref:idx_ref+1].item()))
    plt.imshow(groundtruth.transpose(1,2,0)/255.0, cmap="viridis")

    plt.show(block = True)
    
    plt.figure(2,frameon=False)
    plt.imshow(observed.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('observed.png',bbox_inches='tight', pad_inches=0)

    plt.figure(2,frameon=False)
    plt.imshow(generated.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('generated.png',bbox_inches='tight', pad_inches=0)


def test2(net):
    # Generates a reference view: FIXED camera view with VARYING reference distance
    idx = 150       # Picking an observation
    camera_feed_ = train_img0[idx].to('cpu').numpy()
    plt.figure(3, figsize=(9,5))
    plt.ion()
    plt.subplot(1,2,1)
    plt.title("Observed View")
    plt.imshow(camera_feed_.transpose(1,2,0)/255.0, cmap="viridis")
    #plt.show(block = False)
    
    camera_feed = train_img0[idx:idx+1].to(device)
    plt.subplot(1,2,2)
    for idx_ref in tqdm(range(5,40,1)):
        img_hat = net(camera_feed, torch.Tensor([[[[idx_ref]]]]).to(device))
        img_hat = img_hat.to('cpu').detach().numpy()[0]
        plt.imshow(img_hat.transpose(1,2,0)/255.0, cmap="viridis")
        plt.title("Generated View for " + '{:4.2f}'.format(idx_ref))
        plt.pause(0.25)
        #plt.draw()

    input("Press [enter] to close.")

def test3(net):
    # Generates a reference view: FIXED reference distance with VARYING camera views
    idx_ref = 10        # Picking a desired spacing 

    fig = plt.figure(4, figsize=(9,5))
    plt.ion()
    sub1 = fig.add_subplot(1,2,1)
    sub1.set_title("Observed View")    
    sub2 = fig.add_subplot(1,2,2)
    sub2.set_title("Generated View for " + '{:4.2f}'.format(idx_ref))

    for idx in tqdm(range(0,len(train_img0),1)):
        camera_feed_ = train_img0[idx].to('cpu').numpy()
        camera_feed = train_img0[idx:idx+1].to(device)
        img_hat = net(camera_feed, torch.Tensor([[[[idx_ref]]]]).to(device))
        img_hat = img_hat.to('cpu').detach().numpy()[0]
        sub1.imshow(camera_feed_.transpose(1,2,0)/255.0, cmap="viridis")
        sub2.imshow(img_hat.transpose(1,2,0)/255.0, cmap="viridis")
        plt.pause(0.25)
        #plt.draw()

    input("Press [enter] to close.")


def testGeneralizationDataSet(net):
    # Generates a reference view: FIXED reference distance with FIXED camera view.

    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    CAMERA_IMAGES_FOLDER_Web = "CameraViewDistanceDataSet/GeneralizationDataSet"
    LABELS_Web = [f for f in os.listdir(CAMERA_IMAGES_FOLDER_Web) if not f.startswith('.')] # Use this to avoid hidden files
    LABELS_Web.sort()

    test_data = []
    for label in tqdm(LABELS_Web):
        try:
            path = os.path.join(CAMERA_IMAGES_FOLDER_Web, label)
            img = cv2.imread(path, cv2.IMREAD_COLOR)   # HxWxC
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.transpose(2,0,1) # HxWxC ==> CxHxW
            img = img[::-1,:,:]  # BGR ==> RGB
            test_data.append(np.array(img))
        except Exception as e:
            print(e)
            pass

    img_web = torch.Tensor([i for i in test_data])

    idx = 333
    camera_feed_ = img_web[idx].to('cpu').numpy()
    plt.figure(6, figsize=(9,5))
    plt.ion()
    plt.subplot(1,2,1)
    plt.title("Observed Camera Feed")
    plt.imshow(camera_feed_.transpose(1,2,0)/255.0, cmap="viridis")
    
    camera_feed = img_web[idx:idx+1].to(device)
    plt.subplot(1,2,2)
    for idx_ref in tqdm(range(10,21,10)):
        img_hat = net(camera_feed, torch.Tensor([[[[idx_ref]]]]).to(device))
        img_hat = img_hat.to('cpu').detach().numpy()[0]
        plt.imshow(img_hat.transpose(1,2,0)/255.0, cmap="viridis")
        plt.title("Generated Scene View for Spacing " + '{:4.2f}'.format(idx_ref))
        plt.pause(0.25)

    input("Press [enter] to close.")

    plt.figure(6,frameon=False)
    plt.imshow(camera_feed_.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('observed.png',bbox_inches='tight', pad_inches=0)

    plt.figure(6,frameon=False)
    plt.imshow(img_hat.transpose(1,2,0)/255.0, cmap="viridis")
    plt.axis("off")
    plt.savefig('generated.png',bbox_inches='tight', pad_inches=0)


def testDiff(net,yref,y):
    i2 = yref
    i1 = y
    plt.figure(4)
    plt.show(block = False)
    img1 = net(s_sorted_A[i1:(i1+1)].to(device))
    img1 = img1.to('cpu')
    img1 = img1.detach().numpy()[0]
    img2 = net(s_sorted_A[i2:(i2+1)].to(device))
    img2 = img2.to('cpu')
    img2 = img2.detach().numpy()[0]
    imgdiff = img2 - img1
    print(imgdiff)
    imgdiffabs = cv2.convertScaleAbs(imgdiff)
    plt.imshow(imgdiffabs.transpose(1,2,0)/255.0, cmap="viridis")
    plt.draw()
    K = np.zeros((3,150,200))
    K[:,:,75:125] = 1
    u = np.sum(K*(imgdiff)/255.0)
    return img1, img2, u



#test0()
#test1(net)
#test2(net)
#test3(net)
#testGeneralizationDataSet(net)