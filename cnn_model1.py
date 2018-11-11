################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Carl Tan, Robert Eaton, & Amanda Smith
#
# Filename: cnn_model1.py
# 
# Description: 
# 
# This file contains the code for the first CNN model,
# corresponding to Part 2, No. 3 in the assignment 
################################################################################


# PyTorch and neural network imports
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init

class Model1CNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    Consists of three Conv2d layers, followed by one 4x4 max-pooling layer, 
    and 2 fully-connected (FC) layers:
    
    conv1 -> conv2 -> conv3 -> maxpool -> fc1 -> fc2 (outputs)
    
    Make note: 
    - Inputs are expected to be grayscale images (how many channels does this imply?)
    - The Conv2d layer uses a stride of 1 and 0 padding by default
    """
    
    def __init__(self, image_size, channels, kernels, block_size=3, pool_kernel=2, pool_stride=2):
        super(Model1CNN, self).__init__()
        
        self.conv = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        self.block_size = block_size
        
        final_image_size = image_size
        
        # convolution layers
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            k = kernels[i]
                        
            self.conv.append(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k))
            self.conv_norm.append(nn.BatchNorm2d(out_c))
            torch_init.xavier_normal_(self.conv[i].weight)
            
            # calculates the image size after going through this layer
            final_image_size = final_image_size - k + 1
            if (i + 1) % block_size == 0:
                final_image_size = (final_image_size - pool_kernel) // pool_stride + 1
                       
        # max pool layer
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        
        # fully connected layers
        self.fc1 = nn.Linear(in_features=(channels[-1] * (final_image_size ** 2)), out_features=4096)
        torch_init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=4096, out_features=14).cuda()
        torch_init.xavier_normal_(self.fc2.weight)


    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying 
        non-linearities after each layer.
        
        Note that this function *needs* to be called "forward" for PyTorch to 
        automagically perform the forward pass. 
        ••••••••
        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """
                             
        for i in range(len(self.conv)):
            batch = func.relu(self.conv_norm[i](self.conv[i](batch)))
                             
            if (i + 1) % self.block_size == 0:
                batch = self.pool(batch)
                             
        
        batch = batch.view(-1, self.num_flat_features(batch))
        batch = func.relu(self.fc1(batch)) 
        batch = self.fc2(batch)
                     
        # Return the class predictions
        # Apply an activition function to 'batch'
        return torch.sigmoid(batch)
    
    

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

