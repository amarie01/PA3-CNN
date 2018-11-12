################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Carl Tan, Robert Eaton, & Amanda Smith
#
# Filename: cnn_model2.py
# 
# Description: 
# 
# This file contains the starter code for the second CNN model,
# corresponding to Part 2, No. 3 in the assignment 
################################################################################


# PyTorch and neural network imports
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init

class Model2CNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    Consists of three Conv2d layers, followed by one 4x4 max-pooling layer, 
    and 2 fully-connected (FC) layers:
    
    conv1 -> conv2 -> conv3 -> maxpool -> fc1 -> fc2 (outputs)
    
    Make note: 
    - Inputs are expected to be grayscale images (how many channels does this imply?)
    - The Conv2d layer uses a stride of 1 and 0 padding by default
    """
    
    def __init__(self, skip_sigmoid=False):
        super(Model2CNN, self).__init__()
        
        self.skip_sigmoid = skip_sigmoid
        
        #---- First Block ------------------------------------------------------# 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        #self.conv1_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv2_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        #self.conv3_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv3.weight)
        
        #---- Second Block -----------------------------------------------------# 
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        #self.conv4_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv4.weight)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv5_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv5.weight)

        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        #self.conv6_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv6.weight)

        #---- Third Block -----------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        #self.conv7_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv7.weight)

        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv8_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv8.weight)

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        #self.conv9_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv9.weight)
        
        #---- Fourth Block -----------------------------------------------------#
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        #self.conv10_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv10.weight)
        
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv11_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv11.weight)
        
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        #self.conv12_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv12.weight)
        
        #---- END Blocks -------------------------------------------------------# 
        
        # apply max-pooling with a [2x2] kernel using tiling (*NO SLIDING WINDOW*)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected layers
        self.fc1 = nn.Linear(in_features=(8192), out_features=128)
        torch_init.xavier_normal_(self.fc1.weight)

        # out_features = # of possible diseases
        self.fc2 = nn.Linear(in_features=128, out_features=14).cuda()
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
        
        # Apply first convolution, followed by ReLU non-linearity; 
        
        # use batch-normalization on its outputs'
        batch = self.conv1(batch)
        temp = batch
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = self.conv3(batch)
        batch += temp
        batch = func.relu(self.pool(batch))
        
        # Second block
        batch = self.conv4(batch)
        temp = batch
        batch = func.relu(self.conv5_normed(self.conv5(batch)))
        batch = self.conv6(batch)
        batch += temp
        batch = func.relu(self.pool(batch))
        
        # Third block
        batch = self.conv7(batch)
        temp = batch
        batch = func.relu(self.conv8_normed(self.conv8(batch)))
        batch = self.conv9(batch)
        batch += temp
        batch = func.relu(self.pool(batch))
        
        # Fourth block
        batch = self.conv10(batch)
        temp = batch
        batch = func.relu(self.conv11_normed(self.conv11(batch)))
        batch = self.conv12(batch)
        batch += temp
        batch = func.relu(self.pool(batch))

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))
        
        # Connect the reshaped features of the pooled conv3 to fc1
        batch = func.relu(self.fc1(batch)) 
                                                                            
        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        batch = self.fc2(batch)
        
        if self.skip_sigmoid:
            return batch

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

