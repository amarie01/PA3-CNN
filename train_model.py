from baseline_cnn import *
from baseline_cnn import BasicCNN
from cnn_model1 import Model1CNN
from cnn_model2 import Model2CNN

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import torch
import torch.nn as nn
import numpy as np
import time

def accuracy(predictions, labels):
    return np.sum(predictions == labels) / float(labels.size)

def precision(predictions, labels):
    TP = np.sum(np.logical_and(predictions == labels, labels == 1))
    FP = np.sum(np.logical_and(predictions != labels, predictions == 1))
    return TP / float(FP + TP + 1)

def recall(predictions, labels):
    TP = np.sum(np.logical_and(predictions == labels, labels == 1))
    FN = np.sum(np.logical_and(predictions != labels, predictions == 0))
    return TP / float(FN + TP + 1)

def BCR(predictions, labels):
    return (precision(predictions, labels) + recall(predictions, labels)) / 2.0

def accuracy_per_class(predictions, labels):
    return np.sum(predictions == labels, axis=0) / float(labels.shape[0])

def precision_per_class(predictions, labels):
    TP = np.sum(np.logical_and(predictions == labels, labels == 1), axis=0)
    FP = np.sum(np.logical_and(predictions != labels, predictions == 1), axis=0)
    return TP / np.asfarray(FP + TP + 1)

def recall_per_class(predictions, labels):
    TP = np.sum(np.logical_and(predictions == labels, labels == 1), axis=0)
    FN = np.sum(np.logical_and(predictions != labels, predictions == 0), axis=0)
    return TP / np.asfarray(FN + TP + 1)

def BCR_per_class(predictions, labels):
    return (precision_per_class(predictions, labels) + recall_per_class(predictions, labels)) / 2.0

def confusion_matrix(mtx, predictions, actuals): 
    for p,a in zip(predictions, actuals):
        
        for i in range(p.shape[0]):
            # If TP, add 1 to diagonal
            # Then discard the other outputs
            if p[i] == 1 and a[i] == 1:
                mtx[i][i] += 1
                
            elif p[i] == 1:
                mtx[i] += a
    
    return mtx

def print_scores(batch_start, batch_count, accuracies, precisions, recalls, BCRs, aggregate=True):
    if aggregate:
        acc = np.mean(accuracies[batch_start:])
        pre = np.mean(precisions[batch_start:])
        rec = np.mean(recalls[batch_start:])
        bcr = np.mean(BCRs[batch_start:])
    else:
        acc = np.mean(accuracies[batch_start:], axis=0)
        pre = np.mean(precisions[batch_start:], axis=0)
        rec = np.mean(recalls[batch_start:], axis=0)
        bcr = np.mean(BCRs[batch_start:], axis=0)
            
    # Print the loss averaged over the last N mini-batches    
    print('Minibatch ' + str(batch_count) + ' accuracy: ' + str(acc))
    print('Minibatch ' + str(batch_count) + ' precision: ' + str(pre))
    print('Minibatch ' + str(batch_count) + ' recall: ' + str(rec))
    print('Minibatch ' + str(batch_count) + ' bcr: ' + str(bcr))

def train_model(run, model, model_name, num_epochs, train_loader, computing_device, optimizer, criterion):
    # Track the loss across training
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    
    start = time.time()
    
    # When to record results
    N_training_loss = len(train_loader) // 4
    N_minibatch_print = 50
    
    min_val_loss = 100    
    
    for epoch in range(num_epochs):
        
        # Training
        minibatch_loss_for_print = 0.0
        minibatch_loss_for_results = 0.0
        minibatch_acc_for_results = 0.0
        
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Performs sigmoid on the output for when the criterion is BCEWithLogitsLogs
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                outputs = torch.sigmoid(outputs)
                
            predicted = torch.round(outputs.data)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()
            acc = accuracy(predicted, labels)

            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add this iteration's loss to the total minibatch loss
            minibatch_loss_for_print += loss
            minibatch_loss_for_results += loss
            minibatch_acc_for_results += acc
            
            # Save the training loss every quarter of the training data
            if (minibatch_count + 1) % N_training_loss == 0:
                train_loss.append(minibatch_loss_for_results / N_training_loss)
                minibatch_loss_for_results = 0.0
                train_accuracy.append(minibatch_acc_for_results / N_training_loss)
                minibatch_acc_for_results = 0.0

            # Print the training loss every N_minibatches
            if (minibatch_count + 1) % N_minibatch_print == 0:   
                print('Run %d, Epoch %d, average minibatch %d loss: %.3f' %
                    (run, epoch + 1, minibatch_count + 1, minibatch_loss_for_print / N_minibatch_print))
                print(100 * minibatch_count/len(train_loader),"% done, " ,time.time() - start, " Seconds elapsed")
                
                minibatch_loss_for_print = 0.0
                
            #if len(train_accuracy) >= 2:
                #break

                
        # Validation
        val_loss = 0.0
        val_acc = 0.0
        early_stop_count = 0 
        
        # When to record results
        N_valid_loss = len(val_loader) // 4
        
        with torch.no_grad():
            for batch_count, (images, labels) in enumerate(val_loader, 0):
                images, labels = images.to(computing_device), labels.to(computing_device)

                optimizer.zero_grad()

                outputs = model(images)
                val_loss += criterion(outputs, labels)
                
                # Performs sigmoid on the output for when the criterion is BCEWithLogitsLogs
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    outputs = torch.sigmoid(outputs)
                
                predicted = torch.round(outputs.data)
                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy()
                val_acc += accuracy(predicted, labels)
                
                # Save the validation loss every quarter of the validation data
                if (batch_count + 1) % N_valid_loss == 0:
                    valid_loss.append(val_loss / N_valid_loss)
                    val_loss = 0.0
                    valid_accuracy.append(val_acc / N_valid_loss)
                    val_acc = 0.0
                    
                #if len(valid_accuracy) >= 2:
                    #break
                

            # Early stopping
            '''  
            if val_loss >= min_val_loss:
                early_stop_count += 1
                if early_stop_count == 2:
                    break

            else:
                early_stop_count = 0
                min_val_loss = val_loss
            '''

        print('Saving model')
        torch.save(model, model_name + str(epoch) + '_kf.pt')
        print("Finished", epoch + 1, "epochs of training")
        
    print("Training complete after", epoch + 1, "epochs")
    
    return train_loss, valid_loss, train_accuracy, valid_accuracy, time.time() - start

def test_model(model, test_loader, computing_device, optimizer, criterion):
    agg_accuracies = []
    agg_precisions = []
    agg_recalls = []
    agg_BCRs = []

    class_accuracies = []
    class_precisions = []
    class_recalls = []
    class_BCRs = []
    
    total_acc_per_class = [0] * 14
    total_pre_per_class = [0] * 14
    total_rec_per_class = [0] * 14
    total_BCR_per_class = [0] * 14
    
    conf_mtx = np.zeros((14, 14), dtype = np.float32)
    
    batch_start = 0
    N = 50

    with torch.no_grad():
        # Get the next minibatch of images, labels 
        for minibatch_count, (images, labels) in enumerate(test_loader, 0):
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Zero out the stored gradient (buffer) from the previous iteration
            #optimizer.zero_grad()

            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            
            # Performs sigmoid on the output for when the criterion is BCEWithLogitsLogs
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                outputs = torch.sigmoid(outputs)
            
            predicted = torch.round(outputs.data)

            # Convert from Cuda tensor -> numpy array
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            conf_mtx = confusion_matrix(conf_mtx, predicted, labels)

            # Compute aggregated scores
            acc = accuracy(predicted, labels)
            pre = precision(predicted, labels)
            rec = recall(predicted, labels)
            bcr = BCR(predicted, labels)

            agg_accuracies.append(acc)
            agg_precisions.append(pre)
            agg_recalls.append(rec)
            agg_BCRs.append(bcr)

            # Compute scores by class
            acc = accuracy_per_class(predicted, labels)
            pre = precision_per_class(predicted, labels)
            rec = recall_per_class(predicted, labels)
            bcr = BCR_per_class(predicted, labels)

            class_accuracies.append(acc)
            class_precisions.append(pre)
            class_recalls.append(rec)
            class_BCRs.append(bcr)
            
            total_acc_per_class += acc
            total_pre_per_class += pre
            total_rec_per_class += rec
            total_BCR_per_class += bcr

            if (minibatch_count + 1) % N == 0: 
                # Print the loss averaged over the last N mini-batches
                print('----- Aggregated Scores -----')
                print_scores(batch_start, minibatch_count, agg_accuracies, 
                             agg_precisions, agg_recalls, agg_BCRs, aggregate=True)
                '''
                print('----- Scores By Class -----')
                print_scores(batch_start, minibatch_count, class_accuracies, 
                             class_precisions, class_recalls, class_BCRs, aggregate=False)
                '''
                batch_start = minibatch_count + 1
                
                #break
                
                
    return total_acc_per_class / (minibatch_count + 1), total_pre_per_class / (minibatch_count + 1), total_rec_per_class / (minibatch_count + 1), total_BCR_per_class / (minibatch_count + 1), conf_mtx 

# Define the hyperparameters
num_epochs = 2           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.0001  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.4              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

runs = 2                 # Used in k-fold validation

#------------- BASELINE

# Define the model parameters
transform = transforms.Compose(
    [transforms.Resize(512), 
     transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# trains the model
model = BasicCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

# Define the loss criterion and instantiate the gradient descent optimizer
criterion = nn.BCELoss()

# Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []

train_accs = []
valid_accs = []

class_accs = []
class_pres = []
class_recs = []
class_bcrs = []

conf_mtxs = []

for r in range(runs):
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)
    train_loss, valid_loss, train_acc, valid_acc, time_elapsed = train_model(r, model, 'Baseline', num_epochs, train_loader, computing_device, optimizer, criterion)
    class_acc, class_pre, class_rec, class_bcr, conf_mtx = test_model(model, test_loader, computing_device, optimizer, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    class_accs.append(class_acc)
    class_pres.append(class_pre)
    class_recs.append(class_rec)
    class_bcrs.append(class_bcr)
    conf_mtxs.append(conf_mtx)
    
train_losses = np.mean(train_losses, axis=0)
valid_losses = np.mean(valid_losses, axis=0)
train_accs = np.mean(train_accs, axis=0)
valid_accs = np.mean(valid_accs, axis=0)
class_accs = np.mean(class_accs, axis=0)
class_pres = np.mean(class_pres, axis=0)
class_recs = np.mean(class_recs, axis=0)
class_bcrs = np.mean(class_bcrs, axis=0)
conf_mtxs = np.mean(conf_mtxs, axis=0) 

# Normalize the confusion matrix
for j in range(conf_mtxs.shape[0]):
    if np.sum(conf_mtxs[j]) != 0:
        conf_mtxs[j] /= np.sum(conf_mtxs[j])
        
print('--------------------------------------------------------------------------')
print("Training loss")
print(train_losses)

print('--------------------------------------------------------------------------')
print("Validation loss")
print(valid_losses)

print('--------------------------------------------------------------------------')
print("Training accuracy")
print(train_accs)

print('--------------------------------------------------------------------------')
print("Validation accuracy")
print(valid_accs)

print('--------------------------------------------------------------------------')
print("Per class accuracy")
print(class_accs)

print('--------------------------------------------------------------------------')
print("Per class precision")
print(class_pres)

print('--------------------------------------------------------------------------')
print("Per class recall")
print(class_recs)

print('--------------------------------------------------------------------------')
print("Per class BCR")
print(class_bcrs)

print('--------------------------------------------------------------------------')
print("Confusion matrix")        
print(conf_mtxs)

plt.plot(range(len(train_loss)), train_loss, 'b--', label = 'Training Loss')
plt.plot(range(len(valid_loss)), valid_loss, 'r--', label = 'Validation Loss')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.title("Baseline Model Loss Curve")
plt.legend(loc="upper right")

plt.savefig("images/m0_loss.png")

plt.plot(range(len(train_acc)), train_acc, 'b--', label = 'Training Accuracy')
plt.plot(range(len(valid_acc)), valid_acc, 'r--', label = 'Validation Accuracy')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Accuracy")
plt.title("Baseline Model Accuracy Curve")
plt.legend(loc='lower right')

plt.savefig("images/m0_acc.png")

weights = model.conv1.weight.data.cpu().numpy()
plt.imsave('images/m0_early1.png', weights[0, 0])
plt.imsave('images/m0_early2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv2.weight.data.cpu().numpy()
plt.imsave('images/m0_mid1.png', weights[0, 0])
plt.imsave('images/m0_mid2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv3.weight.data.cpu().numpy()
plt.imsave('images/m0_last1.png', weights[0, 0])
plt.imsave('images/m0_last2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])


#------------- CUSTOM MODEL 1

# Define the model architecture and parameters
channel = [1, 8, 8, 8, 16, 16, 16, 32, 32, 32]
block_size = 3
kernel = [5]*9
pool_kernel = 2
pool_stride = 2
transform = transforms.Compose(
    [transforms.Resize(256), 
     transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = Model1CNN(256, channel, kernel, block_size, pool_kernel, pool_stride)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

# Define the loss criterion and instantiate the gradient descent optimizer
criterion = nn.BCELoss()

# Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []

train_accs = []
valid_accs = []

class_accs = []
class_pres = []
class_recs = []
class_bcrs = []

conf_mtxs = []

for r in range(runs):
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)
    train_loss, valid_loss, train_acc, valid_acc, time_elapsed = train_model(r, model, 'Custom1', num_epochs, train_loader, computing_device, optimizer, criterion)
    class_acc, class_pre, class_rec, class_bcr, conf_mtx = test_model(model, test_loader, computing_device, optimizer, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    class_accs.append(class_acc)
    class_pres.append(class_pre)
    class_recs.append(class_rec)
    class_bcrs.append(class_bcr)
    conf_mtxs.append(conf_mtx)
    
train_losses = np.mean(train_losses, axis=0)
valid_losses = np.mean(valid_losses, axis=0)
train_accs = np.mean(train_accs, axis=0)
valid_accs = np.mean(valid_accs, axis=0)
class_accs = np.mean(class_accs, axis=0)
class_pres = np.mean(class_pres, axis=0)
class_recs = np.mean(class_recs, axis=0)
class_bcrs = np.mean(class_bcrs, axis=0)
conf_mtxs = np.mean(conf_mtxs, axis=0) 

# Normalize the confusion matrix
for j in range(conf_mtxs.shape[0]):
    if np.sum(conf_mtxs[j]) != 0:
        conf_mtxs[j] /= np.sum(conf_mtxs[j])
        
print('--------------------------------------------------------------------------')
print("Training loss")
print(train_losses)

print('--------------------------------------------------------------------------')
print("Validation loss")
print(valid_losses)

print('--------------------------------------------------------------------------')
print("Training accuracy")
print(train_accs)

print('--------------------------------------------------------------------------')
print("Validation accuracy")
print(valid_accs)

print('--------------------------------------------------------------------------')
print("Per class accuracy")
print(class_accs)

print('--------------------------------------------------------------------------')
print("Per class precision")
print(class_pres)

print('--------------------------------------------------------------------------')
print("Per class recall")
print(class_recs)

print('--------------------------------------------------------------------------')
print("Per class BCR")
print(class_bcrs)

print('--------------------------------------------------------------------------')
print("Confusion matrix")        
print(conf_mtxs)

plt.plot(range(len(train_loss)), train_loss, 'b--', label = 'Training Loss')
plt.plot(range(len(valid_loss)), valid_loss, 'r--', label = 'Validation Loss')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.title("Customized Model Loss Curve")
plt.legend(loc="upper right")

plt.savefig("images/m1_loss.png")

plt.plot(range(len(train_acc)), train_acc, 'b--', label = 'Training Accuracy')
plt.plot(range(len(valid_acc)), valid_acc, 'r--', label = 'Validation Accuracy')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Accuracy")
plt.title("Customized Model Accuracy Curve")
plt.legend(loc='lower right')

plt.savefig("images/m1_acc.png")

weights = model.conv[2].weight.data.cpu().numpy()
plt.imsave('images/m1_early1.png', weights[0, 0])
plt.imsave('images/m1_early2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv[5].weight.data.cpu().numpy()
plt.imsave('images/m1_mid1.png', weights[0, 0])
plt.imsave('images/m1_mid2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv[8].weight.data.cpu().numpy()
plt.imsave('images/m1_last1.png', weights[0, 0])
plt.imsave('images/m1_last2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

#------------- CUSTOM MODEL 2

# Define the model parameters
transform = transforms.Compose(
    [transforms.Resize(256), 
     transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# trains the model
model = Model2CNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

# Define the loss criterion and instantiate the gradient descent optimizer
criterion = nn.BCELoss()

# Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []

train_accs = []
valid_accs = []

class_accs = []
class_pres = []
class_recs = []
class_bcrs = []

conf_mtxs = []

for r in range(runs):
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)
    train_loss, valid_loss, train_acc, valid_acc, time_elapsed = train_model(r, model, 'Custom2', num_epochs, train_loader, computing_device, optimizer, criterion)
    class_acc, class_pre, class_rec, class_bcr, conf_mtx = test_model(model, test_loader, computing_device, optimizer, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    class_accs.append(class_acc)
    class_pres.append(class_pre)
    class_recs.append(class_rec)
    class_bcrs.append(class_bcr)
    conf_mtxs.append(conf_mtx)
    
train_losses = np.mean(train_losses, axis=0)
valid_losses = np.mean(valid_losses, axis=0)
train_accs = np.mean(train_accs, axis=0)
valid_accs = np.mean(valid_accs, axis=0)
class_accs = np.mean(class_accs, axis=0)
class_pres = np.mean(class_pres, axis=0)
class_recs = np.mean(class_recs, axis=0)
class_bcrs = np.mean(class_bcrs, axis=0)
conf_mtxs = np.mean(conf_mtxs, axis=0) 

# Normalize the confusion matrix
for j in range(conf_mtxs.shape[0]):
    if np.sum(conf_mtxs[j]) != 0:
        conf_mtxs[j] /= np.sum(conf_mtxs[j])
        
print('--------------------------------------------------------------------------')
print("Training loss")
print(train_losses)

print('--------------------------------------------------------------------------')
print("Validation loss")
print(valid_losses)

print('--------------------------------------------------------------------------')
print("Training accuracy")
print(train_accs)

print('--------------------------------------------------------------------------')
print("Validation accuracy")
print(valid_accs)

print('--------------------------------------------------------------------------')
print("Per class accuracy")
print(class_accs)

print('--------------------------------------------------------------------------')
print("Per class precision")
print(class_pres)

print('--------------------------------------------------------------------------')
print("Per class recall")
print(class_recs)

print('--------------------------------------------------------------------------')
print("Per class BCR")
print(class_bcrs)

print('--------------------------------------------------------------------------')
print("Confusion matrix")        
print(conf_mtxs)

plt.plot(range(len(train_loss)), train_loss, 'b--', label = 'Training Loss')
plt.plot(range(len(valid_loss)), valid_loss, 'r--', label = 'Validation Loss')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.title("ResNet12 Loss Curve")
plt.legend(loc='upper right')

plt.savefig("images/m2_loss.png")

plt.plot(range(len(train_acc)), train_acc, 'b--', label = 'Training Accuracy')
plt.plot(range(len(valid_acc)), valid_acc, 'r--', label = 'Validation Accuracy')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Accuracy")
plt.title("ResNet12 Accuracy Curve")
plt.legend(loc='lower right')

plt.savefig("images/m2_acc.png")

weights = model.conv3.weight.data.cpu().numpy()
plt.imsave('images/m2_early1.png', weights[0, 0])
plt.imsave('images/m2_early2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv6.weight.data.cpu().numpy()
plt.imsave('images/m2_mid1.png', weights[0, 0])
plt.imsave('images/m2_mid2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv12.weight.data.cpu().numpy()
plt.imsave('images/m2_last1.png', weights[0, 0])
plt.imsave('images/m2_last2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])


#------------- RERUN CUSTOM MODEL 2 WITH CLASS BALANCING

# Define the model parameters
transform = transforms.Compose(
    [transforms.Resize(256), 
     transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Train the model
model = Model2CNN(skip_sigmoid=True)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

# Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []

train_accs = []
valid_accs = []

class_accs = []
class_pres = []
class_recs = []
class_bcrs = []

conf_mtxs = []

for r in range(runs):
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)
    
    # Creates the weights to address the class imbalance
    total_data_points = 0
    if use_cuda:
        total_class_counts = torch.cuda.FloatTensor(14).fill_(0)
    else:
        total_class_counts = torch.FloatTensor(14).fill_(0)    
        
    N = 50
    NN = 100
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):
        total_data_points += labels.shape[0]
        total_class_counts += torch.sum(labels, dim = 0).cuda()
        if (minibatch_count + 1) % N == 0:
            print("Finished counting minibatch " + str(minibatch_count + 1))
            print("Total data points = " + str(total_data_points))
            print("Total class counts = " + str(total_class_counts))
        if (minibatch_count + 1) % NN == 0:
            break

    inverse_class_frequency = (total_data_points - total_class_counts) / (total_class_counts + 1)

    print('--------------------------------------------------------------------------')
    print("Inverse Class Frequency")
    print(inverse_class_frequency)
    
    # Define the loss criterion and instantiate the gradient descent optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=inverse_class_frequency)
    
    train_loss, valid_loss, train_acc, valid_acc, time_elapsed = train_model(r, model, 'Custom2ClassBalance', num_epochs, train_loader, computing_device, optimizer, criterion)
    class_acc, class_pre, class_rec, class_bcr, conf_mtx = test_model(model, test_loader, computing_device, optimizer, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    class_accs.append(class_acc)
    class_pres.append(class_pre)
    class_recs.append(class_rec)
    class_bcrs.append(class_bcr)
    conf_mtxs.append(conf_mtx)
    
train_losses = np.mean(train_losses, axis=0)
valid_losses = np.mean(valid_losses, axis=0)
train_accs = np.mean(train_accs, axis=0)
valid_accs = np.mean(valid_accs, axis=0)
class_accs = np.mean(class_accs, axis=0)
class_pres = np.mean(class_pres, axis=0)
class_recs = np.mean(class_recs, axis=0)
class_bcrs = np.mean(class_bcrs, axis=0)
conf_mtxs = np.mean(conf_mtxs, axis=0) 

# Normalize the confusion matrix
for j in range(conf_mtxs.shape[0]):
    if np.sum(conf_mtxs[j]) != 0:
        conf_mtxs[j] /= np.sum(conf_mtxs[j])
        
print('--------------------------------------------------------------------------')
print("Training loss")
print(train_losses)

print('--------------------------------------------------------------------------')
print("Validation loss")
print(valid_losses)

print('--------------------------------------------------------------------------')
print("Training accuracy")
print(train_accs)

print('--------------------------------------------------------------------------')
print("Validation accuracy")
print(valid_accs)

print('--------------------------------------------------------------------------')
print("Per class accuracy")
print(class_accs)

print('--------------------------------------------------------------------------')
print("Per class precision")
print(class_pres)

print('--------------------------------------------------------------------------')
print("Per class recall")
print(class_recs)

print('--------------------------------------------------------------------------')
print("Per class BCR")
print(class_bcrs)

print('--------------------------------------------------------------------------')
print("Confusion matrix")        
print(conf_mtxs)

plt.plot(range(len(train_loss)), train_loss, 'b--', label = 'Training Loss')
plt.plot(range(len(valid_loss)), valid_loss, 'r--', label = 'Validation Loss')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.title("ResNet12 With Class Balancing Loss Curve")
plt.legend(loc='upper right')

plt.savefig("images/m3_loss.png")

plt.plot(range(len(train_acc)), train_acc, 'b--', label = 'Training Accuracy')
plt.plot(range(len(valid_acc)), valid_acc, 'r--', label = 'Validation Accuracy')

plt.grid(True)

plt.xlabel("Minibatch")
plt.ylabel("Accuracy")
plt.title("ResNet12 With Class Balancing Accuracy Curve")
plt.legend(loc='lower right')

plt.savefig("images/m3_acc.png")

weights = model.conv3.weight.data.cpu().numpy()
plt.imsave('images/m3_early1.png', weights[0, 0])
plt.imsave('images/m3_early2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv6.weight.data.cpu().numpy()
plt.imsave('images/m3_mid1.png', weights[0, 0])
plt.imsave('images/m3_mid2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

weights = model.conv12.weight.data.cpu().numpy()
plt.imsave('images/m3_last1.png', weights[0, 0])
plt.imsave('images/m3_last2.png', weights[1, 0])
plt.figure().add_subplot(111).imshow(weights[0, 0])
plt.figure().add_subplot(111).imshow(weights[1, 0])

# Loads the model
# model2 = torch.load('file_name')

