To execute our code, simply run:

### python train_model.py

This will execute the training, validation, and testing for all three of our models, 
i.e. Baseline (Model 0), Customized CNN (Model 1), and ResNet12 (Model 2). 

While a model is training, you will see it output the run #, epoch #, minibatch #, the 
loss for that minibatch, and the amount of time elapsed. Afterwards, when a model is 
testing, it will print the aggregated performance scores for each minibatch. Finally, 
after the model finishes testing, the program will print performance metrics (losses, 
accuracies, precisions, recalls, BCRs, and a confusion matrix) that were calculated 
for a per-class basis.

For each model, we save a ".pt" file of that model, so if the need to rerun the code 
arises, you can just load in the file using:

### model = torch.load('pt_file_name')

No need to rerun the training block. We have also provided the ".ipynb" versions of 
each file, in case you want to only run blocks of the code.

Also for each model, we save the filter maps and graphs to the "\images" folder. In 
this folder, the filter maps are saved as:

"m[0/1/2]_[early/mid/last][1/2].png"

For example, the 2nd filter image from the middle layer of model 0 will be saved as 
"m0_mid2.png". Any graphs, for either accuracy or loss, will be saved as:

"m[0/1/2/3]_[acc/loss].png"

E.G. "m3_loss.png", where "3" represents model 2 with class imbalance implemented. 

******  IMPORTANT NOTICE

Training, validation, and testing will each run for 2 iterations, where each 
iteration will execute over 2 epochs. This means that the run-time in VERY 
long (almost a full day)
******

 
