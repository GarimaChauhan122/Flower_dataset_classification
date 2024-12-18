# Imports here
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
# helps in manipulation 
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from collections import OrderedDict
# used for defining modified state dict later on
import time
import torch
from torch.utils.data import Dataset, DataLoader
#import tensorflow as tf
#from tensorflow import keras
import numpy as np
from PIL import Image
import os
import json
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
# this is our model :)) 
from collections import OrderedDict
#print("TensorFlow version:", tf.__version__)

#########################################################################################
#first up, we are creating the argument parser
parser = argparse.ArgumentParser(description="Train.py")
#The description parameter provides a brief text about the script's purpose, which argparse will display when the user
#runs the script with a help flag, such as --help or -h.
parser.add_argument('--arch', dest="arch", action="store", default="efficientnet-b0", type = str)
parser.add_argument('--save_dir', type = str, default = './',help = 'Provide the save directory')
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
#for the learning rate of the model
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
# for the num of neurons
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=18)
# for the number of epochs 
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
args = parser.parse_args()

#########################################################################################

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#########################################################################################

# Define function to load and preprocess images using PIL and numpy
def load_and_preprocess_image(image_path, image_size=(224, 224)):
    # Open the image file
    with Image.open(image_path) as img:
        img = img.convert('RGB')  # Ensure 3 channels (RGB)
        img = img.resize(image_size)  # Resize to specified size

        # Convert to numpy array and scale pixel values to [0, 1]
        img_np = np.array(img) / 255.0

        # Transpose to match PyTorch's expected shape (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1)).astype(np.float32)
        
    return img_np  # Returns a numpy array

# Custom Dataset class to create a PyTorch-compatible dataset from image files
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Populate image paths and labels from directory structure
        for idx, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for image_file in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_file)
                    self.image_paths.append(image_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess image
        image = load_and_preprocess_image(image_path)

        # Apply transformations if any (note: ToTensor should not be included)
        if self.transform:
            image = self.transform(torch.from_numpy(image))

        return image, label

#########################################################################################

'''
Why normalize??
    Models pre-trained on ImageNet, such as ResNet, VGG, EfficientNet, and others, were trained with this normalization in place.
    To leverage the learned features from these models accurately, the input images need to undergo the same normalization, 
    helping the model interpret new images in a way that aligns with the images it was trained on.
    
so here, we also perform the randomhorizontalflip() on the train dataset. This flips the images with a certain probability because we wish the model to be trained such that it can predict mirror images of the flowers too.
since the transformations required for the training and testing differs in this sense, we are defining different transformations and storing it in a dictionary.
'''
data_transforms = {
    'train': transforms.Compose([transforms.RandomHorizontalFlip(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

# Create PyTorch data loaders
train_loader = DataLoader(ImageFolderDataset(train_dir, transform=data_transforms['train']), batch_size=32, shuffle=True)
# now what we did here was that, we want the data to first undergo transformations accoridng to whether it is train or test
# and then we want that in each epoch, it undergoes training for many mini-batches coz the number of images otherwise are too huge
# therefore, the batch_size= 32 implies 32 images in every batch and we are shuffling it here, because we dont want the model
# to train on the same batch each time
valid_loader = DataLoader(ImageFolderDataset(valid_dir, transform=data_transforms['valid']), batch_size=32, shuffle=False)
# here, shuffling doesnt matter that much, so shuffling = falsee
test_loader = DataLoader(ImageFolderDataset(test_dir, transform=data_transforms['test']), batch_size=32, shuffle=False)

#########################################################################################

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Load the pretrained EfficientNet model (B0 variant)
model = EfficientNet.from_pretrained('efficientnet-b0')

'''
    when we talk about teh structure of efficient net, it can be divided into 4 major components (for our understanding right now):
    1. Feature Extraction Layers: The entire stack of convolutional layers, batch normalization layers, pooling layers,
    and other intermediate components (such as the EfficientNet’s mobile inverted bottleneck blocks). 
    These layers are responsible for extracting features from images, such as edges, textures, shapes, and higher-level features.
    
    2. Initial Layers: The initial layers, including the input layer and early convolutional layers, remain untouched. 
    These layers process the raw pixel data and begin extracting low-level features, which serve as the foundational 
    details needed by deeper layers.
    
    3. Global Pooling Layer: EfficientNet includes a global average pooling layer that reduces the spatial dimensions
    of the feature maps to a fixed-size vector (of size 1280 for EfficientNet B0).
    
    4. And now you have the classifier. The initial classifier of efficientnet-b0 has been trained on 1000 classes for the
    dataset of imagenet..what we are doing right now is replacing the original classifier with what we have built using the dataset
    for classifying flowers across 102 classes. this is the part that we will be rebuilding!!(drumroll please** Transfer learning ) 
    
'''


# Freeze all parameters in the pretrained EfficientNet model. This is typically done when certain parts of the model
# are already well-trained and reliable, so we freeze them to avoid recomputation and allow the model to focus on
# training new or modified layers. In our case, we are freezing all parameters except for the 4th component—the classifier,
# which we’ll be replacing with a custom classifier to suit our specific task.
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(1280, 512)),  # First hidden layer (1280 -> 512)
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),  # Dropout with 0.5 probability
            ('hidden_layer1', nn.Linear(512, 256)),  # Second hidden layer (512 -> 256)
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(256, 102)),  # Output layer (256 -> 102)
            ('output', nn.LogSoftmax(dim=1))  # LogSoftmax for NLLLoss
]))

'''
    alrightyy so
    1. the first layer takes an input vector of length 1280 (which is the output of EfficientNet B0’s feature extractor) and 
       has checkpoint['hidden_layer1'] number of neurons..therefore that is its output
    2. here, the activation function is recitified linear unit (relu) that will introduce non linearity and help us capture complex patterns
    3. adds a dropout layer defined by the rate checkpoint['dropout']. this randomly selects certain neurons and makes their activation 0,
       so we dont rely on them that much- helps prevent overfitting
    4. since the last time, our output was of size checkpoint['hidden_layer1'], this will be considered as the input vector and gives an
       output of size checkpoint['hidden_layer2'] by having those many number of neurons.
    5. again, the activation function here is relu
    6. this time, the input vector is of size checkpoint['hidden_layer2'] and since this is the last layer, we define num_of_output_neurons=
       num_of_classes will equal 102. each neuron can be thought of as representing one class
    7. since the last layer's output need to basically represent the probability of being in that class, we apply logsoftmax as
       the activation function. It takes a vector of raw scores (logits) from the model’s output layer and converts them into
       a probability distribution. 
       
'''


# now that we have built our custom classifier, we can replace the model's with ours
model._fc = classifier

# Move the model to GPU if available (because we will be beginning the computation from here)
if torch.cuda.is_available():
    model = model.cuda()

'''
Now, lets say that the model has predicted some output..we would want the difference between the predicted and the ground truth 
or the loss basically to be less..how to calculate the loss? tht is what the criterion tells
here, NLL_Loss is basically negative log likelihood loss..it takes the probabilities and applies negative log likelihood upon it
'''   
criterion = nn.NLLLoss()

'''
After determining the loss, we would need to update the weights and biases on the basis of the loss function so as to reduce
it..the optimizer comes into the picture here
An optimizer in machine learning (particularly in the context of neural networks) is an algorithm used to update the model's 
parameters (such as weights and biases) during the training process. The goal of an optimizer is to minimize the loss 
function (or cost function), which measures how well the model's predictions align with the actual targets.
so lets break down the next piece of code
the optimizer we are using is Adam's that combines the advantages of both Adagrad and RMSprop. It adapts the learning rate 
for each parameter based on the first and second moments of the gradients. Adam is known for being efficient in terms of
memory and for performing well in many types of deep learning models.
now, model._fc basically corresponds to a certain part of the model. (remember, we were talking about the convolutional layers and how we dont wish to change their parameters, so here, its crucial that we explicitely mention which part we wish to change)
fc corresponds to fully connected layer, which is basically the classifier that we just built. parameters means the weights and biases that
the optimizer will update during the training phase.
now, how much do we wish to change the loss, or in other words,at what rate do you wish the model to learn the newer parameters?
this is given by the learning rate(lr) and its usually good to begin with lr= 0.001
'''
optimizer = optim.Adam(model._fc.parameters(), lr=0.001)

'''
since we are already having a separate validation and testing dataset, this is what we will do..the training process is as belows:
there are 18 epochs, in each epoch we will basically take each batch of data from the train_loader, pass it thru the network, then 
get the output, calculate the loss and then update the parameters. we do it for all the mini-batches.
this is a single epoch. at the end of it, we are calculating the accuracy and loss obtained on the train dataset
similarly, we just test it on the validation dataset and check the accuracy and loss
'''
#########################################################################################

# Training on test data and validating on validation data
epochs = args.epochs
l_r= args.learning_rate
h_u= args.hidden_units

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}/{epochs}")
    #this is just printing the current epoch that we are on... adding 1 to make it human-readable

    # Set model to training mode
    '''
    here, we are setting the model to tarin mode so that it satisfies the given two functionalities
    1. if you remember,we had aksed to dropped certain values(ie, set certain neuron output to 0 during forward pass) while building
       the model, because we didnt want it to overfit. therefore, put it in train mode to ensure that
    2. batch normalization: during training, it computes the error& accuracy etc statistics for each batch, but while testing
       we want it to be cumulative so again,we explicitely mention that it is training
    '''
    model.train()

    # we would wanna track the test_loss and the test_accuracy on the data
    test_loss = 0.0
    test_acc = 0.0

    for batch_idx, (inputs, labels) in enumerate(test_loader, 1): 
        
        '''
        here batch_index refers to the index of each batch. each batch of the test_loader consists of the inputs (ie the images)
        and the labels (which tells which one of the 102 classes it belongs to) soo yeahh
        The enumerate() function is used to loop over a sequence and keep track of the index of the current element.
        By default, enumerate() starts the index at 0. However, the 1 passed as the second argument to enumerate() tells 
        it to start the index from 1 instead of 0.
        So, batch_idx will represent the batch number, starting from 1.
        '''
        
        # Move data to GPU if available
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # In PyTorch, during the backward pass, gradients are accumulated by default. This means that the gradients from the
        #current batch are added to the gradients from the previous batch. While this accumulation can be useful in some cases
        #(e.g., when implementing custom gradient updates or working with larger batch sizes), for typical training loops, we 
        #want to reset the gradients before each new backward pass to avoid using gradients from previous batches.
        #zero_grad() does exactly that 
        optimizer.zero_grad()

        # Forward pass
        # here, the shape of the outputs will be (N,C) where N is the number of images in that batch and C is 102, ie, the classes
        outputs = model(inputs)

        # we move onto calculating the loss which was defined by our criterion as NLLLoss
        loss = criterion(outputs, labels)
        
        '''
        Now that the loss is computed , we need to know how much adjustment is needed for each parameter so that the loss is minimised
        Backpropagation uses the chain rule of calculus to compute these gradients. It works by propagating the error backward
        through the network, layer by layer, computing how the error changes with respect to each parameter.
        This is done for every layer in the model, from the output layer back to the input layer.
        '''
        loss.backward()

        
        '''
        After computing the gradients using loss.backward(), the optimizer uses those gradients to update the parameters of the model
        (weights and biases). The specific update rule depends on the optimization algorithm being used (e.g., Stochastic Gradient
        Descent (SGD), Adam, RMSprop). The optimizer adjusts the parameters based on the gradient and learning rate.
        '''
        optimizer.step()

        # Track accuracy and loss
        # the loss tensor contains the scalar loss value and the .item() extracts it as a python float value and
        #by multiplying it with the inputs.size(0), we are basically totalling it over the entire batch dataset
        # basically, we are finding it for an entire epoch, so we are summing over each batch
        test_loss += loss.item() * inputs.size(0)
        # here,the torch.max() basically returns a tuple where the first value is the maximum value of the logits and the second is the 
        # predicted class label..since we do not need the max logit value, you can discard it using _ 
        _, predictions = torch.max(outputs.data, 1)
        # here, this is now checking if the predictions and labels are equal and then returning a tensor of equal size as either..
        # with true if the label matches and false if it doesnt
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # since we need the total number of correct predicted classes, we can just find the sum
        # basically, we are finding it for an entire epoch, so we are summing over each batch
        test_acc += correct_counts.sum().item()

        # Print batch metrics to track progress within the epoch
        # we will get the batch_loss thru the loss.item()
        batch_loss = loss.item()
        # we will get the accuracy by summing the total correct_counts and dividing by the batch size..this will be a value between (0,1)
        batch_acc = correct_counts.sum().item() / inputs.size(0)
        print(f'  Batch {batch_idx}/{len(test_loader)}: Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.4f}')

    # Calculate and print epoch-level metrics for training on test data
    epoch_test_loss = test_loss / len(test_loader.dataset)
    epoch_test_acc = test_acc / len(test_loader.dataset)
    print(f'Epoch {epoch + 1} Summary - Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.4f}')

    # now that we are done with training thru all the batches in the test loader for a given epoch, we can move onto testing it
    # on the validation data..for that, first,Set the model to evaluation mode
    model.eval() 
    valid_loss = 0.0
    valid_acc = 0.0

    with torch.no_grad():  # because you are just evaluating, no need to compute the gradient for each loss..so just disable it
        for inputs, labels in valid_loader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                
            # doing the same procedure 
            # Forward pass
            outputs = model(inputs)
           
            # Calculate loss
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)

            # Track accuracy
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            valid_acc += correct_counts.sum().item()

    # Calculate and print epoch-level metrics for validation
    epoch_valid_loss = valid_loss / len(valid_loader.dataset)
    epoch_valid_acc = valid_acc / len(valid_loader.dataset)
    print(f'Epoch {epoch + 1} Summary - Validation Loss: {epoch_valid_loss:.4f}, Validation Accuracy: {epoch_valid_acc:.4f}\n')

#########################################################################################

# now that we are done training the model overall (achieved above 85%+ accuracy on the training dataset overall on average)
# we will test it on our overall test dataset
correct, total = 0, 0
with torch.no_grad():
    # again, setting it to eval mode lol
    model.eval()
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy on test images: {100 * correct / total:.2f}%')

#########################################################################################

# here, just as a part of sanity checking, i will be checking how it is working on the very first image in the test loader
with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        # Take only the first image and label in the batch
        image = images[0].unsqueeze(0)  # Add batch dimension for a single image
        label = labels[0].unsqueeze(0)
        
        # No need to move data to CUDA
        output = model(image)
        print(output)
        _, predicted = torch.max(output.data, 1)
        
        # Calculate accuracy for the single image
        is_correct = (predicted == label).item()
        
        # and i will just print the statistics involved..ie the predicted and the true class and the accuracy which is just 0 or 1
        print(f"Predicted class: {predicted.item()}, True class: {label.item()}")
        print(f"Accuracy on single test image: {'100%' if is_correct else '0%'}")
        
        #and now we break coz we just wanted to test on a single image :)))
        break  
#########################################################################################

model.class_to_idx = train_loader.dataset.class_to_idx
# we are adding another attribute to the model which is the class_to_index mapping for better unpacking
#print(train_loader.dataset.class_to_idx)

#now, this is basically some shortcut that i am applying because when i saved the checkpoints, they were saved in a certain format
# and when i had to load it back, there was a mismatch and they couldnt figure out that the missing and the extra pieces are the same
state_dict = model.state_dict()
modified_state_dict = OrderedDict()
# so here, i am manually renaming so that this mismatch is resolved.
for key, value in state_dict.items():
    new_key = key.replace('inputs', '0').replace('relu1', '1').replace('dropout', '2') \
                 .replace('hidden_layer1', '3').replace('relu2', '4').replace('hidden_layer2', '5')
    modified_state_dict[new_key] = value
# and now we move onto saving the checpoints so that the model can be rebuilt
# here, hidden_layer1 is the number of neurons in the first hidden layer, irrespective of the input vector, the output vector
# will be of this size
# dropout is again the number of activations we wish to ignore so that we wouldn't have to face overfitting
# if u see state_dict is very smartly the modified_state_dict :)))
# and we are done
torch.save({
    'structure': 'efficientnet-b0',
    'hidden_layer1': 512,           
    'hidden_layer2': 256,
    'dropout': 0.5,
    'epochs': 18,                    
    'state_dict': modified_state_dict,
    'class_to_idx': model.class_to_idx,
    'optimizer_dict': optimizer.state_dict()
}, 'model_checkpoint_via_pyfile.pth')

print("Checkpoint saved yayy.")

#########################################################################################
# usage: python train.py
