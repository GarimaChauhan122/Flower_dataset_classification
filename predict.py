import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
#from train import check_gpu
from torchvision import models
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
#nn stands for neural network. here, it is being used to rebuild the classifier from the checkpoints
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
#transforms is used to apply transformations to input images
import torchvision.models as models
import argparse
from collections import OrderedDict
import json
import PIL
import seaborn as sns
from PIL import Image
#usage: used in the process_image function to load images
import time

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    #The description parameter provides a brief text about the script's purpose, which argparse will display when the user
    #runs the script with a help flag, such as --help or -h.
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    #here, you are adding an argument with the variable name (--image) and providing its description thry help and that 
    #the type you expect is string. it is required as the program needs an input image to predict its class
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=False, default='model_checkpoint_via_pyfile.pth')
    #another argument u need is the model checkpoint file to load all the weights and biases into our model
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.',default=7)
    # this is an optional argument..if the user doesnt provide an input,it will display top 7 classes otherwise. 
    #why 7? harry potter 7. ms dhoni 7 :))
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    # again if by some freak chance all flowers are renamed tomorrow, u can provide another file to help us map the 
    #class labels to the actual flower name
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()    
    return args

#####################################################################


def loading_the_checkpoint(path='checkpoint.pth'):
    #okayy so this function is used to load all the weights and biases into my model.so this function first constructs the classifier
    #of my model and then loads everything else from the saved checkpoints file
    
    checkpoint = torch.load(path)
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
        
    
    classifier = nn.Sequential(
        nn.Linear(1280, checkpoint['hidden_layer1']),  # Input size: 1280 (EfficientNet B0 output), first hidden layer size
        nn.ReLU(),
        nn.Dropout(checkpoint['dropout']),  # Dropout layer
        nn.Linear(checkpoint['hidden_layer1'], checkpoint['hidden_layer2']),  # Second hidden layer
        nn.ReLU(),
        nn.Linear(checkpoint['hidden_layer2'], 102),  # Output layer (102 classes)
        nn.LogSoftmax(dim=1)  # LogSoftmax for NLLLoss
    )
    
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

    # Load the state_dict into the model (weights)
    # A state dictionary in PyTorch is a Python dictionary that maps each layer in the model to its corresponding parameters
    # (e.g., weights and biases). When you save a model’s state, it’s stored as a state dictionary, which is what 
    # checkpoint['state_dict'] contains here.
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Load class_to_idx for later use (for mapping classes to indices)
    model.class_to_idx = checkpoint['class_to_idx']

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    return model


#########################################################################

def process_image(image):
    
    '''
    so a pytorch model basically requires an image in a tensor format, however we will be preprocessing it till an intermediate step
    and will be taking an image_path (ie, path to the image's directory and its name), applying some transformations and making it
    into a numpy array. 
    '''
    
    #using the PIL (or pillow library) for opening the image    
    img = Image.open(image) 
    
    '''
    so now, we will be defining certain transformations
    1. resize: this will take the shortest side and make it 255 pixels, maintaining the ratio throughout
    2. centrecrop: this will crop the image to a centre square of 224*224 pixels
    3. toTensor(): converts it to a tensor
    4. normalize: so these are the mean and standard deviations for the RGB channels, we normalise using these
    
    Why normalize??
    Models pre-trained on ImageNet, such as ResNet, VGG, EfficientNet, and others, were trained with this normalization in place.
    To leverage the learned features from these models accurately, the input images need to undergo the same normalization, 
    helping the model interpret new images in a way that aligns with the images it was trained on.
    
    '''
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    #converting to an np array and returning it
    img = np.array(transform(img))
    return img


#########################################################################

def imshow(image, ax=None, title=None):
    
    '''
    This takes up the np array we have from the process_image function and displays it.
    '''

    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # This was one of the transformations we did in process_image, and in order to get the original pixel values for each channel
    # we again denormalise it
    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    image = std_vector * image + mean_vector
    
    #if they have given some input title, then you can set the image title to it
    if title is not None:
        ax.set_title(title)
        
    # if the values are outside the range (0,1) it will appear as noise,so we clip it
    # here, np.clip clamps the value between 0 and 1
    image = np.clip(image, 0, 1)
    
    #show the imagee
    ax.imshow(image)
    
    return ax

#########################################################################


def predict(image_path, model, cat_to_name, top_k=5):
    ''' 
    now we get down to predicting the image. first, we do the technicalities: setting it to cpu mode and eval mode
    then, we process the input image using pil library(done using process_image function)
    then, we pass it through the model, convert into appropriate probabilities and return the top_k classes and indices
    '''
    
    # since we are just predicting, we need not need gpu, just cpu
    model.to("cpu")
    
    '''
    here, we are setting the model to eval mode so that it satisfies the given two functionalities
    1. if you remember,we had dropped certain values(ie, set certain neuron output to 0 during forward pass) while building
       the model, because we didnt want it to overfit. however, during testing we would want the entire network to be active
       so as to make accurate predictions, therefore its necessary to explicitely mention and convert to eval mode
    2. batch normalization: during testing, it computes the error& accuracy etc statistics for each batch, but while testing
       we want it to be cumulative so again,we explicitely mention that
    '''
    model.eval();

    # now, process_image returns the image as a np array, we just change that to a pytorch tensor
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to("cpu")

    # now, model.forward(image) and model(image) does the same thing. here, since we have used logsoftmax, it applies the logarithmic
    # function on the softmax probabilties obtained because they might be extremly small and this leads to numerical instability
    # inorder to undo that step, we just apply the exponential function, so that the log gets cancelled and we obtain the original
    # probability
    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    #print(linear_probs)

    # now, we just need the top_k values
    #Detaching is useful in PyTorch when you don’t want to track the operations for gradients, which is typically the case
    #during inference (when you're just making predictions, not training).
    top_probs, top_labels = linear_probs.topk(top_k)
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # now that we have the indices, we need the corresponding classes. since class_to_index is already saved, we just reverse it.
    # then we have the top_labels and the top_flowers, we just return it
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    return top_probs, top_labels, top_flowers


#########################################################################

def print_probability(probs, flowers):
    #now here, we are having the zip(flowers,probs) function which basically creates an iterator of tuples where each tuple
    # contains the flower name and its corresponding probability. 
    # The enumerate function helps to iterate over where i stands for the index and j stands for the tuple
    for i, j in enumerate(zip(flowers, probs)):
        # here, since its 0-based indexing, we print i+1 for the rank and do j[0]*100 for the probability 
        print ("Rank {}:".format(i+1),
               "Flower Name: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
    return
        
#########################################################################

def main():
    
    args = arg_parser() 
    with open(args.category_names, 'r') as f:
    	cat_to_name = json.load(f)
                
    # first, we construct the model and then load all the weights and biases into the classifier part
    model = loading_the_checkpoint(args.checkpoint)
    top_k= args.top_k
    # now, we predict the image and then print the probability
    top_probs, top_labels, top_flowers = predict(args.image, model,cat_to_name,top_k) 
    print_probability(top_flowers, top_probs)

########################################################################

'''
so like, if this py script is being run directly, then we would want the main() function to get executed, however if it is
being imported as a module in some other py file, then the __name__ is set to that module name else __main__. 
so if the __name__ is set to that module name, then basically, you would just want to import all the function definitions
and not execute any test code.
'''

if __name__ == '__main__': main()

# usage: python try_predict.py --image flowers/test/37/image_03734.jpg --checkpoint model_checkpoint.pth --top_k 8
