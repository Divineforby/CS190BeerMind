
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from models import *
from configs import cfg
import pandas as pd
from nltk.translate import bleu_score
import pickle


# In[ ]:


def load_data(fname):
    # From the csv file given by filename and return a pandas DataFrame of the read csv.
    
    # Define path to data
    # Note: Path relative to KHIEM's files, make changes to relative path if necessary
    dataPath = 'BeerAdvocatePA4/' + fname
    
    # Read csv into pandas frame
    df = pd.read_csv(dataPath)
    
    # Return frame
    return df

def char2oh(padded,translate, beers):
    # Each row has form: beer style index||overall||character indices
    # TODO: Onehot the beer style and concatenate to overall
    # Onehot the character indices and concatenate the beerstyle||overall to each onehotted char
    
    # Each row's beer has form 1x#ofpossiblebeers
    beerstyles = np.zeros((padded.shape[0],len(beers)))
    # Each row's review has form max(lenofsequence) x #ofpossiblecharacters
    # Since we padded each review, they all have the same number of characters
    # Subtract two since we know the first two values aren't characters
    chars = np.zeros((padded.shape[0], (padded.shape[1] - 2), len(translate)))
    
    # First two columns are beerstyle indices and overalls
    bsidx = padded[:,0]
    ovrl = padded[:,1]
    
    # The rest are characters
    ch = padded[:,2:]
    
    # Index with bsidx
    beerstyles[np.arange(padded.shape[0]), bsidx.astype(int)] = 1
    
    igrid = np.mgrid[0:padded.shape[0], 0:(padded.shape[1]-2)]
    # Index with ch, we use meshgrid since this is a 3d array
    chars[igrid[0], igrid[1], ch.astype(int)] = 1
    
    # Concatenate overall and beer style
    meta_data = np.c_[ovrl, beerstyles]
    
    # Tile and reshape meta_data so we have a copy for each one of the characters
    tiled_meta = np.tile(meta_data, padded.shape[1] - 2).reshape(padded.shape[0], (padded.shape[1] - 2), -1)
    
    # Concatenate the items 
    
    # Return both the concatenated and just the one hot
    return np.c_[tiled_meta, chars],ch
    

def process_train_data(data):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).

    # Get the dictionary to translate between ASCII and onehot index
    with open("ASCII2oneHot.pkl", "rb") as f:
        translate = pickle.load(f)
    
    # Get the dictionary to translate between beer style and index
    with open("BeerDict.pkl", "rb") as f:
        beers = pickle.load(f)
        
    # List of reviews to onehot after translation
    toOnehot = []
    
    # For each review, convert to list of its translated characters
    # Translated means ord(c) -> onehot index
    # Also translate the beer style to its index value
    # Concatenate all the data and convert to tensor
    for idx,rev in data.iterrows():
        if isinstance(rev['review/text'], str):
            toOnehot.append(torch.Tensor([beers[rev['beer/style']]] + [rev['review/overall']] + 
                                         [translate[ord(x)] for x in list(chr(0)+rev['review/text']+chr(1))]))
    
    # Pad all smaller sentences with 1s to signify <EOS>
    padded = pad_data(toOnehot, translate[1])
    del toOnehot

    # Take the array padded sentences and one-hot the characters.
    # Beer style also gets one-hot
    # Overall does not
    reviews,labels = char2oh(np.array(padded), translate, beers)
    del padded
    
    # Since the labels are simply the next characters, we take all characters except the last one
    # for the review, and everything but the first one for the labels
    return torch.Tensor(reviews[:,0:-1,:]), torch.Tensor(labels[:,1:]).type(torch.LongTensor)
    
    
def train_valid_split(data):
    # TODO: Takes in train data as dataframe and
    # splits it into training and validation data.
    
    # List of indices of the data
    ind = np.arange(len(data))
    
    # Randomize the split
    np.random.shuffle(ind)
    
    # Where to split the indices
    # We'll take first 20% for validation, the rest for training
    split = int(0.2*len(data))
    
    # Split the indices
    vIndices = ind[0:split]
    tIndices = ind[split:]
    
    # Group the indices into their frames then return those
    validation_data = data.iloc[vIndices]
    train_data = data.iloc[tIndices]
    
    return train_data,validation_data
    
def process_test_data(data):
    # TODO: Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.
    raise NotImplementedError

    
def pad_data(orig_data, pad):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.
    # Data comes in as translated ASCII representation, simply sort and call torch pad
    return torch.nn.utils.rnn.pad_sequence(sorted(orig_data, key = lambda x: len(x), reverse=True), 
                                           batch_first=True, padding_value=pad)
    

def getBatchIter(data, batchSize):
    # TODO: Returns a list of batches of indices
    # The list of batch indices will be used to index into the
    # corresponding data frame to extract the data
    
    # List of all possible indices
    ind = np.arange(len(data))
    
    # Calculate how many batches of batchSize would fit into
    # into the length of the data
    numBatches = int(len(data)/batchSize)
    
    # Split the array of indices into roughly equivalent batch sized batches
    batchedInd = np.array_split(ind, numBatches)
    
    return batchedInd
    
def validate(model, validIter, X_valid):
    # TODO: Run the model on the entire validation set for loss
    # Loss
    Criterion = torch.nn.CrossEntropyLoss()
    # No need for gradient
    with torch.no_grad():
        totalLoss = 0
        # Validation loop
        for batch_count, batchInd in enumerate(validIter, 0):
             # Get the dataframe for the batch
            batchFrame = X_valid.iloc[batchInd]

            # Process the batch for data and labels
            batch, labels = process_train_data(batchFrame)
            batch, labels = batch.to(computing_device), labels.to(computing_device)
            
            # Run our batch through the model
            # batch has shape Batchsize x Seqlen x Input Dim
            output, (h,c) = model(batch)
            
            # Save space
            del h
            del c
            
            # Reshape the output and labels to so that the loss function
            # can simply interpret time as another batch
            # This will be fine since sum of sum can be thought of as just
            # one sum
            output = output.contiguous().view(-1, output.shape[2])
            labels = labels.view(-1)

            # Get loss and compute gradients
            loss = Criterion(output,labels)
            totalLoss += float(loss)
            
            # Progress bar
            if batch_count % 50 == 0:
                print("Validation: On batch %d" % (batch_count))
                
        # Return the average loss over the batch count
        return totalLoss/(batch_count+1)
            

def train(model, X_train, X_valid, cfg):
    # TODO: Train the model!
    # Datas are given as pandas data frame. Call process on-line as we train to
    # get the data and label
    
    epochs = np.arange(cfg['epochs'])
    l_rate = cfg['learning_rate']
    penalty = cfg['L2_penalty']
    
    
    # Define loss and optimizer
    Criterion = torch.nn.CrossEntropyLoss() # We'll use NLL
    Optimizer = optim.Adam(model.parameters(), lr=l_rate, weight_decay=penalty) # Let's use ADAM
    
    # Size of each batch
    batchSize = 150
    
    # Create the batch iterator for the data
    trainIter = getBatchIter(X_train, batchSize)
    validIter = getBatchIter(X_valid, batchSize)
    
    all_loss = []
    v_loss = []
    # Training loop
    for e in epochs:
        batch_loss = 0
        for batch_count, batchInd in enumerate(trainIter,0):
            # Get the dataframe for the batch
            batchFrame = X_train.iloc[batchInd]

            # Process the batch for data and labels
            batch, labels = process_train_data(batchFrame)
            batch, labels = batch.to(computing_device), labels.to(computing_device)

            # Run our batch through the model
            # batch has shape Batchsize x Seqlen x Input Dim
            output, (h,c) = model(batch)
            
            # Save space
            del h
            del c
            
            # Reshape the output and labels to so that the loss function
            # can simply interpret time as another batch
            # This will be fine since sum of sum can be thought of as just
            # one sum

            output = output.contiguous().view(-1, output.shape[2])
            labels = labels.view(-1)

            # Get loss and compute gradients
            loss = Criterion(output,labels)
            loss.backward()
            
            # Optimize step
            Optimizer.step()
            batch_loss += float(loss)

            # Progress bar
            if batch_count % 50 == 0:
                batch_loss /= 50
                print("On batch %d with loss %f" % (batch_count, batch_loss))
                all_loss.append(batch_loss)
                batch_loss = 0
                
            # TODO: Implement validation
            if batch_count % 3000 == 0:
                # Validate and save
                vloss = validate(model, validIter, X_valid)
                print("Validation on epoch %d on batch % has loss %f" % (e,batch_count,vloss))
                v_loss.append(vloss)
                
                # Model checkpoint
                torch.save(model.state_dict(), 'ModelCheckpoints/BaseLSTM'+str(batch_count+1)+'.mdl')           
            
        print("Completed epoch %d" % e)
    
    print("Completed Training")
        
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    raise NotImplementedError
    
    
def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    raise NotImplementedError
    


# In[ ]:


if __name__ == "__main__":
    train_data_fname = "Beeradvocate_Train.csv"
    test_data_fname = "Beeradvocate_Test.csv"
    out_fname = "model_outputs.out"
    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame
    X_train, X_valid = train_valid_split(train_data) # Splitting the train data into train-valid data
    
    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)
    
    train(model, X_train, X_valid, cfg) # Train the model
    outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    save_to_file(outputs, out_fname) # Save the generated outputs to a file

