
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from models import *
from configs import cfg
import pandas as pd
from nltk.translate import bleu_score
import pickle
import sys


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
    
    # No need to pad, we can just stack 
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
    
    # Get the dictionary to translate between ASCII and onehot index
    with open("ASCII2oneHot.pkl", "rb") as f:
        translate = pickle.load(f)
    
    # Get the dictionary to translate between beer style and index
    with open("BeerDict.pkl", "rb") as f:
        beers = pickle.load(f)
        
    tostk = []
    # Take each row and establish the metadata||<SOS> 
    for idx,rev in data.iterrows():
        tostk.append(torch.Tensor([beers[rev['beer/style']]] + [rev['review/overall']] + 
                                     [translate[0]]))
                        
    # Stack the tensors
    stked = torch.stack(tostk)
    del tostk
    
    # Pass back the meta data to concatenate in each time step
    orig = stked
    stked, start = char2oh(np.array(stked), translate, beers)
    
    return torch.Tensor(stked), orig
        

    
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
            
            del output
            del labels
            del loss
            
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
    Criterion = torch.nn.CrossEntropyLoss() # We'll use cross entropy
    Optimizer = optim.Adam(model.parameters(), lr=l_rate, weight_decay=penalty) # Let's use ADAM
    
    # Size of each batch
    trainbatchSize = 32
    validbatchSize = 32
    
    # Create the batch iterator for the data
    trainIter = getBatchIter(X_train, trainbatchSize)
    validIter = getBatchIter(X_valid, validbatchSize)
    
    # For graphs
    all_loss = []
    v_loss = []
    
    # Early stopping conditions
    thresh = 2
    spikes = 0
    
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
            
            # Reset the gradients of the graph
            Optimizer.zero_grad()
            # Get loss and compute gradients
            loss = Criterion(output,labels)
            loss.backward()
            
          
            # Clip the gradient so it doesn't explode
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # Optimize step
            Optimizer.step()
            batch_loss += float(loss)
            
            # Delete unneeded references
            del loss
            del output
            del batch
            del labels
            
            # Progress bar
            if batch_count % 50 == 0:
                batch_loss /= 50
                print("On batch %d with loss %f" % (batch_count, batch_loss))
                all_loss.append(batch_loss)
                batch_loss = 0
                
                # Flush the write buffer
                sys.stdout.flush()
                
            # Save model checkpoint
            if batch_count % 1000 == 0:
                # Model checkpoint
                torch.save(model.state_dict(), 'ModelCheckpoints/GRU.mdl')   
            
            # Implement validation
            if batch_count % 20000 == 0:
                # Validate and save
                vloss = validate(model, validIter, X_valid)
                print("Validation on epoch %d on batch % has loss %f" % (e,batch_count,vloss))
                v_loss.append(vloss)
                
                # If there's more than one loss, we can start comparing
                # to check for early stopping
                if len(v_loss) > 1:
                    
                    # If we see an increase, we add 1 to the counter
                    if v_loss[-1] > v_loss[-2]:
                        spike += 1
                    # Else, we reset the counter
                    else:
                        spike = 0
                    
                    # If we have continously spiked >=- thresh, we stop
                    if spike >= thresh:
                        print("Early stopping on epoch %d" % e)
                        return
            
            
        print("Completed epoch %d" % e)
    
    print("Completed Training after %d epochs" % e)
        
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    
    # Get the dictionary to translate between ASCII and onehot index
    with open("ASCII2oneHot.pkl", "rb") as f:
        translate = pickle.load(f)
    
    # Get the dictionary to translate between beer style and index
    with open("BeerDict.pkl", "rb") as f:
        beers = pickle.load(f)
        
    # Get the dictionary to translate the indices back to ASCII
    with open("oneHot2ASCII.pkl", "rb") as f:
        ASCII = pickle.load(f)
        
    allGenerated = []
    with torch.no_grad():
        # Get iterator
        testIter = getBatchIter(X_test, batchSize = 1000)
    
        for batch_count, batchInd in enumerate(testIter, 1):
            print("Generating on batch %d" % batch_count)
             # Get the dataframe for the batch
            batchFrame = X_test.iloc[batchInd]

            # Process the batch for test data
            # Orig is the array of meta data and <SOS> character
            batch, orig = process_test_data(batchFrame)
            batch = batch.to(computing_device)
            
            # Starting characters and meta datas for the next time step
            start = orig[:,-1]
            metas = orig[:,0:2]
            
            # The generated strings so far
            generated = np.array(start).reshape(-1,1)
            
            probs, hc = model(batch)
            # Softmax over the input dim to get the probabilities
            # Divide by temperature to implement the temperature softmax
            probs = torch.nn.functional.softmax(probs/cfg['gen_temp'], dim = 2)

            # Sample from our batchsize of probabilities to get batchsize of next output
            sampled = np.array(torch.distributions.Categorical(probs).sample())

            # Concatenate the sampled to our generated
            generated = np.c_[generated,sampled].astype(int)

            # Make our next input
            newInput,_ = char2oh(np.c_[metas, sampled], translate, beers)
            batch = torch.Tensor(newInput)
            batch = batch.to(computing_device)
            
            # Current to length of the strings
            curlen = 2
            # While not all batches have at least one EOS we continue generating.
            while(not((generated == 110).any(axis=1).all())):
                # If the string length is greater than the max allowed length, break
                if curlen > cfg['max_len']:
                    break
                # Batch is in the shape of batchSize x 1 x input dim
                # Feed the batch to get the outputs
                probs, hc = model(batch, hc)

                # Softmax over the input dim to get the probabilities
                # Divide by temperature to implement the temperature softmax
                probs = torch.nn.functional.softmax(probs/cfg['gen_temp'], dim = 2)

                # Sample from our batchsize of probabilities to get batchsize of next output
                sampled = np.array(torch.distributions.Categorical(probs).sample())

                # Concatenate the sampled to our generated
                generated = np.c_[generated,sampled].astype(int)
                # Increase length
                curlen += 1
                
                # Make our next input
                newInput,_ = char2oh(np.c_[metas, sampled], translate, beers)
                batch = torch.Tensor(newInput)
                batch = batch.to(computing_device)
            
            # We don't need the first and last characters/ <SOS> and <EOS>
            allGenerated += generated[:,1:-1].tolist()
        
        # Process each sentence back to ASCII, remove trailing <EOS>
        allGenerated = [''.join([chr(ASCII[c]).strip('\x01') for c in s]) for s in allGenerated]
        
        
        return allGenerated
        
    
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
    
    model = GRU(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)
    
    train(model, X_train, X_valid, cfg) # Train the model
    #outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    #save_to_file(outputs, out_fname) # Save the generated outputs to a file

