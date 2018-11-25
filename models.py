import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class baselineLSTM(nn.Module):
    def __init__(self, config):
        super(baselineLSTM, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
        
        # Output recurrent layers and initialized
        self.output = torch.nn.LSTM(input_size=config['input_dim'], 
                                    hidden_size=config['output_dim'], num_layers=1, dropout=config['dropout'], 
                                    batch_first=True, bidirectional=config['bidirectional'])
        # Initialized hh weights
        torch_init.xavier_normal_(self.output.weight_hh_l0)
        # Initialize ih weights
        torch_init.xavier_normal_(self.output.weight_ih_l0)
        
    def forward(self, sequence, hc=None):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        if hc == None:
            #Output layer if no hidden/cell
            out, (h,c) = self.output(sequence)
        else:
            #Output layer if hidden/cell
            out, (h,c) = self.output(sequence, hc)
            
        # Apply softmax on the outputs
        return out,(h,c)

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
        
        # Output recurrent layers and initialized
        self.recurrent = torch.nn.LSTM(input_size=config['input_dim'], 
                                    hidden_size=config['hidden_dim'], num_layers=config['layers'], dropout=config['dropout'], 
                                    batch_first=True, bidirectional=config['bidirectional'])
        # Initialized hh weights
        torch_init.xavier_normal_(self.recurrent.weight_hh_l0)
        # Initialize ih weights
        torch_init.xavier_normal_(self.recurrent.weight_ih_l0)
        
         # Initialized hh weights
        torch_init.xavier_normal_(self.recurrent.weight_hh_l1)
        # Initialize ih weights
        torch_init.xavier_normal_(self.recurrent.weight_ih_l1)
        
        # Fully connected output layer to map hidden to output dimensions/classes
        self.output = torch.nn.Linear(in_features=config['hidden_dim'], out_features=config['output_dim'])
        
        # Initialize fully connected weights
        torch_init.xavier_normal_(self.output.weight)

        
    def forward(self, sequence, hc=None):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        if hc == None:
            #Output layer if no hidden/cell
            out, (h,c) = self.recurrent(sequence)
        else:
            #Output layer if hidden/cell
            out, (h,c) = self.recurrent(sequence, hc)
            
        # Apply linear transformation to output
        out = self.output(out.contiguous().view(-1,out.shape[2]))
                         
        # Reshape the output of the linear to match (batch_size x sequence_length x output_dim)
        out = out.view(sequence.shape[0],sequence.shape[1],-1)
        return out,(h,c)
    
class baselineGRU(nn.Module):
    def __init__(self, config):
        super(baselineGRU, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
        
        # Output recurrent layers and initialized
        self.output = torch.nn.LSTM(input_size=config['input_dim'], 
                                    hidden_size=config['output_dim'], num_layers=1, dropout=config['dropout'], 
                                    batch_first=True, bidirectional=config['bidirectional'])
        # Initialized hh weights
        torch_init.xavier_normal_(self.output.weight_hh_l0)
        # Initialize ih weights
        torch_init.xavier_normal_(self.output.weight_ih_l0)
        
    def forward(self, sequence, hc=None):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        if hc == None:
            #Output layer if no hidden/cell
            out, (h,c) = self.output(sequence)
        else:
            #Output layer if hidden/cell
            out, (h,c) = self.output(sequence, hc)
            
        # Apply softmax on the outputs
        return out,(h,c)


class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
        
        # Output recurrent layers and initialized
        self.output = torch.nn.GRU(input_size=config['input_dim'], 
                                    hidden_size=config['output_dim'], num_layers=['layers'], dropout=config['dropout'], 
                                    batch_first=True, bidirectional=config['bidirectional'])
        # Initialized hh weights
        torch_init.xavier_normal_(self.output.weight_hh_l0)
        # Initialize ih weights
        torch_init.xavier_normal_(self.output.weight_ih_l0)
        
         # Initialized hh weights
        torch_init.xavier_normal_(self.output.weight_hh_l1)
        # Initialize ih weights
        torch_init.xavier_normal_(self.output.weight_ih_l1)
        
    def forward(self, sequence, hc=None):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        if hc == None:
            #Output layer if no hidden/cell
            out, (h,c) = self.output(sequence)
        else:
            #Output layer if hidden/cell
            out, (h,c) = self.output(sequence, hc)
            
        # Apply softmax on the outputs
        return out,(h,c)