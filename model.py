import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        self.vocab_size = vocab_size
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        #self.init_weights()

    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a caption (excluding <end> token
        embeddings = self.word_embeddings(captions[:,:-1])
        
        # get the output and hidden state by passing the lstm over our word embeddings
        
         #Concatenate captions embedidings and images features in one dimension array
        lstm_input = torch.cat((features.unsqueeze(1), embeddings), 1)
        x, (h, c) = self.lstm(lstm_input)
        x = x.view(x.size()[0], x.size()[1], self.hidden_dim)
        x = self.fc(x)
        
        return x
        

        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_ids=[]
        
        for i in range(max_len):
            #pass data through recurrent network
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.fc(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted)

            # prepare chosen words for next decoding step
            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1)
        output_ids = torch.stack(output_ids, 1)
        return output_ids.squeeze().tolist()