import torch
import torch.nn as nn
import torchvision.models as models


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

        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):

        # get the batch size
        batch_size = features.shape[0]

        # embed the captions and remove the <end>
        embeds = self.embedding(captions[:,:-1])

        # initial hidden state and initial cell state should be zero
        h0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()

        # Concatenates the features and embeds in one tensor
        # and unsqueeze features to shape: batch_size x 1 x embed_size
        input = torch.cat((features.unsqueeze(1), embeds), 1)

        # run the LSTM
        output, _ = self.rnn(input, (h0, c0))

        # the last linear layer
        output = self.hidden2tag(output)

        return output


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        # get the batch size from the inputs
        batch_size = inputs.shape[0]

        # initial hidden state and initial cell state should be zero
        hn = torch.zeros(1, batch_size, self.hidden_size).cuda()
        cn = torch.zeros(1, batch_size, self.hidden_size).cuda()

        # output is an empty list
        output = []

        for i in range(max_len):
            if i == 0:
                x = inputs  # first time assign the CNN feature vector
            else:
                x = self.embedding(x)  # otherwise assign the token ID

            # run the LSTM once
            x, (hn, cn) = self.rnn(x, (hn, cn))

            # the linear layer
            x = self.hidden2tag(x)

            # the index of the maximum value is the token ID
            x = torch.argmax(x, dim=2)

            # to integer
            idx = int(x.cpu().data.numpy()[0][0])

            output.append(idx)
            if idx == 1: # <end>
                break

        return output