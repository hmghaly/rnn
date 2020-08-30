import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

torch.manual_seed(1)
random.seed(1)

#network definition
#setting up the RNN to accept a sequence of freq values at each time step, and predict the corresponding phoneme
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_depth,number_layers, batch_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.lstm = nn.LSTM(input_size, hidden_size,number_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_size, output_depth)
        self.hidden = self.init_hidden()

    def forward(self, feature_list): #emeds are the list of features for each word in the sentece
        #sent_size=len(embeds)
        lstm_out, _ = self.lstm( feature_list.view(len( feature_list), 1, -1))
        tag_space = self.hidden2out(lstm_out.view(len( feature_list), -1))
        #print(tag_space.view([1,1,1]))
        tag_scores = torch.sigmoid(tag_space)
        #return tag_scores
        return tag_space
       
    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size),
                torch.zeros(1, self.batch_size, self.hidden_size))  
    

#get inputs of any shape, gets outputs of a certain size 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size)      
        self.hidden2out=nn.Linear(hidden_size,output_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1) #new
    def forward(self, x): #need to iterate over each item in the input tensor
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        outs = self.out(lstm_out)
        return outs
    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size),
                torch.zeros(1, self.batch_size, self.hidden_size))

n_input=4
n_hidden=16
n_output=10
rnn = RNN(n_input, n_hidden, n_output)
loss_func = nn.MSELoss()
LR=0.005
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters

#test it on data generated from random numbers
for k in range(3000):
  a=random.randint(0,9) #start from a random number
  rand_tensor = 0.2*torch.rand((3, 4)) + a #generating input tensor from the random number, that consists of random numbers +/- 0.1 of the random number
  outcome=[0.]*n_output #initializing outcome tensor
  outcome[a]=1. #filling the index corresponding to the generated random number, which is the outcome
  outcome_tensor=torch.tensor(outcome).view([1,1,n_output]) #convert it to tensor with shape (1,1,size of outcome/output)
  rnn.hidden = rnn.init_hidden()
  rnn.zero_grad()
  for i in range(len(rand_tensor)): #feed the network sequentially with the input tensors
    cur_tensor=rand_tensor[i].view([1,1,n_input])
    output = rnn(cur_tensor)
    #print(output)
  if k>2950: #show the predictions
    print(k, a, rand_tensor.mean())
    #print("our number is",a, outcome_tensor)
    output_list=output.tolist()[0][0]
    #print(output_list)
    max_val_index=[i for i, j in enumerate(output_list) if j == max(output_list)]
    print("predicted output",max_val_index)#, output
    print("--------")
  
  loss = loss_func(output, outcome_tensor) #calculate the loss, difference between the output and the desired outcome tensors
  loss.backward()
  optimizer.step()
  
###################################################
#here the size of the output is the same as the size of the input
#the depth of the output depends on the number of possible outcome categories (e.g. different phonemes)
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_depth,batch_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.lstm = nn.LSTM(input_size, hidden_size)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_size, output_depth)
        self.hidden = self.init_hidden()

    def forward(self, feature_list): #emeds are the list of features for each word in the sentece
        #sent_size=len(embeds)
        lstm_out, _ = self.lstm( feature_list.view(len( feature_list), 1, -1))
        tag_space = self.hidden2out(lstm_out.view(len( feature_list), -1))
        #print(tag_space.view([1,1,1]))
        tag_scores = torch.sigmoid(tag_space)
        #return tag_scores
        return tag_space
       
    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size),
                torch.zeros(1, self.batch_size, self.hidden_size))  

n_input=4
n_hidden =16

rnn = RNN(n_input, n_hidden, 5)
line_tensor=torch.rand((10, 4))
output = rnn(line_tensor)
print(output)
print(output.shape)
###############################################
#now we want to just do regression to get one value from a tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_size)      
        #self.lstm.cuda(cuda0)  
        self.hidden2out=nn.Linear(hidden_size,output_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1) #new
        
    def forward(self, x):
        #self.hidden
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        outs = self.out(lstm_out)
        #outs = self.softmax(outs)
        return outs#, h_state


    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size),
                torch.zeros(1, self.batch_size, self.hidden_size))

n_input=4
n_hidden = 16
rnn = RNN(n_input, n_hidden, 1)
rnn.hidden = rnn.init_hidden()
rnn.zero_grad()
line_tensor=torch.rand((10, 4))

for i in range(line_tensor.size()[0]):
  cur_tensor=line_tensor[i]#.view([(1, 1, 3)])
  cur_tensor=cur_tensor.view([1,1,n_input])
  output = rnn(cur_tensor)
print(output)


