import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

torch.manual_seed(1)
random.seed(1)

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
    def forward(self, x):
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
  
