import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math

train_test_ratio = 0.8

train1 = np.load("./data/train_non.npy") 
label1 = np.load("./data/label_non.npy") 
c = list(zip(train1,label1))
random.shuffle(c)
train1, label1 = zip(*c)
print(np.shape(train1))
print(np.shape(label1))

ratio_len = math.ceil(train_test_ratio*len(train1))
x_train = train1[0:ratio_len]
x_test = train1[ratio_len+1:]
y_train = label1[0:ratio_len]
y_test = label1[ratio_len+1:]

#define neuron network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(17, 30)
        self.fc2 = torch.nn.Linear(30, 10)
        self.fc3 = torch.nn.Linear(10, 1)
        self.sigmond=torch.nn.Sigmoid()
    def forward(self, x):
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        x = self.sigmond(x)
        return x
model = Net()
#model.load_state_dict(torch.load('try.pt'))    # load the model
criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 
x = torch.tensor(x_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.float32)


data_loss = []
for e in range(26000):
    inputs = Variable(x)
    target = Variable(y)
    out = model(inputs) # forward propagation
    out  = torch.squeeze(out)
    #print(out)
    #print(out.shape)
    loss = criterion(out, target) # cal loss
    optimizer.zero_grad() # zero gradient
    #loss.requires_grad_(True)
    loss.backward() 
    optimizer.step() # adjust the parameter
    data_loss.append(loss.item())
    if (e+1) % 1000 == 0: # 每1000次迭代打印一次误差值
        print('Epoch:{}, Loss:{:.5f}'.format(e+1, loss.item()))
torch.save(model.state_dict(), 'try.pt')  # save the model


x2 = torch.unsqueeze(torch.tensor(x_test, dtype=torch.float32), dim=1)
y2 = y_test
print(np.shape(y2))
model.eval() # switch from training mode to test mode
predict = model(Variable(x2)) # prediction
predict = torch.squeeze(predict.data,dim = 1).numpy()  # convert into numpy format
print(np.shape(predict))
predict = np.squeeze(predict)
predict[predict<0.5]=0
predict[predict>0.5]=1

test_size = len(train1)-math.ceil(train_test_ratio*len(train1))
accuracy = np.sum(predict==y2)/(test_size)
print('Accuracy on test set:%d %% [%d %d]' % (100 * accuracy, np.sum(predict==y2),test_size))

plt.plot(np.arange(len(data_loss)),data_loss,color='r',label='loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title("Training loss")
plt.legend()
plt.show()
