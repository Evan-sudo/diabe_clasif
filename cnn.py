import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pylab as plt
from pic import *
import math
import random

train_test_ratio = 0.8
# Set the folder path where the Excel files are stored
#folder_path = [r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/ContrGr", r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/DiabGr"]
## size of training batch
batch_size = 20   

#train, label =  data2img(folder_path)
train1 = np.load("./data/train.npy") 
label1 = np.load("./data/label.npy") 
c = list(zip(train1,label1))
random.shuffle(c)
train1, label1 = zip(*c)
print(np.shape(train1))
print(np.shape(label1))

ratio_len = math.ceil(train_test_ratio*len(train1))
train = train1[0:ratio_len]
test = train1[ratio_len+1:]
label = label1[0:ratio_len]
label2 = label1[ratio_len+1:]

## normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])                    

class GetLoader(Dataset):
    def __init__(self, data_root, data_label,transform = transform):
        self.data = data_root
        self.label = data_label
        self.transform = transform
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        if self.transform is not None:
            #data = torch.from_numpy(data)
            data = self.transform(data)
            labels = torch.from_numpy(labels)
            #labels = self.transform(labels)
        return data, labels
    def __len__(self):
        return len(self.data)

## processing of dataset
train_dataset = GetLoader(train, label, transform=transform)
test_dataset = GetLoader(test, label2, transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)                  
test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=batch_size)


## define the architecture of convolutional neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 24,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(24, 48,padding=2,kernel_size=5)
        self.conv3 = torch.nn.Conv2d(48, 64,padding=2,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(64, 2)

    def forward(self, x):
        data_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(data_size, -1)
        x = self.fc(x)
        return x


net = Net()
print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    ## use CUDA to accelerate training
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.6)


def train(epoch):
    print('current epoch= %d' % (epoch+1))
    running_loss = 0.0
    data_loss=[]
    for i, data in enumerate(train_loader):
        inputs, target = data
        #print(inputs)
        target = target.squeeze()
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        data_loss.append(loss.item())
        if i % 200 == 199:
            print('[%d,%.5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    return data_loss

def test():
    correct = 0
    total = 0
    data_acc=[]
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            target = target.squeeze()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            data_acc.append(100 * correct / total)
    print('Accuracy on test set:%d %% [%d %d]' % (100 * correct / total, correct, total))
    return data_acc


if __name__ == '__main__':
    
    training_epoch= 9
    data_loss = []
    data_acc = []

    for epoch in range(training_epoch):
        data_loss=data_loss+train(epoch)
        data_acc=test()+data_acc
    iter=np.arange(len(data_loss))
    print("Training finished!")
    plt.plot(len(train_loader)*training_epoch,data_loss,color='r',label='loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title("Training loss")
    plt.legend()
    plt.show()
    
