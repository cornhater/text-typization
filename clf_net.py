from torchvision import transforms
import os
from PIL import Image 
import torch
import numpy as np

# 0 - classic
# 1 - bold

classic_path = r'E:\botay\диплом\classic'
bold_path = r'E:\botay\диплом\bold'

group_len = 20

target = [0]*len(os.listdir(classic_path))+[1]*len(os.listdir(bold_path))
target = torch.LongTensor(target)
target = torch.split(target, group_len)

def ImgToData(classic_path, bold_path):
    data = torch.tensor(())
    transform = transforms.Compose([
      transforms.Resize((40, 40)),
      transforms.ToTensor()])
    classic = os.listdir(classic_path)
    bold = os.listdir(bold_path)
    for file in classic:
        path = os.path.join(classic_path, file)
        img = Image.open(path)
        tensor = transform(img)
        data = torch.cat((data, tensor))
    for file in bold:
        path = os.path.join(bold_path, file)
        img = Image.open(path)
        tensor = transform(img)
        data = torch.cat((data, tensor))
    return data
    
data = ImgToData(classic_path, bold_path)
data = torch.split(data, group_len)

import sklearn
from sklearn import model_selection

data_train, data_test, targets_train, targets_test = sklearn.model_selection.train_test_split(data, target, test_size = 0.2)
data_val, data_test, targets_val, targets_test = sklearn.model_selection.train_test_split(data_test, targets_test, test_size = 0.5)

def SplitData(groups):
    groups = torch.stack(groups, 1)
    groups = groups.reshape(-1, groups.shape[-1], groups.shape[-2])
    return groups

def SplitTargets(groups):
    groups = torch.stack(groups, 1)
    groups = groups.reshape(-1,)
    return groups

data_train = SplitData(data_train)
data_val = SplitData(data_val)
data_test = SplitData(data_test)
targets_train = SplitTargets(targets_train)
targets_val = SplitTargets(targets_val)
targets_test = SplitTargets(targets_test)

data_train = data_train.unsqueeze(1).float()
data_val = data_val.unsqueeze(1).float()
data_test = data_test.unsqueeze(1).float()

print(data_train.shape, targets_train.shape, data_val.shape, targets_val.shape, data_test.shape, targets_test.shape)

class MyLeNet(torch.nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        
        self.conv1_1 = torch.nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=3, padding=0)
        self.conv1_2 = torch.nn.Conv2d(
                in_channels=6, out_channels=6, kernel_size=3, padding=0)
        self.act1  = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=6)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2_1 = torch.nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=3, padding=0)
        self.conv2_2 = torch.nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.act2  = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1   = torch.nn.Linear(7 * 7 * 16, 120)
        self.act3  = torch.nn.ReLU()
        
        self.fc2   = torch.nn.Linear(120, 84)
        self.act4  = torch.nn.ReLU()
        
        self.fc3   = torch.nn.Linear(84, 2)
    
    def forward(self, x):
        
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        
        return x

my_net = MyLeNet()
    
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_net = my_net.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_net.parameters(), lr=10**-3)

batch_size = 25

data_test = data_test.to(device)
targets_test = targets_test.to(device)

for epoch in range(20):
    order = np.random.permutation(len(data_train))
    
    for start_index in range(0, len(data_train), batch_size):
        optimizer.zero_grad()
        
        batch_indexes = order[start_index:start_index+batch_size]
        
        data_batch = data_train[batch_indexes].to(device)
        target_batch = targets_train[batch_indexes].to(device)
        
        preds = my_net.forward(data_batch) 
        
        loss_value = loss(preds, target_batch)
        loss_value.backward()
        
        optimizer.step()

    test_preds = my_net.forward(data_test)
    
    accuracy = (test_preds.argmax(dim=1) == targets_test).float().mean()
    if (epoch%2==0):
        print('Accuracy = {}'.format(accuracy))
        print('Loss = {}\n'.format(loss_value))
        
optimizer.zero_grad()
val_preds = my_net.forward(data_val.to(device))
targets_val = targets_val.to(device)
acc = (val_preds.argmax(dim=1) == targets_val).float().mean()
print('Validation accuracy = {}'.format(acc))