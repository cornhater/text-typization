from torchvision import transforms
import os
from PIL import Image 
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.datasets

root = r'E:\botay\диплом\font_dataset'
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor()])

class FontDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.bold_root = os.path.join(root, os.listdir(root)[0])
        self.classic_root = os.path.join(root, os.listdir(root)[1])
        self.items = []
        for i in range(len(os.listdir(self.classic_root))):
            self.items.append(os.path.join(self.classic_root, os.listdir(self.classic_root)[i]))
        for i in range(len(os.listdir(self.bold_root))):
            self.items.append(os.path.join(self.bold_root, os.listdir(self.bold_root)[i]))
        self.targets = [0]*len(os.listdir(self.classic_root))+[1]*len(os.listdir(self.bold_root))
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = Image.open(self.items[idx])
        if self.transform:
            item = transform(item)
        target = self.targets[idx]
        return item, target

dataset = FontDataset(root, transform)

def Splitter(dataset, val_size = 0.2, test_size = 0.2):
    order = np.random.permutation(len(dataset))
    val_len = int(val_size*(len(dataset)))
    test_len = int(test_size*(len(dataset)))
    val_set = data.Subset(dataset, order[:val_len])
    test_set = data.Subset(dataset, order[val_len:val_len+test_len])
    train_set = data.Subset(dataset, order[val_len+test_len:])
    return train_set, val_set, test_set

train_set, val_set, test_set = Splitter(dataset)

print('Number of train samples: ', len(train_set))
print('Number of val samples: ', len(val_set))
print('Number of test samples: ', len(test_set))

train_loader = data.DataLoader(train_set, batch_size=20, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=20, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=20, shuffle=True)

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

# Дописать цикл для val и test аналогично train

for epoch in range(100):
    optimizer.zero_grad()
    train_features, train_targets = next(iter(train_loader))
    train_preds = my_net.forward(train_features.to(device))
    train_targets = train_targets.to(device)
    loss_value = loss(train_preds, train_targets)
    loss_value.backward()
    optimizer.step()
    
    val_features, val_targets = next(iter(val_loader))
    val_preds = my_net.forward(val_features.to(device))
    val_targets = val_targets.to(device)
    accuracy = (val_preds.argmax(dim=1) == val_targets).float().mean()
    if (epoch%5==0):
        print('Accuracy = {}'.format(accuracy))
        print('Loss = {}\n'.format(loss_value))
        
optimizer.zero_grad()
test_features, test_targets = next(iter(test_loader))
test_preds = my_net.forward(test_features.to(device))
test_targets = test_targets.to(device)
acc = (test_preds.argmax(dim=1) == test_targets).float().mean()
print('Test accuracy = {}'.format(acc))