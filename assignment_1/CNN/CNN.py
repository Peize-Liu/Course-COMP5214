import numpy as np
import torch.utils.data
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import cv2
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
#load dataset
train_set=datasets.MNIST(root="../../dataset/train_data",train=True,download=False,transform=torchvision.transforms.ToTensor())
test_set=datasets.MNIST(root="../../dataset/test_data",train=False,download=False,transform=torchvision.transforms.ToTensor())




class LENET(nn.Module):
    def __init__(self):
        super(LENET, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.Tanh()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def cnn_test():
    batch_size = 64
    data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    epoch_num = 20
    model = LENET().to('cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fuc = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epoch_num)):
        for i, (train_img, train_label) in enumerate(data_loader_train):
            optimizer.zero_grad()
            pred = model(train_img.to('cuda'))
            loss = loss_fuc(pred,train_label.to('cuda'))
            loss.backward()
            optimizer.step()
    model.eval()
    test_pred = model(test_img.float().unsqueeze(1).to('cuda'))
    acc = (test_pred.argmax(dim=1) == test_label.to('cuda')).sum() / test_label.shape[0]
    print(acc)


train_img, train_label = train_set.data, train_set.targets
test_img, test_label = test_set.data, test_set.targets
cnn_test()
