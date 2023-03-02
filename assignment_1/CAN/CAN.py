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

class CAN(nn.Module):
    def __init__(self,feature_channel):
        super(CAN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, feature_channel, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, kernel_size=3, dilation=4, padding=4),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, kernel_size=3, dilation=8, padding=8),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(feature_channel, 10, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU())
        self.ave_pool = nn.AvgPool2d(28)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.ave_pool(out)

        return out[:,:,0,0]

def can_test():
    batch_size = 64
    data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    epoch_num = 20
    acc_all = []
    for feature_channel in [4,8,16,32,64,128,256]:
        model = CAN(feature_channel).to('cuda')
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
        right_num = 0
        for i, (test_img, test_label) in enumerate(data_loader_test):
            test_pred = model(test_img.to('cuda'))
            acc = (test_pred.argmax(dim=1) == test_label.to('cuda')).sum() / test_label.shape[0]
            right_num += (test_pred.argmax(dim=1) == test_label.to('cuda')).sum()
        acc = right_num / test_set.targets.shape[0]
        acc_all.append(acc.detach().cpu().numpy())
    acc_all = np.array(acc_all)
    plt.plot(np.array([4,8,16,32,64,128,256]), acc_all)
    plt.xscale('log',base=2)
    plt.show()


train_img, train_label = train_set.data, train_set.targets
test_img, test_label = test_set.data, test_set.targets
can_test()
