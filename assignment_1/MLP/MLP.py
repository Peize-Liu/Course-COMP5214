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
train_set=datasets.MNIST(root="../../dataset/train_data",train=True,download=True,transform=torchvision.transforms.ToTensor())
test_set=datasets.MNIST(root="../../dataset/test_data",train=False,download=True,transform=torchvision.transforms.ToTensor())



class MLP(nn.Module):
    def __init__(self, hidden_layer):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, 10),
        )

    def forward(self, x):
        out = self.model(x)
        return out

def mlp_test():
    batch_size = 64
    data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    # data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    epoch_num = 20
    acc_all = []
    for hidden_layer in [4,8,16,32,64,128,256]:
        model = MLP(hidden_layer).to('cuda')
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fuc = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epoch_num)):
            for i, (train_img, train_label) in enumerate(data_loader_train):
                optimizer.zero_grad()
                pred = model(train_img.reshape(train_img.shape[0],-1).to('cuda'))
                loss = loss_fuc(pred,train_label.to('cuda'))
                loss.backward()
                optimizer.step()
        model.eval()
        test_pred = model(test_img.reshape(test_img.shape[0],-1).float().to('cuda'))
        acc = (test_pred.argmax(dim=1) == test_label.to('cuda')).sum() / test_label.shape[0]
        acc_all.append(acc.detach().cpu().numpy())
    acc_all = np.array(acc_all)
    plt.plot(np.array([4,8,16,32,64,128,256]), acc_all)
    plt.xscale('log',base=2)
    plt.show()

train_img, train_label = train_set.data, train_set.targets
test_img, test_label = test_set.data, test_set.targets
mlp_test()
