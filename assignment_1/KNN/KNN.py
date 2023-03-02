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

def knn_test():
    acc_all = []
    for k in range(1,10):
        print(k)
        classifier = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree',p=1, n_jobs=-1)
        knn = classifier.fit(train_img.reshape(60000,-1), train_label)
        acc = knn.score(test_img.reshape(10000,-1), test_label)
        acc_all.append(acc)
    acc_all = np.array(acc_all)
    plt.plot(np.arange(1,10), acc_all)
    plt.show()


train_img, train_label = train_set.data, train_set.targets
test_img, test_label = test_set.data, test_set.targets
knn_test()
# test_data =np.load("../input/test_data.npy") #import test data
# test_label = np.load("../input/test_labels.npy") #import test labesls