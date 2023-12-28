import pandas as pd
from PIL import Image
import io
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



LOAD = False

df = pd.read_parquet('data/data1.parquet', engine='fastparquet')

rows = df.shape[0]

cntry = df.iloc[:, 0].values
cntry_unique = np.unique(cntry, return_inverse=True, return_index=True)
cntry_str = cntry_unique[0]
cntry_int = cntry_unique[1]

im_bytes = df.iloc[:, 4].values
im_shape = np.array(Image.open(io.BytesIO(im_bytes[0]))).shape

if LOAD:
    base_height = 128
    hpercent = (base_height / float(im_shape[0]))
    wsize = int((im_shape[1] * float(hpercent)))

    ims = np.zeros((rows, base_height, wsize, 3))
    for i, im in enumerate(im_bytes):
       ims[i] = np.array(Image.open(io.BytesIO(im)).resize((wsize, base_height), Image.Resampling.LANCZOS))
    
    np.save('data.npy', ims)
else:
    ims =  np.load('data.npy')

rand_ng = np.random.default_rng()
perm = rand_ng.permutation(rows)

ims = torch.tensor(ims[perm], dtype=torch.float32)
cntry = cntry[perm]

if torch.cuda.is_available():
  torch.set_default_device('cuda')
  device = 'cuda'
else:
  device = 'cpu'

ims = ims.to(device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(78880, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 53)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


cntry_new = []
for i in range(cntry.shape[0]):
    cntry_new.append(cntry_str.tolist().index(cntry[i]))
cntry_new = torch.tensor(cntry_new)

print(torch.max(cntry_new))

for epoch in range(5):  # loop over the dataset multiple times

    print(f'new epoch')
    running_loss = 0.0
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = ims.transpose(1,3).transpose(2,3), cntry_new

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    c = 0
    for i in range(cntry.shape[0]):
        c = c + (cntry_str[torch.argmax(net(ims[i].transpose(0,2).transpose(1,2)[None, :]))] == cntry[i])
    print(c)


    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')