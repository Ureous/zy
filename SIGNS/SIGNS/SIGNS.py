import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from cnn_utils import *
import collections
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

to_tensor = transforms.ToTensor()

X_train = torch.zeros(X_train_orig.shape[0],
                      X_train_orig.shape[3],
                      X_train_orig.shape[1],
                      X_train_orig.shape[2])
for i in range(X_train.shape[0]):
    X_train[i] = to_tensor(X_train_orig[i])

X_test = torch.zeros(X_test_orig.shape[0],
                     X_test_orig.shape[3],
                     X_test_orig.shape[1],
                     X_test_orig.shape[2])
for i in range(X_test.shape[0]):
    X_test[i] = to_tensor(X_test_orig[i])

Y_train = torch.from_numpy(Y_train_orig).permute(1, 0)

Y_test = torch.from_numpy(Y_test_orig).permute(1, 0)

imgs = torch.stack([img_t for img_t in X_train], dim=3)

# imgs.view(3, -1).mean(dim=1)
# 0.7630, 0.7105, 0.6634
# imgs.view(3, -1).std(dim=1)
# 0.1538, 0.1998, 0.2221

normalize = transforms.Normalize((0.7630, 0.7105, 0.6634), (0.1538, 0.1998, 0.2221))

X_train = normalize(X_train)

X_test = normalize(X_test)

data = [(X_train[i], Y_train.squeeze()[i])
        for i in range(X_train.shape[0])]

data_val = [(X_test[i], Y_test.squeeze()[i])
            for i in range(X_test.shape[0])]


class Net(nn.Module):
    def __init__(self, n_chans1=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.n_chans1 = n_chans1
        self.conv11 = nn.Conv2d(1, n_chans1, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv12 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.fc1 = nn.Linear(n_chans1 // 2 * 8 * 8, 32)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        pool1_out = F.max_pool2d(conv1_out, kernel_size=2, stride=2)

        conv2_out = F.relu(self.conv2(pool1_out))
        pool2_out = F.max_pool2d(conv2_out, kernel_size=2, stride=2)

        conv3_out = F.relu(self.conv3(pool2_out))
        pool3_out = F.max_pool2d(conv3_out, kernel_size=2, stride=2)

        conv4_out = F.relu(self.conv4(pool3_out))
        pool4_out = F.max_pool2d(conv4_out, kernel_size=2, stride=2)

        conv5_out = F.relu(self.conv5(pool4_out))

        upconv1_out = F.relu(self.upconv1(conv5_out))
        conv6_in = torch.cat([upconv1_out, conv4_out], dim=1)
        conv6_out = F.relu(self.conv6(conv6_in))

        upconv2_out = F.relu(self.upconv2(conv6_out))
        conv7_in = torch.cat([upconv2_out, conv3_out], dim=1)
        conv7_out = F.relu(self.conv7(conv7_in))

        upconv3_out = F.relu(self.upconv3(conv7_out))
        conv8_in = torch.cat([upconv3_out, conv2_out], dim=1)
        conv8_out = F.relu(self.conv8(conv8_in))

        upconv4_out = F.relu(self.upconv4(conv8_out))
        conv9_in = torch.cat([upconv4_out, conv1_out], dim=1)
        conv9_out = F.relu(self.conv9(conv9_in))

        output = F.sigmoid(self.conv10(conv9_out))

        out = F.max_pool2d(self.act1(self.conv11(output)), 4)
        out = F.max_pool2d(self.act2(self.conv12(out)), 2)
        out = out.view(-1, self.n_chans1 // 2 * 8 * 8)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


def training_loop(n_epochs, optimizer, model, loss_fn,
                  train_loader, val_loader):
    L_train = []
    L_val = []
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs_train, labels_train in train_loader:
            imgs_train = imgs_train.to(device=device)
            labels_train = labels_train.to(device=device)
            outputs = model(imgs_train)
            loss = loss_fn(outputs, labels_train)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            loss = loss + l2_lambda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            L_train.append(loss_train / len(train_loader))

            with torch.no_grad():
                loss_val = 0.0
                for imgs_val, labels_val in val_loader:
                    imgs_val = imgs_val.to(device=device)
                    labels_val = labels_val.to(device=device)
                    outputs = model(imgs_val)
                    loss = loss_fn(outputs, labels_val)

                    l2_lambda = 0.001
                    l2_norm = sum(p.pow(2.0).sum()
                                  for p in model.parameters())
                    loss = loss + l2_lambda * l2_norm
                    loss_val += loss.item()
                L_val.append(loss_val / len(val_loader))
                print('{} Epoch {}, Training loss {}, Validating loss {}'.format(
                    datetime.datetime.now(), epoch,
                    loss_train / len(train_loader),
                    loss_val / len(val_loader)))
    return L_train, L_val


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

train_loader = torch.utils.data.DataLoader(data, batch_size=64,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=8,
                                         shuffle=True)
model = Net().to(device=device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# L_train, L_val = training_loop(n_epochs=300,
#                                optimizer=optimizer,
#                                model=model,
#                                loss_fn=loss_fn,
#                                train_loader=train_loader,
#                                val_loader=val_loader,
#                                )
#
# torch.save(model.state_dict(), 'D:\python\SIGNS.pt')


model2 = Net().to(device=device)
model2.eval()
model2.load_state_dict(torch.load('SIGNS.pt'))

train_loader = torch.utils.data.DataLoader(data, batch_size=64,
                                           shuffle=False)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=64,
                                         shuffle=False)
all_acc_dict = collections.OrderedDict()

def validate(model, train_loader, val_loader):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <1>
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))
        accdict[name] = correct / total
    return accdict

all_acc_dict["baseline"] = validate(model2, train_loader, val_loader)