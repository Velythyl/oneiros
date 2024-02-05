
import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils.dataloader import dataloader

import wandb



class DiscNet(nn.Module):
    def __init__(self, class_num = 3 , lr = 0.001, weight_decay = 0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (10,3))
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = (16,16))
        self.fc1 = nn.Linear(in_features = 208, out_features = 120) # Too bad there is no automatic method
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = class_num)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x


def dis_train(model, trainloader, device, stacksize = 50, epoch=2):
    model.train()
    train_loss = 0
    train_acc = 0
    total_data = 0
    step = -1

    for i in range(epoch):
        for idx, batch in enumerate(trainloader):
            
            # Temporary solution for now
            if batch.shape[0] != stacksize:
                break
            
            
            batch.isinf()


            batch = batch.movedim(0,1) #Due to how the dataloader works

            batch = batch[~batch.isinf().any(dim=1).any(dim=1),:,:]


            
            labels = batch[:,0,-3:].to(device)
            inputs = batch[:,:,:-3].unsqueeze(dim=1).to(device)
            # inputs = batch.unsqueeze(dim=1).to(device) # for testing

            # import pdb; pdb.set_trace()
            model.optimizer.zero_grad()
            outputs = model.forward(inputs)
            
            loss = model.loss_criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            train_loss += loss.item()
            total_data += labels.shape[0]
            train_acc = torch.eq(torch.argmax(outputs, dim=1),torch.argmax(labels,dim=1)).float().mean()
            step += 1
            
            print(f"current index: {idx}, train accuracy: {train_acc}")

            wandb.log({
                "charts/total_train_data_used": total_data,
                "charts/training_loss": train_loss,
                "charts/training_accuracay": train_acc,                
                        }, step=step)


def dis_test(model, testloader, device, stacksize = 50):
    model.eval()
    test_loss = 0
    test_acc = 0
    total_data = 0
    step = -1

    with torch.no_grad():
        for idx, batch in enumerate(testloader):

            # Temporary solution for now
            if (batch.shape[0] != stacksize):
                break

            batch = batch.movedim(0,1)
            batch = batch[~batch.isinf().any(dim=1).any(dim=1),:,:]


            # print(f"current index: {idx}")
            labels = batch[:,0,-3:].to(device)
            inputs = batch[:,:,:-3].unsqueeze(dim=1).to(device)

            # model.optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = model.loss_criterion(outputs, labels)

            test_loss += loss.item()
            total_data += labels.shape[0]
            test_acc = torch.eq(torch.argmax(outputs, dim=1),torch.argmax(labels,dim=1)).float().mean()

            step += 1

            print(f"current index: {idx}, test accuracy: {test_acc}")

            wandb.log({
                "charts/total_test_data_used": total_data,
                "charts/testing_loss": test_loss,
                "charts/testing_accuracay": test_acc,                
                        }, step=step)


def dis_save(Discrmin, path):
    torch.save(Discrmin.state_dict(),path)


def dis_load(path):
    Discrim = DiscNet()
    Discrim.load_state_dict(torch.load(path))
    return Discrim

