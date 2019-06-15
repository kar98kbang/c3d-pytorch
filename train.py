import numpy as np
import torch.nn
import torch
from C3D_model import C3D
import os
from MyDataset import ParkinsonDataset
from torch.utils.data import DataLoader
os.environ['CUDA_VISIABLE_DEVICES'] = '0,1'


def train():
    C3dNet = C3D()
    C3dNet.cuda()
    C3dNet.train()

    learning_rate = 0.001
    optimizer = torch.optim.SGD(C3dNet.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    dset_train = ParkinsonDataset(data_type='train')

    train_loader = DataLoader(dset_train, batch_size=20, shuffle=True, num_workers=0)

    print("Training Data : ", len(train_loader.dataset))
    print("training start!")

    for epoch in range(400):
        '''
        if epoch>0 and epoch % 20 ==0:
            learning_rate = learning_rate / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        '''
        for batch_index, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            # # # label = label.float()
            predict = C3dNet(data)
            # print("predict and label size: ", predict.size(), label.size())
            loss = loss_func(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {}/399 | batch_index: {} | loss: {}".format(epoch, batch_index, loss.item()))
        if epoch > 0 and (epoch+1) % 100 == 0:
            torch.save(C3dNet.state_dict(), './weights/MyC3dNet{}.pth'.format(epoch+1))

    # torch.save(C3dNet.state_dict(), './weights/MyC3dNet.pth')

if __name__ == '__main__':
    train()
