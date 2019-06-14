from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import csv
import numpy as np
import torch

FramePath = "/home/data/imaginist_data/Trajectory/training_data/"
LabelPath = "./label.csv"

class ParkinsonDataset(Dataset):
    def __init__(self, data_type='train'):
        self.data_path = []
        self.label = []
        if data_type == 'train':
            with open(LabelPath) as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                count = 0
                for row in csv_reader:  # 将csv文件中的数据保存到birth_data中
                    # print(row)
                    if count >= 80:
                        break
                    patient_path = FramePath + row[0] + "_" + row[1] + "/"
                    for data_index in sorted(os.listdir(patient_path)):
                        self.data_path.append(patient_path+data_index)
                        self.label.append(int(row[2]))
                    count += 1

        if data_type == 'test':
            print("im in")
            with open(LabelPath) as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                count = 0
                for row in csv_reader:  # 将csv文件中的数据保存到birth_data中
                    # print(count, row)
                    if count < 80:
                        count += 1
                        continue
                    patient_path = FramePath + row[0] + "_" + row[1] + "/"
                    for data_index in sorted(os.listdir(patient_path)):
                        self.data_path.append(patient_path+data_index)
                        self.label.append(int(row[2]))
                    count += 1


    def __getitem__(self, index):
        clip = np.load(self.data_path[index])
        clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
        # clip = np.expand_dims(clip, axis=0)  # batch axis
        clip = np.float32(clip)
        label = np.zeros((5,))
        label[self.label[index]] = 1
        return torch.from_numpy(clip), torch.from_numpy(label)

    def __len__(self):
        # 返回数据库长度
        return len(self.label)


if __name__ == "__main__":
    dset_train = ParkinsonDataset(data_type='train')
    train_loader = DataLoader(dset_train, batch_size=1, shuffle=False, num_workers=0)
    print("Training Data : ", len(train_loader.dataset))
    # for batch_idx, (data, label) in enumerate(train_loader):
    #     print(data.size(), label)
    test_data = ParkinsonDataset(data_type='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    print("Training Data : ", len(test_loader.dataset))
