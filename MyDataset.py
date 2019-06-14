from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import csv

FramePath = "./data/"
LabelPath = "./label.csv"

class ParkinsonDataset(Dataset):
    def __init__(self, data_type='filter'):
        self.data = []
        self.label = []
        with open(LabelPath) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:  # 将csv文件中的数据保存到birth_data中
                print(row)
                self.label.append(row)

        patient_list = os.listdir(FramePath)
        for patient in patient_list:
            self.data.append(patient)
        # print(len(data))
            # print(patient)
    # initial coding

    def __getitem__(self, index):
        # 按照序号index返回数据和标签
        # return self.data[index]
        return self.label[index]
    def __len__(self):
        # 返回数据库长度
        return len(self.label)


if __name__ == "__main__":
    dset_train = ParkinsonDataset()
    train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=0)
    print("Training Data : ", len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        print(data)