from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import csv

FramePath = "/home/data/imaginist_data/Trajectory/training_data/"
LabelPath = "./label.csv"

class ParkinsonDataset(Dataset):
    def __init__(self, data_type='filter'):
        self.data_path = []
        self.label = []
        with open(LabelPath) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:  # 将csv文件中的数据保存到birth_data中
                # print(row)
                patient_path = FramePath + row[0] + "_" + row[1] + "/"
                for data_index in sorted(os.listdir(patient_path)):
                    self.data_path.append(patient_path+data_index)
                    self.label.append(int(row))


    def __getitem__(self, index):
        # 按照序号index返回数据和标签
        # return self.data[index]
        return self.data_path[index], self.label[index]
    def __len__(self):
        # 返回数据库长度
        return len(self.label)


if __name__ == "__main__":
    dset_train = ParkinsonDataset()
    train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=0)
    print("Training Data : ", len(train_loader.dataset))
    for batch_idx, data, label in enumerate(train_loader):
        print(data, label)