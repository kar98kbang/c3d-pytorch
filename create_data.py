import os
import skimage.io as io
from skimage.transform import resize
import numpy as np

father_path = "/home/data/imaginist_data/Trajectory/cropped_left/"
patient_list = sorted(os.listdir(father_path))
save_folder = "/home/data/imaginist_data/Trajectory/training_data/"

for patient in patient_list:
    print(patient)
    data_index = 0
    count = 0
    image_list = os.listdir(father_path + patient)
    image_list = sorted(image_list)
    save_path = save_folder + str(patient) + '_L/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    block_list1 = []
    block_list2 = []
    block_list3 = []
    for image in image_list:
        image_path = father_path + patient + '/' + image
        if count % 3 == 0:
            block_list1.append(image_path)
            if len(block_list1) == 16:
                clip = np.array([resize(io.imread(frame), output_shape=(112, 112), preserve_range=True) for frame in block_list1])
                np.save(save_path+str(data_index)+'.npy', clip)
                data_index += 1
                block_list1 = []

        if count % 3 == 1:
            block_list2.append(image_path)
            if len(block_list2) == 16:
                clip = np.array([resize(io.imread(frame), output_shape=(112, 112), preserve_range=True) for frame in block_list2])
                np.save(save_path+str(data_index)+'.npy', clip)
                data_index += 1
                block_list2 = []

        if count % 3 == 2:
            block_list3.append(image_path)
            if len(block_list3) == 16:
                clip = np.array([resize(io.imread(frame), output_shape=(112, 112), preserve_range=True) for frame in block_list3])
                np.save(save_path+str(data_index)+'.npy', clip)
                data_index += 1
                block_list3 = []
        count += 1

        # print(image_path)
