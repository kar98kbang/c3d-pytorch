import os

father_path = "/home/data/imaginist_data/Trajectory/cropped_left/"
patient_list = sorted(os.listdir(father_path))
for patient in patient_list:
    print(patient)
    folder_index = 0
    count = 0
    image_list = os.listdir(father_path + patient)
    image_list = sorted(image_list)
    for image in image_list:
        image_path = father_path + patient + '/' + image
        # print(image_path)
