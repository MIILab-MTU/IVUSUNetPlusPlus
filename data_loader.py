import numpy as np
from glob import glob
import helpers
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random


def load_data_path(data_dir,cv, cv_max, mode):
    assert cv < cv_max
    patient_list = os.listdir(data_dir)
    np.random.seed(seed=11000)
    p_index = np.random.permutation(len(patient_list))
    tr_val_index = p_index[:-2]
    tr_val_num_data = len(tr_val_index)
    num_per_cv = int(tr_val_num_data / cv_max)
    train_patient_list = []
    val_patient_list = []
    for i in range(tr_val_num_data):
        if i not in list(range(cv * num_per_cv, (cv + 1) * num_per_cv)):
            train_patient_list.append(patient_list[p_index[i]])
        else:
            val_patient_list.append(patient_list[p_index[i]])
    test_index = p_index[-2:]
    test_patient_list = []
    for i in test_index:
        test_patient_list.append(patient_list[i])
    print("loading..............data")
    train_data_x = []
    train_data_y = []
    for patient in train_patient_list:
        train_data_x_patient = glob(data_dir+"/{}/train/*.jpg".format(patient))
        train_data_y_patient = glob(data_dir + "/{}/{}/*.png".format(patient,mode))
        train_data_x_patient = sorted(train_data_x_patient)
        train_data_y_patient = sorted(train_data_y_patient)
        assert len(train_data_x_patient) == len(train_data_y_patient)
        train_data_x.extend(train_data_x_patient)
        train_data_y.extend(train_data_y_patient)
    print("n_image_files_for_train:%d, n_label_files_for_train:%d" % (
        len(train_data_x), len(train_data_y)))

    val_data_x = []
    val_data_y = []
    for patient in val_patient_list:
        val_data_x_patient = glob(data_dir + "/{}/train/*.jpg".format(patient))
        val_data_y_patient = glob(data_dir + "/{}/{}/*.png".format(patient, mode))
        val_data_x_patient = sorted(val_data_x_patient)
        val_data_y_patient = sorted(val_data_y_patient)
        assert len(val_data_x_patient) == len(val_data_y_patient)
        val_data_x.extend(val_data_x_patient)
        val_data_y.extend(val_data_y_patient)
    print("n_image_files_for_val:%d, n_label_files_for_val:%d" % (
        len(val_data_x), len(val_data_y)))

    test_data_x = []
    test_data_y = []
    for patient in test_patient_list:
        test_data_x_patient = glob(data_dir + "/{}/train/*.jpg".format(patient))
        test_data_y_patient = glob(data_dir + "/{}/{}/*.png".format(patient, mode))
        test_data_x_patient = sorted(test_data_x_patient)
        test_data_y_patient = sorted(test_data_y_patient)
        assert len(test_data_x_patient) == len(test_data_y_patient)
        test_data_x.extend(test_data_x_patient)
        test_data_y.extend(test_data_y_patient)
    print("n_image_files_for_test:%d, n_label_files_for_test:%d" % (
        len(test_data_x), len(test_data_y)))


    return train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y

def load_data(data_dir,cv, cv_max, mode):
    train_x_data_paths, train_y_data_paths, val_x_data_paths, val_y_data_paths, test_x_data_paths, test_y_data_paths = load_data_path(data_dir,cv,cv_max, mode)
    train_x_images = helpers.load_image_batch(train_x_data_paths)
    train_y_images = helpers.load_image_batch(train_y_data_paths,label = True)
    r = random.random
    random.seed(2)
    random.shuffle(train_x_images, random=r)
    random.seed(2)
    random.shuffle(train_y_images, random=r)
    random.seed(2)
    random.shuffle(train_x_data_paths,random=r)
    val_x_images = helpers.load_image_batch(val_x_data_paths)
    val_y_images = helpers.load_image_batch(val_y_data_paths, label=True)
    test_x_images = helpers.load_image_batch(test_x_data_paths)
    test_y_images = helpers.load_image_batch(test_y_data_paths,label = True)

    data_images = {"train_x_images": train_x_images,
                   "train_y_images": train_y_images,
                   "val_x_images": val_x_images,
                   "val_y_images": val_y_images,
                   "test_x_images": test_x_images,
                   "test_y_images": test_y_images}

    train_filenames = []
    val_filenames = []
    test_filenames = []
    for train_file_path in train_x_data_paths:
        train_filenames.append(train_file_path[train_file_path.rfind("data")+5: train_file_path.rfind("train")-1]+ "_"+ train_file_path[train_file_path.rfind("train")+6: train_file_path.rfind(".")])
    for val_file_path in val_x_data_paths:
        val_filenames.append(val_file_path[val_file_path.rfind("data")+5: val_file_path.rfind("train")-1]+ "_"+ val_file_path[val_file_path.rfind("train")+6: val_file_path.rfind(".")])
    for test_file_path in test_x_data_paths:
        test_filenames.append(test_file_path[test_file_path.rfind("data")+5: test_file_path.rfind("train")-1]+ "_"+ test_file_path[test_file_path.rfind("train")+6: test_file_path.rfind(".")])


    data_names = {"train_filenames": train_filenames,
                  "val_filenames":val_filenames,
                  "test_filenames": test_filenames}

    return data_images, data_names
