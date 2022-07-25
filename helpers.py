import cv2
import numpy as np
import itertools
import operator
import os, csv
import argparse
import tensorflow as tf
from keras.utils import to_categorical


import time, datetime
import utils
import random
import skimage
import SimpleITK as sitk
import matplotlib.pyplot as plt

from skimage import morphology, measure, transform, exposure


_RGB_MEAN = [123.68, 116.78, 103.94]


def search_largest_region(image):
    labeling = measure.label(image)
    regions = measure.regionprops(labeling)

    largest_region = None
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    return largest_region


def generate_largest_region(image):
    region = search_largest_region(image)
    bin_image = np.zeros_like(image)
    if region != None:
        for coord in region.coords:
            bin_image[coord[0], coord[1]] = 1

    return bin_image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_GRAY2BGR)
    return image


def load_image_batch(paths,label = False):
    imgs = []
    for path in paths:
        if label:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img[img==30] = 0
            img[img==215] = 1
        else:
            img = cv2.imread(path)
        img = cv2.copyMakeBorder(img,6,6,6,6,cv2.BORDER_CONSTANT,value=0)
        imgs.append(img)
    return imgs


def filter_image(image, img_filter):
    assert len(image.shape) == 2
    return img_filter(image)



def load_image_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def resize_batch_image(batch_images, resize_factor):
    resized_batch_images = []
    for i in range(batch_images.shape[0]):
        image = batch_images[i]
        new_size = [image.shape[0]/resize_factor, image.shape[1]/resize_factor]
        resized_image = transform.resize(image, new_size)
        resized_batch_images.append(resized_image)

    return np.array(resized_batch_images)

def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


def data_augmentation(input_image, output_image, rotation, move, gause, prob):
    # Data augmentation
    a = np.random.random()
    if rotation and a<=prob:
        angle = random.uniform(-1*rotation, rotation)
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    assert np.max(input_image) < 256 and np.min(input_image) >= 0
    b = np.random.random()
    if move and b<=prob:
        x = np.random.randint(-5,6)
        y = np.random.randint(-5,6)
        M = np.float32([[1, 0, x], [0, 1, y]])
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]))

    c = np.random.random()
    if gause and c <= prob:
        input_image = augment_gaussian_noise(input_image,noise_variance=(0, 0.05))

    return input_image, output_image

def get_train_transform(patch_size,args):

    rotation_angle = args.rotation_angle  #15,
    elastic_deform = args.elastic_deform  #(0, 0.25),
    scale_factor = args.scale_factor  #(0.75, 1.25),
    augmentation_prob = args.augmentation_prob  #0.1
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size,
            patch_center_dist_from_border=args.patch_center_dist_from_border, #[i // 2 for i in patch_size]
            do_elastic_deform=args.do_elastic_deform, deformation_scale=elastic_deform,
            do_rotation=args.do_rotation,
            angle_x=(0,0), #(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
            angle_y=(0,0), #(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
            angle_z=(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
            do_scale=args.do_scale, scale=scale_factor,
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=1,
            random_crop=args.do_random_crop,
            p_el_per_sample=augmentation_prob,
            p_rot_per_sample=augmentation_prob,
            p_scale_per_sample=augmentation_prob
        )
    )

    # now we mirror along all axes
    #tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform(args.brightness,
                                                            per_channel=True,
                                                            p_per_sample=augmentation_prob))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    if args.do_GammaTransform ==True:
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    if args.do_GaussianNoise == True:
        tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    #tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,  p_per_channel=0.5, p_per_sample=0.15))

    # new TODO
    #tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.5, 2), per_channel=True, p_per_sample=0.15))
    #tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# gt = reverse_one_hot(one_hot_it(gt, class_dict))
# gt = colour_code_segmentation(gt, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(gt))
