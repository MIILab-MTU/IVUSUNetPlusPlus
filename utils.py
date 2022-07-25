from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
import subprocess
# from scipy.misc import imread
from imageio import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

import helpers
from evaluator import evaluate_single_image
import cv2
from visualize import DataVisualizer


def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])


# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou


def tv_loss(img, tv_weight):
    """
    Total Variation Loss
    :param img:
    :param tv_weight:
    :return:
    """
    left_loss = tf.reduce_sum((img[:, 1:, :, :] - img[:, :-1, :, :])**2)
    down_loss = tf.reduce_sum((img[:, :, 1:, :] - img[:, :, :-1, :])**2)
    loss = tv_weight * (left_loss + down_loss)
    return loss

def dice_coef_loss(prob, label):
    flat_label = tf.layers.flatten(label)
    flat_prob = tf.layers.flatten(prob)
    intersection = tf.reduce_mean(2*tf.multiply(flat_prob, flat_label))
    union = tf.reduce_mean(tf.add(flat_prob, flat_label))
    loss = 1 - tf.div(intersection, union)
    return loss


def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])

def trans_img(imgs):
    imgs = imgs.transpose((2,0,1,3))
    saveimg = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2],3),dtype=float)
    for i in range(imgs.shape[0]):
        img = np.argmax(imgs[i], axis=2)
        saveimg[i][img == 0] = [0, 0, 0]
        saveimg[i][img == 1] = [0, 0, 128]
        saveimg[i][img == 2] = [0, 128, 0]
        saveimg[i][img == 3] = [0, 128, 128]
        saveimg[i][img == 4] = [128, 0, 0]
        saveimg[i][img == 5] = [128, 0, 128]
    return saveimg.transpose((1,2,0,3))


def evaluate_results( gt, y_pred, patient_ids, x):
    evaluation_list = []
    auc_roc_list = []
    auc_pr_list = []
    dice_coeff_list = []
    acc_list = []
    sensitivity_list = []
    specificity_list = []
    binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
        evaluate_single_image(y_pred[0], gt[0])
    auc_roc_list.append(auc_roc)
    auc_pr_list.append(auc_pr)
    dice_coeff_list.append(dice_coeff)
    acc_list.append(acc)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    pred_bin = binary_image.transpose((3,0,1,2))
    gt_trans = np.squeeze(gt).transpose((3,0,1,2))
    for i in range(gt_trans.shape[0]):
        pred_area = np.expand_dims(pred_bin[i],axis=3)
        gt_area = np.expand_dims(gt_trans[i],axis=3)
        _, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
            evaluate_single_image(pred_area, gt_area)
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        dice_coeff_list.append(dice_coeff)
        acc_list.append(acc)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
    result = {'patient_id': patient_ids[0],
              'bin': binary_image,
              'x': x[0],
              'gt': gt[0],
              'acc': acc_list,
              'sn': sensitivity_list,
              'sp': specificity_list,
              'dice': dice_coeff_list,
              'auc_roc': auc_roc_list,
              'auc_pr': auc_pr_list}
    evaluation_list.append(result)
    return evaluation_list

def save_result(evaluated_results,save_path,epoch):
    if not os.path.exists(save_path+"/%04d" % (epoch)):
        os.mkdir(save_path+"/%04d" % (epoch))

    target = open(save_path + "/%04d/val.csv" % (epoch), "w")

    target.write('patient,dice_background,dice_Super,dice_Later,dice_Media，dice_Active,dice_Iner,dice_Mean\n')

    for i, evaluated_result in enumerate(evaluated_results):
        Dice_mean = (evaluated_result['dice'][1]+evaluated_result['dice'][2]+evaluated_result['dice'][3]
                     +evaluated_result['dice'][4]+evaluated_result['dice'][5]+evaluated_result['dice'][6])/6
        target.write("%s,%f,%f,%f,%f,%f,%f,%f\n" % (evaluated_result['patient_id'],
                                                 evaluated_result['dice'][1],
                                                 evaluated_result['dice'][2],
                                                 evaluated_result['dice'][3],
                                                 evaluated_result['dice'][4],
                                                 evaluated_result['dice'][5],
                                                 evaluated_result['dice'][6],
                                                 Dice_mean))

        np.save(file=save_path + "/%04d/" % (epoch) +"{}_x.npy".format(evaluated_result['patient_id']),
                arr=evaluated_result['x'])
        np.save(file=save_path + "/%04d/" % (epoch) +"{}_gt.npy".format(evaluated_result['patient_id']),
                arr=evaluated_result['gt'])
        np.save(file=save_path + "/%04d/" % (epoch) +"{}_bin.npy".format(evaluated_result['patient_id']),
                arr=evaluated_result['bin'])
        dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                             trans_img(np.squeeze(evaluated_result['bin'])),
                             trans_img(np.squeeze(evaluated_result['bin'])),
                             trans_img(np.squeeze(evaluated_result['gt']))],
                            save_path= save_path+ "/%04d/" % (epoch) +"{}.png".format( evaluated_result['patient_id']))
        dv.visualize(evaluated_result['x'].shape[2])
    target.close()

