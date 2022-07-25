import os
import argparse
import numpy as np
import cv2
import evaluator
import helper_functions
import helpers
import tensorflow as tf
import matplotlib.pyplot as plt
import data_loader
from segmentation_models import Unet, Nestnet, Xnet, FPN_XNET
from keras.optimizers import RMSprop
from helper_functions import str2bool
from tqdm import tqdm
import time
import xlwt

from multi_thread_data_loader import get_generator, get_generator_for_val, get_patients_in_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--mode', type=str, default='lumen')
    parser.add_argument('--n_epoch', type=int, default=201)
    parser.add_argument('--validate_epoch', type=int, default=5)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=8)
    parser.add_argument('--exp_dir', type=str, default="exp_lumen_test")
    parser.add_argument('--backbone', type=str, default='inceptionresnetv2')
    parser.add_argument('--image_size', type=int, default=512)

    # data augmentation
    parser.add_argument('--rotation', type=float, default=15)
    parser.add_argument('--move', type=int, default=0)
    parser.add_argument('--prob', type=float, default=0.1)
    parser.add_argument('--gause', type=helpers.str2bool, default=True)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--train', type=str2bool, default=True)
    # parser.add_argument('--test_dir', type=str, default="./data_test/")
    #parser.add_argument('--test_dir', type=str, default="/home/zhaochen/Desktop/gitclone/UNetPlusPlus-master/SaveImage/train/random selected 10/xiaoqi^zhang")
    args = parser.parse_args()

    data, filenames = data_loader.load_data(data_dir=args.data_path, cv=args.cv, cv_max=args.cv_max, mode=args.mode)
    train_x_images, train_y_images, val_x_images, val_y_images, test_x_images, test_y_images = \
        np.array(data["train_x_images"]), np.array(data["train_y_images"]), np.array(data["val_x_images"]), np.array(
            data["val_y_images"]), np.array(data["test_x_images"]), np.array(data["test_y_images"])
    train_filenames, val_filenames, test_filenames = filenames["train_filenames"], filenames["val_filenames"], \
                                                     filenames["test_filenames"]

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # model
    model = FPN_XNET(backbone_name=args.backbone, encoder_weights='imagenet', decoder_block_type='transpose',
                     input_shape=(args.image_size, args.image_size, 3))  # build UNet++
    model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001),
                  loss=helper_functions.dice_coef_loss,
                  metrics=["binary_crossentropy", helper_functions.mean_iou, helper_functions.dice_coef])

    """
    check_point = keras.callbacks.ModelCheckpoint(os.path.join(args.exp_name, "model_weights.h5"),
                                                  monitor='val_loss', verbose=0,
                                                  save_best_only=True, mode='min')
    """

    if args.train:
        global_evaluation = 0.0
        global_epoch = 0
        for epoch in tqdm(range(0, args.n_epoch)):
            for mini_batch in tqdm(range(int(train_x_images.shape[0] / args.batch_size))):
                training_batch_x = train_x_images[mini_batch * args.batch_size:(mini_batch + 1) * args.batch_size]
                training_batch_y = train_y_images[mini_batch * args.batch_size:(mini_batch + 1) * args.batch_size]
                input_image_batch = []
                output_image_batch = []
                for img_index in range(training_batch_x.shape[0]):
                    input_image, output_image = helpers.data_augmentation(training_batch_x[img_index],
                                                                          training_batch_y[img_index],
                                                                          args.rotation,
                                                                          args.move,
                                                                          args.gause,
                                                                          args.prob)
                    input_image = np.float32(input_image) / 255.0
                    input_image_batch.append(input_image)
                    output_image_batch.append(output_image)
                input_image_batch = np.array(input_image_batch)
                output_image_batch = np.array(output_image_batch)
                model.fit(input_image_batch, output_image_batch, epochs=1, verbose=0)

            # validate
            if epoch % args.validate_epoch == 0:
                print("[x] validation on validation data set")
                avg_auc_roc = []
                avg_auc_pr = []
                avg_dice = []
                avg_acc = []
                avg_sn = []
                avg_sp = []
                pred_images = []
                largest_regions = []

                for img_index in tqdm(range(val_x_images.shape[0])):
                    input_image = val_x_images[img_index]
                    output_image = val_y_images[img_index]

                    input_image = np.expand_dims(input_image, axis=0)
                    input_image = np.array(input_image) / 255.

                    pred_image = model.predict(input_image)
                    binary_vessels = helper_functions.threshold_by_otsu(np.squeeze(pred_image), flatten=False)

                    # evaluate on largest region
                    largest_region = helpers.generate_largest_region(binary_vessels)
                    _, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity \
                        = evaluator.metric_single_img(largest_region, np.squeeze(output_image))

                    avg_auc_roc.append(auc_roc)
                    avg_auc_pr.append(auc_pr)
                    avg_dice.append(dice_coeff)
                    avg_acc.append(acc)
                    avg_sn.append(sensitivity)
                    avg_sp.append(specificity)
                    pred_images.append(pred_image)
                    largest_regions.append(largest_region)

                # if better， save the new evaluation
                if np.mean(avg_dice) > global_evaluation:
                    if not os.path.isdir("%s/%04d" % (args.exp_dir, epoch)):
                        os.makedirs("%s/%04d" % (args.exp_dir, epoch))
                    if not os.path.isdir("%s/%04d/val" % (args.exp_dir, epoch)):
                        os.makedirs("%s/%04d/val" % (args.exp_dir, epoch))
                    target = open("%s/%04d/val_score.csv" % (args.exp_dir, epoch), "w")
                    target.write("filename,auc_roc,auc_pr,dice,acc,sn,sp\n")
                    target.write("validating data\n")
                    for i in range(len(val_x_images)):
                        target.write("%s_LR,%f,%f,%f,%f,%f,%f\n" % (val_filenames[i], avg_auc_roc[i], avg_auc_pr[i],
                                                                    avg_dice[i], avg_acc[i], avg_sn[i], avg_sp[i]))
                        plt.imsave(arr=np.squeeze(val_x_images[i]),
                                   fname="%s/%04d/val/%s_x.png" % (args.exp_dir, epoch, val_filenames[i]), cmap="gray")
                        plt.imsave(arr=np.squeeze(val_y_images[i]),
                                   fname="%s/%04d/val/%s_y.png" % (args.exp_dir, epoch, val_filenames[i]), cmap="gray")
                        plt.imsave(arr=np.squeeze(pred_images[i]),
                                   fname="%s/%04d/val/%s_pred.png" % (args.exp_dir, epoch, val_filenames[i]), cmap="gray")
                        plt.imsave(arr=largest_regions[i],
                                   fname="%s/%04d/val/%s_largest_region.png" % (args.exp_dir, epoch, val_filenames[i]),
                                   cmap="gray")
                    print("\nAverage validation accuracy for epoch # %04d" % (epoch))
                    print("Average per class validation accuracies for epoch # %04d:" % (epoch))
                    print("Validation auc_roc = ", np.mean(avg_auc_roc))
                    print("Validation auc_pr = ", np.mean(avg_auc_pr))
                    print("Validation dice = ", np.mean(avg_dice))
                    print("Validation acc = ", np.mean(avg_acc))
                    print("Validation sn = ", np.mean(avg_sn))
                    print("Validation sp = ", np.mean(avg_sp))
                    target.write(
                        "x,%f,%f,%f,%f,%f,%f\n" % (np.mean(avg_auc_roc), np.mean(avg_auc_pr), np.mean(avg_dice),
                                                   np.mean(avg_acc), np.mean(avg_sn), np.mean(avg_sp)))
                    target.close()

                    # saving model
                    print("[x] saving model to %s" % "weights.h5")
                    global_evaluation = np.mean(avg_dice)
                    global_evaluation_epoch = epoch
                    model.save_weights(filepath=os.path.join(args.exp_dir, "weights.h5"))

                    # predict an test data set
                    print("[x] predict an test data set")
                    avg_auc_roc = []
                    avg_auc_pr = []
                    avg_dice = []
                    avg_acc = []
                    avg_sn = []
                    avg_sp = []
                    pred_images = []
                    largest_regions = []

                    for img_index in tqdm(range(test_x_images.shape[0])):
                        input_image = test_x_images[img_index]
                        output_image = test_y_images[img_index]

                        input_image = np.expand_dims(input_image, axis=0)
                        input_image = np.array(input_image) / 255.

                        pred_image = model.predict(input_image)
                        binary_vessels = helper_functions.threshold_by_otsu(np.squeeze(pred_image), flatten=False)

                        # evaluate on largest region
                        largest_region = helpers.generate_largest_region(binary_vessels)
                        _, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity \
                            = evaluator.metric_single_img(largest_region, np.squeeze(output_image))

                        avg_auc_roc.append(auc_roc)
                        avg_auc_pr.append(auc_pr)
                        avg_dice.append(dice_coeff)
                        avg_acc.append(acc)
                        avg_sn.append(sensitivity)
                        avg_sp.append(specificity)
                        pred_images.append(pred_image)
                        largest_regions.append(largest_region)

                    if not os.path.isdir("%s/%04d/test" % (args.exp_dir, epoch)):
                        os.makedirs("%s/%04d/test" % (args.exp_dir, epoch))
                    target = open("%s/%04d/test_score.csv" % (args.exp_dir, epoch), "w")
                    target.write("filename,auc_roc,auc_pr,dice,acc,sn,sp\n")
                    target.write("predict on test data\n")
                    for i in range(len(test_x_images)):
                        target.write(
                            "%s_LR,%f,%f,%f,%f,%f,%f\n" % (test_filenames[i], avg_auc_roc[i], avg_auc_pr[i],
                                                           avg_dice[i], avg_acc[i], avg_sn[i], avg_sp[i]))
                        plt.imsave(arr=np.squeeze(test_x_images[i]),
                                   fname="%s/%04d/test/%s_x.png" % (args.exp_dir, epoch, test_filenames[i]),
                                   cmap="gray")
                        plt.imsave(arr=np.squeeze(test_y_images[i]),
                                   fname="%s/%04d/test/%s_y.png" % (args.exp_dir, epoch, test_filenames[i]),
                                   cmap="gray")
                        plt.imsave(arr=np.squeeze(pred_images[i]),
                                   fname="%s/%04d/test/%s_pred.png" % (args.exp_dir, epoch, test_filenames[i]),
                                   cmap="gray")
                        plt.imsave(arr=largest_regions[i],
                                   fname="%s/%04d/test/%s_largest_region.png" % (
                                       args.exp_dir, epoch, test_filenames[i]),
                                   cmap="gray")
                    print("\nAverage test accuracy for epoch # %04d" % (epoch))
                    print("Average per class validation accuracies for epoch # %04d:" % (epoch))
                    print("Validation auc_roc = ", np.mean(avg_auc_roc))
                    print("Validation auc_pr = ", np.mean(avg_auc_pr))
                    print("Validation dice = ", np.mean(avg_dice))
                    print("Validation acc = ", np.mean(avg_acc))
                    print("Validation sn = ", np.mean(avg_sn))
                    print("Validation sp = ", np.mean(avg_sp))
                    target.write("x,%f,%f,%f,%f,%f,%f\n" % (
                        np.mean(avg_auc_roc), np.mean(avg_auc_pr), np.mean(avg_dice),
                        np.mean(avg_acc), np.mean(avg_sn), np.mean(avg_sp)))
                    target.close()
            print("global_evaluation = {}".format(global_evaluation))
            print("global_epoch = {}".format(global_evaluation_epoch))
    else:
        # load weights and inference images
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet('My Worksheet')


        model.load_weights(os.path.join(args.exp_name, "weights.h5"))
        patients = get_patients_in_dir(args.test_dir + "/training")
        inference_dir = os.path.join(args.test_dir, "inference")
        if not os.path.isdir(inference_dir):
            os.makedirs(inference_dir)

        dices = []
        for i in tqdm(range(len(patients))):
            image_x_path = args.test_dir + "/training/{}.png".format(patients[i])
            image_y_path = args.test_dir + "/label/{}.png".format(patients[i].replace("_x","_y"))

            image_y = cv2.imread(image_y_path, cv2.IMREAD_GRAYSCALE)
            image_y = cv2.resize(image_y, (512, 512)) / 255.

            image_x = cv2.imread(image_x_path)
            image_x = cv2.resize(image_x, (512, 512)) / 255.
            worksheet.write(i, 0, label=patients[i])
            startTime = time.perf_counter()
            image_pred = model.predict(np.expand_dims(image_x, 0))
            endTime = time.perf_counter()
            worksheet.write(i, 1, endTime - startTime)
            image_bin = helper_functions.threshold_by_otsu(np.squeeze(image_pred), flatten=False)
            image_bin_lr = helper_functions.generate_largest_region(np.squeeze(image_bin))

            dice = helper_functions.compute_dice(image_bin_lr, image_y)
            dices.append(dice)

            plt.imsave(fname=os.path.join(inference_dir, patients[i] + "_x.png"), arr=np.squeeze(image_x))
            plt.imsave(fname=os.path.join(inference_dir, patients[i] + "_pred.png"), arr=np.squeeze(image_pred), cmap="gray")
            plt.imsave(fname=os.path.join(inference_dir, patients[i] + "_pred_bin.png"), arr=np.squeeze(image_bin), cmap="gray")
            plt.imsave(fname=os.path.join(inference_dir, patients[i] + "_pred_bin_lr.png"), arr=np.squeeze(image_bin_lr), cmap="gray")
            plt.imsave(fname=os.path.join(inference_dir, patients[i] + "_y.png"), arr=np.squeeze(image_y), cmap="gray")
            # 写入excel
            # 参数对应 行, 列, 值


            # 保存
        workbook.save('Excel_test.xls')
        print("[x] performance on test set, dice = {}".format(np.mean(dices)))
