"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.

"""
import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import shutil

# class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
#                'Stand up', 'Sit down', 'Fall Down']
from Data.config import path_data_train

main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y',
              'RHip_x', 'RHip_y']
main_idx_parts = [1, 2, 7, 8, -1]  # 1.5 ; L_shoulder, R_shoulder, L_hip, R_hip and point between l, r shoulder

# csv_pose_file = '../Data/Coffee_room_new-pose+score.csv'
# csv_pose_file = path_data_train + '/Home_new-pose+score.csv'

# save_path = path_data_train + '/Coffee_room_new-set(labelXscrw).pkl'

# Params.
smooth_labels_step = 8
n_frames = 30
skip_frame = 1


def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)  # min of x,y in 13 point x,y(13 point is one pose)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels


def write_data_pickle(annot, cols, save_path_train, save_path_test):
    feature_set = np.empty((0, n_frames, 14, 3))
    labels_set = np.empty((0, len(cols)))
    vid_list = annot['video'].unique()
    for vid in vid_list:
        print('Process on - video name: {}'.format(vid))
        data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video')

        # Label Smoothing.
        """
        data[cols] : get col in data frame with name col = cols.
        """
        esp = 0.1
        data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
        data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

        # Separate continuous frames. to divide frame was break to another frame set.
        frames = data['frame'].values
        frames_set = []
        fs = [0]
        for i in range(1, len(frames)):
            if frames[i] < frames[i-1] + 10:
                fs.append(i)
            else:
                frames_set.append(fs)
                fs = [i]
        frames_set.append(fs)

        for fs in frames_set:
            xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 13, 3)
            # Scale pose normalize.
            """
            Normalize point x,y in range [-1:1] in each pose(take min/max in 13 point of each pose)
            """
            xys[:, :, :2] = scale_pose(xys[:, :, :2])
            # Add center point.(add point between right shoulder and left shoulder(middle ~ neck))
            xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1)

            # Weighting main parts score.(shoulder, hip and center point)
            scr = xys[:, :, -1].copy()
            scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
            # Mean score.
            scr = scr.mean(1)

            # Targets.
            lb = data.iloc[fs, -len(cols):].values
            # Apply points score mean to all labels.
            lb = lb * scr[:, None]

            for i in range(xys.shape[0] - n_frames):
                feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
                labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(feature_set, labels_set, test_size=0.2, random_state=0)
    with open(save_path_train, 'wb') as f:
        pickle.dump((X_train, y_train), f)
    with open(save_path_test, 'wb') as f:
        pickle.dump((X_test, y_test), f)


def counter_label_training():
    # list_pickle_file_step3 = glob(path_data_train + "/Data_Step3/*")
    # for file_pickle in list_pickle_file_step3:
        file_pickle = "/storages/data/DATA/Action_Recognition/DataTraining/Coffee_room_01-pose+score.pkl"
        print("Process on file {}".format(file_pickle))
        feature_set, labels_set = np.load(file_pickle, allow_pickle=True)
        if labels_set.shape[1] == 7:
            y_label = np.argmax(labels_set, axis=1)
            # plt.hist(y_label)
            # plt.show()
            # a = 0

            c = Counter(y_label)

            plt.bar(c.keys(), c.values())
            plt.show()


def combine_data_step2():
    list_csv_pose_file_step2 = glob(path_data_train + "/Data_Step2_2_class/*")
    save_path = path_data_train + "/Data_Step2_2_class/total_step2.csv"
    list_csv_pose_file_step2 = sorted(list_csv_pose_file_step2)
    total_annot_step2 = None
    for csv_pose_file in list_csv_pose_file_step2:
        annot = pd.read_csv(csv_pose_file)
        if total_annot_step2 is None:
            total_annot_step2 = annot
        else:
            total_annot_step2 = pd.concat([total_annot_step2, annot]).reset_index(drop=True)
    total_annot_step2.to_csv(save_path, mode='w', index=False)
    # 57567 Frame, 105 video


def prepare_data_step3():
    csv_pose_file = path_data_train + "/Data_Step2_2_class/total_step2.csv"

    save_path_train = path_data_train + "/Data_Step3_2_class/"
    if os.path.exists(save_path_train):
        shutil.rmtree(save_path_train)
    os.makedirs(save_path_train)
    save_path_train = save_path_train + "train.pkl"

    save_path_test = path_data_train + "/Data_Step3_2_class/"
    if os.path.exists(save_path_test):
        shutil.rmtree(save_path_test)
    os.makedirs(save_path_test)
    save_path_test = save_path_test + "test.pkl"

    annot = pd.read_csv(csv_pose_file)
    # 57567 Frame, 105 video

    # Remove NaN.
    idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
    idx = np.where(idx)[0]
    annot = annot.drop(idx)
    # One-Hot Labels.
    # Data in Home have 5 labels missing 2 labels.
    label_onehot = pd.get_dummies(annot['label'])
    annot = annot.drop('label', axis=1).join(label_onehot)
    cols = label_onehot.columns.values

    write_data_pickle(annot, cols, save_path_train, save_path_test)


if __name__ == '__main__':
    # combine_data_step2()
    prepare_data_step3()

    # counter_label_training()