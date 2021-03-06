"""
This script to extract skeleton joints position and score.

- This 'annot_folder' is a action class and bounding box for each frames that came with dataset.
    Should be in format of [frame_idx, action_cls, xmin, ymin, xmax, ymax]
        Use for crop a person to use in pose estimation model.
- If have no annotation file you can leave annot_folder = '' for use Detector model to get the
    bounding box.
"""

import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
from glob import glob
import torchvision.transforms as transforms

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast


path_data_train = "/storages/data/DATA/Action_Recognition/DataTraining"

# save_path = path_data_train + '/Home_new-pose+score.csv'

# annot_file = '../../Data/Home_new.csv'  # from create_dataset_1.py
# video_folder = '../Data/falldata/Home/Videos'

# DETECTION MODEL.
detector = TinyYOLOv3_onecls()

# POSE MODEL.
backbone = 'resnet50'
inp_h = 320
inp_w = 256
pose_estimator = SPPE_FastPose(backbone, inp_h, inp_w)

# with score.
columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


def extract_skeleton_joints_position_and_score(file_label_step1, video_folder, annot_folder, save_path, list_video_name,name_file):
    annot_step1 = pd.read_csv(file_label_step1)
    vid_list = annot_step1['video'].unique()
    for vid in vid_list:
        print(list_video_name)
        if vid in list_video_name:
            vid_rename = vid.split(".")[0] + "_" + name_file + ".avi"
        else:
            vid_rename = vid
        list_video_name.append(vid_rename)
        print(f'Process on: {vid}')
        df = pd.DataFrame(columns=columns)
        cur_row = 0

        # Pose Labels.
        frames_label = annot_step1[annot_step1['video'] == vid].reset_index(drop=True)

        cap = cv2.VideoCapture(os.path.join(video_folder, vid))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Bounding Boxs Labels.
        # annot_file = os.path.join(annot_folder, vid.split('.')[0], '.txt')
        annot_file = os.path.join(annot_folder, vid.split('.')[0]) + '.txt'
        annot = None
        if os.path.exists(annot_file):
            annot = pd.read_csv(annot_file, header=None,
                                      names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
            annot = annot.dropna().reset_index(drop=True)

            if frames_count != len(annot):
                annot = None
                # continue
            # assert frames_count == len(annot), 'frame count not equal! {} and {}'.format(frames_count, len(annot))

        fps_time = 0
        i = 1
        while True:
            ret, frame = cap.read()
            name_video_folder = video_folder.split("/")[-2]
            print("Video folder {} - Video name  {} - frame:  {}".format(name_video_folder, vid, i))
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cls_idx = int(frames_label[frames_label['frame'] == i]['label'])

                if annot is not None:
                    bb = np.array(annot.iloc[i-1, 2:].astype(int))
                else:
                    try:
                        bb = detector.detect(frame)[0, :4].numpy().astype(int)
                    except Exception as e:
                        print(e)
                        continue
                bb[:2] = np.maximum(0, bb[:2] - 5)
                bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]

                result = []
                if bb.any() != 0:
                    result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                                    torch.tensor([[1.0]]))

                if len(result) > 0:
                    pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                         frame_size[0], frame_size[1])
                    pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)

                    #idx = result[0]['kp_score'] <= 0.05
                    #pt_norm[idx.squeeze()] = np.nan
                    row = [vid_rename, i, *pt_norm.flatten().tolist(), cls_idx]
                    scr = result[0]['kp_score'].mean()
                else:
                    row = [vid_rename, i, *[np.nan] * (13 * 3), cls_idx]
                    scr = 0.0

                df.loc[cur_row] = row
                cur_row += 1

                # VISUALIZE.
                # frame = vis_frame_fast(frame, result)
                # frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                # frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, cls_idx, scr),
                #                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # frame = frame[:, :, ::-1]
                # fps_time = time.time()
                i += 1

                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        if os.path.exists(save_path):
            df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            df.to_csv(save_path, mode='w', index=False)


if __name__ == '__main__':
    list_file_data_step1 = glob(path_data_train + "/Data_Step1_2_class/*")
    list_file_data_step1 = sorted(list_file_data_step1)
    list_video_name = []
    for file_label_step1 in list_file_data_step1:
        name_file = file_label_step1.split("/")[-1][:-4]
        video_folder = path_data_train + "/Videos/" + name_file + "/Videos"
        annot_folder = path_data_train + "/Videos/" + name_file + "/Annotation_files"
        save_path = path_data_train + "/Data_Step2_2_class/" + name_file + "-pose+score.csv"
        extract_skeleton_joints_position_and_score(file_label_step1, video_folder, annot_folder, save_path, list_video_name, name_file)

