import numpy as np
import pandas as pd
import cv2


def read_file_npy():
    data = np.load('/storages/data/DATA/Action_Recognition/data/Kinetics/kinetics-skeleton/val_data.npy')
    print(data)


def read_file_pickle():
    path_data_train = "/storages/data/DATA/Action_Recognition/DataTraining"
    file_pickle = path_data_train + '/Coffee_room_new-set(labelXscrw).pkl'

    object = pd.read_pickle(file_pickle)
    print(object)


def read_video():
    # print("Before URL")
    cap = cv2.VideoCapture('rtsp://admin:admin@192.168.111.111/1')
    # print("After URL")

    while True:

        # print('About to start the Read command')
        ret, frame = cap.read()
        # print('About to show frame of Video.')
        cv2.imshow("Capturing", frame)
        # print('Running..')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_file_npy()
    read_file_pickle()
    # read_video()