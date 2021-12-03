# Step create data
## Step 1
+ Run create_dataset_1.py -> create_csv
+ Run all video in folder Videos and create a file annot_file.csv with col ['video', 'frame', 'label'].
+ Each frame label with `0`.
+ Show result: create_dataset_1.py -> show_annot_file, class name modify: `cls_name = class_names[int(annot.iloc[i, -1]) - 1]`

+ File save: "/storages/data/DATA/Action_Recognition/DataTraining/Home_new.csv"
## Step 2
+ Run create_dataset_2.py
+ Using file annot_folder with contain file annotation with format: `[frame_idx, action_cls, xmin, ymin, xmax, ymax]`
  + If video no file annotation using yolo detection to get the bounding box.
+ Skip some file error: assert frames_count == len(annot)

## Step 3

