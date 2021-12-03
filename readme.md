# Action recognition model

# Install environment
`pip install -r requirements.txt`

# Dataset

## Data format
+ Home (Home_01.zip and Home_02.zip) and Coffee room (Coffee_room_01.zip and Coffee_room_02.zip)
+ For each video, the annotation is given video (i).txt in Annotation_files. Each annotation file contains each file contains:

  + The frame number of the beginning of the fall
  + The frame number of the end of the fall
  + The height, the width and the coordinates of the center of the bounding box for each frame
  

## MHI(Motion History Image) process:
  + Fall Detection Dataset (FDD): https://medium0.com/diving-in-deep/fall-detection-with-pytorch-b4f19be71e80
  + https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html
  + https://drive.google.com/drive/folders/19KTp4-0Q4RL7MRsd0Gqxbt-1oKA-pbeY
  + https://github.com/dzungvpham/fall-detection-two-stream-cnn 
  https://github.com/dzungvpham/fall-detection-two-stream-cnn/issues/2


## Le2i data
+ Le2i - Laboratoire Electronique, Informatique et Image](Le2i - Laboratoire Electronique, Informatique et Image) [1]. A dataset contains four scenes: 
  + Home(60 videos), Coffee room(70 videos), Office(64 videos), Lecture room(27 videos)
+ Only Home and Coffee room subset have 'Annotation_files', which describe the frame number of the beginning and end of the fall. 
  + FORMAT: 320x240 25FPS. 
  + High quality. Single person. 
  + A large-volume dataset.


## Data reference
+ https://drive.google.com/drive/folders/19KTp4-0Q4RL7MRsd0Gqxbt-1oKA-pbeY