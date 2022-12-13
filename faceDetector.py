# Usage :-
# Detect Faces from normal images
# python .\faceDetector.py --dataset 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\datasets\train'  --folder 'Akshay'

import mtcnn
import cv2
import os
import argparse
import ntpath
import shutil


def image_number(path):
    tail = ntpath.basename(path)
    i = ''
    arr = tail.split('_')
    for ch in arr[2]:
        if ch == '.':
            break
        i += ch
    return int(i)


detector = mtcnn.MTCNN()

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help="path to the dataset")
ap.add_argument("-e", "--folder", required=True, help="Name of the folder")
ap.add_argument("-c", "--confidence",
                help="minimum confidence for face detection")
args = vars(ap.parse_args())

path = args["dataset"]
folder = args['folder']
try:
    confidence = args["confidence"]
except:
    # default value
    confidence = 0.5

if folder in os.listdir(path):
    if not os.path.isdir(path+"/"+folder+'-Face'):
        print('[INFO] Folder is not present....')
        print('[INFO] Creating folder....')
        os.mkdir(path+"/"+folder+'-Face')
        print('[INFO] Folder created....')
    else:
        print('[INFO] Folder found....')
    for image in os.listdir(path+'/'+folder):
        num = image_number(path+'/'+folder+'/'+image)
        frame = cv2.imread(path+'/'+folder+'/'+image)
        detections = detector.detect_faces(frame)
        for det in detections:
            try:
                # ROI
                x, y, width, height = det['box']
                face = frame[y:y+height, x:x+width]
                print('[INFO] Writing image in the folder')
                cv2.imwrite(path+"/"+folder+"-Face"+"/" +
                            "face_image_{}.png".format(num), face)
                print('[INFO] Successfully Written')
            except:
                print('*********except block*********')
                continue
print('[INFO] Success.')
print('[INFO] Removing original Dataset folder...')
shutil.rmtree(path+'/'+folder, ignore_errors=True)
print('[INFO] Removed folder')
