# Usage :-
# Used to find the accuracy of the model, on validation dataset
# python .\test.py -v 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\datasets\val' -l 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\recognizers\le.pickle' -r 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\recognizers\recognizer_GBC.pickle'

import cv2
import face_recognition
import numpy as np
import argparse
import pickle
import os
import ntpath


def extract_name(path):
    tail = ntpath.basename(path)
    arr = tail.split('_')
    return arr[1]


ap = argparse.ArgumentParser()

ap.add_argument("-v", "--validation", required=True,
                help="Path for validation folder")
ap.add_argument("-l", "--label_encoded", required=True,
                help="Path for label encoded file")
ap.add_argument("-r", "--recognizer", required=True,
                help="path for the recognizer")
args = vars(ap.parse_args())

test_path = args["validation"]
le_path = args["label_encoded"]
recognizer_path = args["recognizer"]

le = pickle.loads(open(le_path, "rb").read())
with open(recognizer_path, "rb") as pickle_file:
    recognizer = pickle.load(pickle_file)

arr = []
for folder in os.listdir(test_path):
    for image in os.listdir(test_path+'/'+folder):
        frame = cv2.imread(test_path+'/'+folder+'/'+image)
        try:
            vector = face_recognition.face_encodings(frame)[0]
            vector = vector.reshape(1, -1)
        except:
            continue
        preds = recognizer.predict_proba(vector)[0]
        j = np.argmax(preds)
        arr.append(preds[j])

print('[INFO] Accuracy for'+' {}'.format(extract_name(recognizer_path)
      [:3])+' is : {}'.format((sum(arr)/len(arr))))
