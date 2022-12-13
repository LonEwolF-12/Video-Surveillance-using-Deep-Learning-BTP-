# Usage:-
# To see recognitions on live stream videos
#  python .\videoRecognize.py -l 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\recognizers\le.pickle' -r 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\recognizers\recognizer_SVM.pickle'

import pickle
import cv2
import face_recognition
from mtcnn import MTCNN
import numpy as np
import argparse

detector = MTCNN()
ap = argparse.ArgumentParser()

ap.add_argument("-l", "--label_encoded", required=True,
                help="Path for label encoded file")
ap.add_argument("-r", "--recognizer", required=True,
                help="path for the recognizer")
args = vars(ap.parse_args())

recognizer_path = args["recognizer"]
le_path = args["label_encoded"]

le = pickle.loads(open(le_path, "rb").read())
with open(recognizer_path, "rb") as pickle_file:
    recognizer = pickle.load(pickle_file)
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    detections = detector.detect_faces(frame)
    for det in detections:
        try:
            x, y, width, height = det['box']
            face = frame[y:y+height, x:x+width]
            vector = face_recognition.face_encodings(frame)[0]
            vector = vector.reshape(1, -1)
        except:
            print('********except block*********')
            continue
        preds = recognizer.predict_proba(vector)[0]
        j = np.argmax(preds)
        name = le.classes_[j]
        confidence = preds[j]
        if confidence > 0.5:
            text = "{}: {:.2f}%".format(name, confidence * 100)
            Y = y-10 if y-10 > 10 else y+10
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0))
            cv2.putText(frame, text, (x, Y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0))
    cv2.imshow('out', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
