# Usage
# To extract standardized face embeddings from images in a pickle file
# python .\faceEmbeddings.py -d 'C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train' -e 'C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/embeddings/embeddings_mtcnn_stdn_5Classes.pickle'

from unicodedata import name
import pickle
import cv2
import os
import argparse
import ntpath
import face_recognition
from sklearn.preprocessing import StandardScaler


def extract_name(path):
    tail = ntpath.basename(path)
    arr = tail.split('-')
    return arr[0]


ap = argparse.ArgumentParser()
scalar = StandardScaler()

ap.add_argument("-d", "--dataset", required=True, help="path to the dataset")
ap.add_argument("-e", "--embedding", required=True,
                help="path to the embeddings")
args = vars(ap.parse_args())

path = args["dataset"]
embedding_path = args['embedding']

embeddings = []
names = []

for folder in os.listdir(path):
    name = extract_name(path+'/'+folder)
    for image in os.listdir(path+'/'+folder):
        frame = cv2.imread(path+'/'+folder+'/'+image)
        try:
            vector = face_recognition.face_encodings(frame)[0]
            vector = vector.reshape(1, -1)
        except:
            print('*********except block*********')
            continue
        names.append(name)
        scalar.fit_transform(vector)
        embeddings.append(vector.flatten())

data = {"embeddings": embeddings, "names": names}
f = open(embedding_path, "wb")
f.write(pickle.dumps(data))
f.close()
print(len(names), len(embeddings))
