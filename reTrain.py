# Usage :-
# The file will update the recognizer corresponding to the embedding file
# python .\reTrain.py -e 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\embeddings\embeddings_mtcnn_stdn_meetingTrain.pickle' -r 'SVM'

from ast import arg
from operator import mod
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import argparse
import pickle
from sklearn.preprocessing import LabelEncoder

ap = argparse.ArgumentParser()
le = LabelEncoder()

ap.add_argument("-e", "--embbedings", required=True,
                help="Name of the embedding file")
ap.add_argument("-r", "--recognizer", choices=[
                "SVM", "XGB", "RFR", "ABC", "GBC"], required=True,
                help="Name of the recognizer : SVM, XGBoost, Random Forest Classifier, Ada Boost, Gradient Boost")
args = vars(ap.parse_args())

embbeding_path = args["embbedings"]
model = args["recognizer"]

data = pickle.loads(open(embbeding_path, "rb").read())
labels = le.fit_transform(data["names"])

if model == 'SVM':
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
elif model == 'XGB':
    recognizer = XGBClassifier(n_estimators=900)
elif model == 'RFR':
    recognizer = RandomForestClassifier(n_estimators=500)
elif model == 'ABC':
    recognizer = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
elif model == 'GBC':
    recognizer = GradientBoostingClassifier(n_estimators=500)

recognizer.fit(data['embeddings'], labels)

recognizer_path = "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/recognizers/recognizer_"+model+".pickle"

f = open(recognizer_path, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/recognizers/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
