# Example input :-
# python dataGenerator.py --dataset 'C:\Users\Akshay Shrivastava\OneDrive\Desktop\BTP\datasets'  --folder 'Akshay'
# Note-Input should in string format

import cv2
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the dataset")
ap.add_argument("-e", "--folder", required=True, help="Name of the folder")
args = vars(ap.parse_args())

path = args["dataset"]
folder = args["folder"]

# 0 means we want to capture from device camera
cap = cv2.VideoCapture(0)

# Example for input
# path="C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets"
# folder="Akshay"

image_count = 0
if not os.path.isdir(path+"/"+folder):
    print('[INFO] Folder is not present....')
    print('[INFO] Creating folder....')
    os.mkdir(path+"/"+folder)
    print('[INFO] Folder created....')
else:
    print('[INFO] Folder found....')
    image_count = len(next(os.walk(path+"/"+folder))[2])
end_count = image_count+30

print('[INFO] Opening camera.... with current image count : {}'.format(image_count))
while True:
    _, frame = cap.read()
    cv2.imshow('output', frame)
    cv2.imwrite(path+"/"+folder+"/" +
                "normal_image_{}.png".format(image_count), frame)
    image_count += 1

    if image_count > end_count:
        print('[INFO] Success..')
        break
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
