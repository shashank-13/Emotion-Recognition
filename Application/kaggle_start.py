import csv
from PIL import Image
import cv2
import numpy as np
import scipy.misc

new_emotions = ["disgust","happy", "neutral", "surprise"] #Define emotions

def readImages():

    width,height=48,48
    image = np.zeros((height,width),np.uint8)
    with open('F:\Chaos\EmotionRecognition\Download_CK+\\fer2013.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print (row['emotion'])
            #{emotion,pixels,Usage}
            '''pixels = list(map(int,row['pixels'].split()))
            pixels_array = np.asarray(pixels)

            image = pixels_array.reshape(width, height)
            #print image.shape
            stacked_image = np.dstack((image,) * 3)

            scipy.misc.imsave('test.jpg', stacked_image)'''
            break


readImages()
