import glob
import os
import shutil
import face_recognition
import cv2
import csv
from PIL import Image
import scipy.misc
import numpy as np
import random
import dlib

detector = dlib.get_frontal_face_detector() #Face detector

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions

src='F:\Chaos\EmotionRecognition\Application\dataset'
dest='F:\Chaos\EmotionRecognition\Application\combineData'

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("F:\Chaos\EmotionRecognition\Download_CK+\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set t

def copyDetectedImage():            #For moving faces from one directory to another

    for subdir in glob.glob(os.path.join(src,"*")):
        for files in glob.glob(os.path.join(subdir,"*")):
            frame = face_recognition.load_image_file(files)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_locations=face_recognition.face_locations(frame)
            for face in face_locations:
                top,right,bottom,left = face
                face_image = frame[top:bottom,left:right]
                try:
                    out = cv2.resize(face_image, (48, 48)) #Resize face so all images have same size
                    cv2.imwrite(dest+files.replace(src,''), out) #Write image
                except:
                   pass



def remove_Mismatch():
    main_Directory='F:\Chaos\EmotionRecognition\Download_CK+\extended-cohn-kanade-images\cohn-kanade-images\*'
    sub_Dir = 'F:\Chaos\EmotionRecognition\Download_CK+'


    for x in glob.glob(main_Directory):
        for subdir in glob.glob(x+"\*"):
                part = "%s" %subdir[-9:]
                temp_Dir=sub_Dir+'\Emotion_labels\Emotion'+part
                if not os.path.isdir(temp_Dir):
                    print ("{0} is Removed".format(subdir))
                    shutil.rmtree(subdir)

def remove_Images():
    main_Directory='F:\Chaos\EmotionRecognition\Download_CK+\extended-cohn-kanade-images\cohn-kanade-images\*'
    for x in glob.glob(main_Directory):
        for subdir in glob.glob(x+"\*"):
            grab_L=glob.glob(subdir+"\*.png")
            temp_List=[grab_L[0],grab_L[-1]]
            for x in grab_L:
                if x not in temp_List:
                    os.remove(x)

def copyImages():

    vendors = ("anger_","disgust_","fear_","happy_","sadness_","surprise_","neutral_")
    counts =[0,0,0,0,0,0,0]


    width,height=48,48
    image = np.zeros((height,width),np.uint8)

    with open('F:\Chaos\EmotionRecognition\Download_CK+\\fer2013.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            index = int(row['emotion'])
            #{emotion,pixels,Usage}
            pixels = list(map(int,row['pixels'].split()))
            pixels_array = np.asarray(pixels)

            image = pixels_array.reshape(width, height)
            #print image.shape
            stacked_image = np.dstack((image,) * 3)

            #print (os.path.join(dest,emotions[index],vendors[index]+str(counts[index])+'.jpg'))
            scipy.misc.imsave(os.path.join(dest,emotions[index],vendors[index]+str(counts[index])+'.jpg'), stacked_image)
            counts[index]+=1


def sp_noise(image,prob):

    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def addGaussianNoise(subdir):

    for files in glob.glob(os.path.join(subdir,'*.jpg')):

        image = cv2.imread(files)

        sigma = 0.005

        for i in range(6):
            noisy_image=sp_noise(image,sigma)
            sigma+=0.005
            try:
                cv2.imwrite(files+'_'+str(i)+'.jpg', noisy_image) #Write image
            except:
               pass


def checkDetect(subdir):

    print (subdir)

    correct=0
    incorrect=0
    for image in glob.glob(os.path.join(subdir,'*.jpg')):

        frame=cv2.imread(image)
        detections =detector(frame,1)
        if(len(detections)<1):
            incorrect+=1
            os.remove(image)
        else:
            correct+=1

    print('Correct {0} , Incorrect {1} , Percentage {2}'.format(correct,incorrect,(100*correct)/(correct+incorrect)))


def drawLandMarks(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    clahe_image  = clahe.apply(gray)
    detections = detector(clahe_image,1)

    for k,d in enumerate(detections):
        shape = predictor(clahe_image,d)
        for i in range(1,68):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame

    cv2.imshow('image',image)

image1=cv2.imread('temp1.jpg')
drawLandMarks(image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#image2=cv2.imread('anger_326.jpg')
#drawLandMarks(image2)
#checkDetect('F:\Chaos\EmotionRecognition\Application\combineData\disgust')
#addGaussianNoise('F:\Chaos\EmotionRecognition\Application\combineData\disgust')
#remove_Images()
#copyImages()
#copyImages()
