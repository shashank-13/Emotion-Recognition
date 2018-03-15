#Import required modules
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
import pickle
import sys

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions
new_emotions = ["neutral","anger", "happy", "sadness"] #Define emotions

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("F:\Chaos\EmotionRecognition\Download_CK+\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
dir2='F:\Chaos\EmotionRecognition\Application\model\images'
dir1='F:\Chaos\EmotionRecognition\Application\combineData'
dir3='F:\Chaos\EmotionRecognition\Application\dataset'
dir4='F:\Chaos\EmotionRecognition\Application\\trainData'
dir5='F:\Chaos\EmotionRecognition\Application\sorted_set'

def shuffle_files(emotion):

    files = glob.glob(dir5+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction


def getSlicedData(emotion):

    files = glob.glob(dir1+"\\"+emotion+"\*")
    #random.shuffle(files)
    training = files[:int(len(files))] #full data
    return training


def getLandmarks(image):                      #Function to detect landmarks in image

    landmarks_Vector=[]
    xList=[]
    yList=[]

    detections =detector(image,1)  #Detect

    for k,d in enumerate(detections):

        shape = predictor(image,d)
        for i in range(1,68):
            xList.append(float(shape.part(i).x))
            yList.append(float(shape.part(i).y))

        xmean = np.mean(xList)  #Calculate mean for the both co-ordinates
        ymean = np.mean(yList)
        xcentral = [(x - xmean) for x in xList] #Taking distance from the  center
        ycentral = [(y-ymean) for y in yList]

        for x,y,w,z in zip(xcentral,ycentral,xList,yList):

            landmarks_Vector.append(w)
            landmarks_Vector.append(z)
            meannp=np.asarray((ymean,xmean))
            coorp = np.asarray((z,w))
            dist = np.linalg.norm(coorp-meannp)           #Square root representation

            landmarks_Vector.append(dist)
            landmarks_Vector.append((math.atan2(y, x)*360)/(2*math.pi))     #Angle method



    return landmarks_Vector


#for generating the generateModel
def generateModel():

    training_data = []
    training_labels = []

    for emotion in emotions:


        files=getSlicedData(emotion)
        for item in files:
            image = cv2.imread(item)
            gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_Vector=getLandmarks(clahe_image)
            if landmarks_Vector:
                training_data.append(landmarks_Vector)
                training_labels.append(emotions.index(emotion))
            print ('Length {0} and Size {1}'.format(len(training_data),sys.getsizeof(training_data)))
            print ('Length {0} and Size {1}'.format(len(training_labels),sys.getsizeof(training_labels)))

    clf = SVC(kernel='linear', probability=True, tol=1e-3)
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)

    clf.fit(training_data,training_labels)
    filename='F:\Chaos\EmotionRecognition\Application\model\\raw\\svmModel_full.sav'
    pickle.dump(clf, open(filename, 'wb'))


#for predicting the svmModel

def predict_Particular():

    filename='F:\Chaos\EmotionRecognition\Application\model\\raw\\svmModel_full.sav'
    loaded_model = pickle.load(open(filename, 'rb'))


    for emotion in new_emotions:

        prediction_data = []
        prediction_labels = []

        for file in glob.glob(dir2+"\\"+emotion+"\*"):

            image = cv2.imread(file)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_Vector=getLandmarks(clahe_image)
            if landmarks_Vector:
                prediction_data.append(landmarks_Vector)
                prediction_labels.append(emotions.index(emotion))
                print ('Expected {0} ,Got {1}'.format(emotion,emotions[loaded_model.predict([landmarks_Vector])]))


        prediction_data = np.array(prediction_data)
        prediction_labels = np.array(prediction_labels)
        pred_score = loaded_model.score(prediction_data,prediction_labels)
        print ('{0} Prediction Score {1}'.format(emotion,pred_score))

def get_Data():                              # Function to return all data

    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:

        training, prediction = shuffle_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_Vector=getLandmarks(clahe_image)
            if landmarks_Vector:
                training_data.append(landmarks_Vector)
                training_labels.append(emotions.index(emotion))

        for item in prediction:

            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_Vector=getLandmarks(clahe_image)
            if landmarks_Vector:
                prediction_data.append(landmarks_Vector)
                prediction_labels.append(emotions.index(emotion))


    return training_data, training_labels, prediction_data, prediction_labels

def applySVM():

    clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

    accuracy_List=[]

    for i in range(0,10):

        training_data, training_labels, prediction_data, prediction_labels = get_Data()
        print ('Got Data')
        print ('{0} {1} {2} {3}'.format(len(training_data),len(training_labels),len(prediction_data),len(prediction_labels)))

        npar_train = np.array(training_data)
        npar_trainlabels = np.array(training_labels)

        clf.fit(npar_train,npar_trainlabels)
        npar_pred=np.array(prediction_data)

        pred_score = clf.score(npar_pred,prediction_labels)
        print ('Score {0}'.format(pred_score))
        accuracy_List.append(pred_score)

    print ('Mean Accuracy {0}'.format(np.mean(accuracy_List)))



applySVM()
#generateModel()  #for generating the model
#predict_Particular() #for predicting the model
