import glob
import cv2
import numpy as np
import random
import sys

emotions = ["anger", "disgust", "fear", "happy","neutral", "sadness", "surprise"] #Define emotions
new_emotions = ["neutral","anger", "happy", "sadness"] #Define emotions

fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
#fishface.read('F:\Chaos\EmotionRecognition\Application\model\\raw\model.xml')
face_classifier = cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_default.xml')
face_classifier2 = cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt2.xml')
face_classifier3=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt.xml')
face_classifier4=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt_tree.xml')
dir2='F:\Chaos\EmotionRecognition\Application\model\images'
dir1='F:\Chaos\EmotionRecognition\Application\combineData'
test='F:\Chaos\EmotionRecognition\Application\dataset'

def shuffle_files(emotion):

    files = glob.glob(dir2+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction


def get_Data():

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
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def classify_Data():

    training_data, training_labels, prediction_data, prediction_labels = get_Data()
    fishface.train(training_data, np.asarray(training_labels))

    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        #print ('expected {0} got {1}'.format(emotions[prediction_labels[cnt]],emotions[pred]))
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    #print ('{0} correct and {1} incorrect'.format(correct,incorrect))
    return ((100*correct)/(correct + incorrect))


def getSlicedData(emotion):

    files = glob.glob(dir1+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.05)] #get first 50% of file list
    return training

def predictDataset():

    prediction_data = []
    prediction_labels = []

    #fishface.read('F:\Chaos\EmotionRecognition\Application\model\\raw\unpredictable_Model.xml')

    for emotion in emotions:


        files = glob.glob(dir1+"\\"+emotion+"\*")
        for item in files:
            image = cv2.imread(item)
            gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        #print ('expected {0} got {1}'.format(emotions[prediction_labels[cnt]],emotions[pred]))
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    #print ('{0} correct and {1} incorrect'.format(correct,incorrect))
    print ('Correct {0} , Incorrect {1} , Percentage {2}'.format(correct,incorrect,(100*correct)/(correct+incorrect)))




def generateModel():

    training_data = []
    training_labels = []

    for emotion in emotions:

        files = glob.glob(test+"\\"+emotion+"\*")
        #files=getSlicedData(emotion)
        for item in files:
            image = cv2.imread(item)
            gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))
            print ('Length {0} and Size {1}'.format(len(training_data),sys.getsizeof(training_data)))
            print ('Length {0} and Size {1}'.format(len(training_labels),sys.getsizeof(training_labels)))

    fishface.train(training_data,np.asarray(training_labels))
    fishface.save('F:\Chaos\EmotionRecognition\Application\model\\raw\\babyModel.xml')



def predictResult(image):

    fishface.read('F:\Chaos\EmotionRecognition\Application\model\\raw\model.xml')
    #model = joblib.load(open(filename, 'rb'))
    #model = pickle.load(open(filename, 'rb'))
    image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pred,conf = fishface.predict(image)
    print ('Result is {0}'.format(emotions[pred]))


def liveFromCam(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        faces = face_classifier2.detectMultiScale(gray,1.3,5)
        if faces is ():
            faces = face_classifier3.detectMultiScale(gray,1.3,5)
            if faces is ():
                faces = face_classifier4.detectMultiScale(gray,1.3,5)
                if faces is ():
                    print ('No Faces Detected')

    for (x,y,w,h) in faces:

        gray = gray[y:y+h,x:x+w]
        gray=cv2.resize(gray, (350, 350))
        pred,conf = fishface.predict(gray)
        cv2.imshow('Original Image',image)
        print ('Result is {0}'.format(emotions[pred]))


def predict_Particular(emotion):


    correct =0
    incorrect =0
    result = emotions.index(emotion)
    for file in glob.glob(dir2+"\\"+emotion+"\*"):

        image = cv2.imread(file)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        pred,conf = fishface.predict(gray)
        print ('Expected {0} , Got {1}'.format(emotion,emotions[pred]))
        cv2.imshow('image',image)
        cv2.waitKey(0)
        if(pred == result):
            correct+=1
        else:
            incorrect+=1

    print ('Correct = {0} , Incorrect = {1} , Percentage = {2}'.format(correct,incorrect,(100*correct)/(correct + incorrect)))






#Now run it

#generateModel()            #for generating the model
#predict_Particular('sadness') #for predicting a directory
fishface.read('F:\Chaos\EmotionRecognition\Application\model\\raw\\babyModel.xml')
#predict_Particular('anger')
for emotion in new_emotions:
    predict_Particular(emotion)

'''image = cv2.imread('F:\Chaos\EmotionRecognition\Application\model\images\\ray_joy_1.png')
predictResult(image)  ''' #For Testing an images

'''cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    cv2.waitKey(3000)
    liveFromCam(frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()'''


'''metascore = []
for i in range(0,10):
    correct = classify_Data()
    print ("got {0},percent correct!".format(correct))
    metascore.append(correct)

print ("\n\nend score: {0}, percent correct!".format(np.mean(metascore)))'''
