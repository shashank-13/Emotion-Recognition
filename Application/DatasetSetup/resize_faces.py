import cv2
import glob
import os
import dlib
import face_recognition
import sys

face_classifier = cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_default.xml')
face_classifier2 = cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt2.xml')
face_classifier3=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt.xml')
face_classifier4=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt_tree.xml')
face_classifier5=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_profileface.xml')


emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions
dir1='F:\Chaos\EmotionRecognition\Application\\testData'
dir2='F:\Chaos\EmotionRecognition\Application\model\images'
test='F:\Chaos\EmotionRecognition\Application\dataset'
new_Dir = 'F:\Chaos\EmotionRecognition\Application\FER_CB_256_COLOR'


def resizeIndividual():

    for emotion in emotions:
        for image in glob.glob(dir2+"\\"+emotion+'\*'):
            frame = face_recognition.load_image_file(image)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_locations=face_recognition.face_locations(frame)
            if(len(face_locations)==0):
                os.remove(image)
            for face in face_locations:
                top,right,bottom,left = face
                face_image = frame[top:bottom,left:right]
                try:
                    out = cv2.resize(face_image, (350, 350)) #Resize face so all images have same size
                    cv2.imwrite(image, out) #Write image
                except:
                   pass


def detect_faces(emotion,correct,incorrect):

    for image in glob.glob(new_Dir+"\\"+emotion+"\*"):

        #print (image)
        frame = cv2.imread(image)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            faces = face_classifier2.detectMultiScale(gray,1.3,5)
            if faces is ():
                faces = face_classifier3.detectMultiScale(gray,1.3,5)
                if faces is ():
                    faces = face_classifier4.detectMultiScale(gray,1.3,5)
                    if faces is ():
                        faces = face_classifier5.detectMultiScale(gray,1.3,5)
                        if faces is ():
                            incorrect+=1

        for (x,y,w,h) in faces:

            correct+=1
            '''gray = gray[y:y+h, x:x+w] #Cut the frame to size

            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite(image, out) #Write image
            except:
               pass #If error, pass file'''






def checkDetect():

    cnt = 0
    incorrect = 0
    #for x in glob.glob(test):
    for subdir in glob.glob('F:\Chaos\EmotionRecognition\Download_CK+\\faces\\at33\*.pgm'):

            image = cv2.imread(subdir)
            cv2.imshow('image',image)
            cv2.waitKey(0)
            '''gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
            if faces is ():
                faces = face_classifier2.detectMultiScale(gray,1.3,5)
                if faces is ():
                    faces = face_classifier3.detectMultiScale(gray,1.3,5)
                    if faces is ():
                        faces = face_classifier4.detectMultiScale(gray,1.3,5)
                        if faces is ():
                            faces = face_classifier5.detectMultiScale(gray,1.3,5)
                            if faces is ():
                                faces = lbph_classifier.detectMultiScale(gray,1.3,5)
                                if faces is ():
                                    print ('{0}'.format(subdir))
                                    incorrect+=1
            for (x,y,w,h) in faces:

                image = image[y:y+h, x:x+w]
                cv2.imshow('image',image)
                cv2.waitKey(0)
            cnt+=1'''

    #print ('Correct = {0}, Incorrect = {1}, Percentage = {2}'.format(cnt-incorrect,incorrect,(100*(cnt-incorrect))/cnt))




#resizeIndividual()
#checkDetect()
#filename = 'F:\Chaos\EmotionRecognition\Application\1.jpg'
for emotion in emotions:
    for filename in glob.glob(os.path.join(test,emotion,'*')):
        image = cv2.imread(filename)
        print(image)
        print('\n')
        '''try:
            out = cv2.resize(image, (350,350)) #Resize face so all images have same size
            cv2.imwrite(filename, out) #Write image
        except:
            os.remove(filename)'''
'''correct=0
incorrect=0
for emotion in emotions:
    detect_faces(emotion,correct,incorrect)

#print ('Correct {0} , Incorrect {1}'.format(correct,incorrect))
print ('Correct {0}, Incorrect {1}, Percentage {2}'.format(correct,incorrect,(100*correct)/(correct+incorrect)))
'''
