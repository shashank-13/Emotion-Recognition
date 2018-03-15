import cv2
import glob
import os
import face_recognition

face_classifier = cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_default.xml')
face_classifier2 = cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt2.xml')
face_classifier3=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt.xml')
face_classifier4=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_frontalface_alt_tree.xml')
face_classifier5=cv2.CascadeClassifier('F:\Chaos\EmotionRecognition\Download_CK+\haarcascade_profileface.xml')

test='F:\Chaos\EmotionRecognition\Application\model\images\*'

def runValidation():

    for image in glob.glob('F:\Chaos\EmotionRecognition\Application\dataset\\anger\*.jpg'):
        '''frame =  face_recognition.load_image_file(image)
        face_locations=face_recognition.face_locations(frame)

        if(len(face_locations)==0):
            print ('Not Detected {0}'.format(image))'''

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
                            print ('Not Detected {0}'.format(image))




def detect_images():

    correct = 0
    incorrect = 0
    for subdir in glob.glob(test):
        for image in glob.glob(subdir+"\*.jpg"):

            frame = cv2.imread(image)
            face_locations = face_recognition.face_locations(frame)

            if(len(face_locations)==0):
                incorrect+=1
            else:
                correct+=1
            '''for face_location in face_locations:

                top, right, bottom, left = face_location
                # You can access the actual face itself like this:
                face_image = frame[top:bottom, left:right]
                cv2.imshow('Image',face_image)
                cv2.waitKey(0)'''

            #print (image)
            '''frame = cv2.imread(image)
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
                frame = frame[y:y+h, x:x+w]
                cv2.imshow('image',frame)
                cv2.waitKey(0)'''

    print ('Correct {0}, Incorrect {1}, Percentage {2}'.format(correct,incorrect,(100*correct)/(correct+incorrect)))
    cv2.destroyAllWindows()
#detect_images()
runValidation()
