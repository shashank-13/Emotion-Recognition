import cv2
import glob
import os
import face_recognition

emotions = ["anger"] #Define emotions
new_Dir = 'F:\Chaos\EmotionRecognition\Application\FER_CB_256_COLOR'



def detect_faces(emotion,correct,incorrect):

    for image in glob.glob(new_Dir+"\\"+emotion+"\*"):

        frame = face_recognition.load_image_file(image)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(frame)

        if(len(face_locations)==0):
            incorrect+=1
        else:
            correct+=1
    return correct,incorrect

def runValidation():

    correct = 0
    incorrect=0
    for image in glob.glob('F:\Chaos\EmotionRecognition\Application\FER_CB_256_COLOR\\anger\*.png'):
        frame =  face_recognition.load_image_file(image)
        face_locations=face_recognition.face_locations(frame)

        if(len(face_locations)==0):
            print ('Not Detected {0}'.format(image))
            incorrect+=1
        else:
            correct+=1
    print ('Correct {0} , Incorrect {1}'.format(correct,incorrect))




runValidation()
'''correct=0
incorrect=0
for emotion in emotions:
    correct,incorrect=detect_faces(emotion,correct,incorrect)

#print ('Correct {0} , Incorrect {1}'.format(correct,incorrect))
print ('Correct {0}, Incorrect {1}, Percentage {2}'.format(correct,incorrect,(100*correct)/(correct+incorrect)))
'''
