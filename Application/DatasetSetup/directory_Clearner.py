import glob
import os
import shutil
import random
import cv2

src ='F:\Chaos\EmotionRecognition\Application\dataset'
dest ='F:\Chaos\EmotionRecognition\Application\sorted_set'

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions


def copyFiles():

    for emotion in emotions:
        files = glob.glob(os.path.join(src,emotion,'*.jpg'))
        #random.shuffle(files)
        #files = files[:int(len(files)*0.1)]
        for f in files:
            face_image=cv2.imread(f)
            filename = os.path.join(dest,emotion,f.replace(os.path.join(src,emotion),'')[1:])
            #print ('{0}'.format(f.replace(os.path.join(src,emotion),'')[1:]))
            #shutil.copyfile(f,os.path.join(dest,emotion,f.replace(os.path.join(src,emotion),'')[1:]))
            try:
                out = cv2.resize(face_image, (48,48)) #Resize face so all images have same size
                cv2.imwrite(filename, out) #Write image
            except:
                pass


def reduceSize():

    for emotion in emotions:
        files = glob.glob(os.path.join(dest,emotion,'*.jpg'))
        temp_files = files[:int(len(files)*0.2)]
        for f in temp_files:
            os.remove(f)

def cleanDirectory():

    main_Directory = 'F:\Chaos\EmotionRecognition\Download_CK+\Emotion_labels\Emotion\*'
    sub_Dir = 'F:\Chaos\EmotionRecognition\Download_CK+'

    list1=[]
    list2=[]

    for x in glob.glob(main_Directory):
        for subdir in glob.glob(x+"\*"):
            if not os.listdir(subdir):
                list1.append(subdir)
                part = "%s" %subdir[-9:]
                temp_Dir=sub_Dir+'\extended-cohn-kanade-images\cohn-kanade-images'+part
                list2.append(temp_Dir)


    for first,second in zip(list1,list2):
        #print ("Removed Directories {0} {1}".format(first,second))
        os.rmdir(first)
        shutil.rmtree(second)

def moveToSorted():
    main_Directory = 'F:\Chaos\EmotionRecognition\Download_CK+\Emotion_labels\Emotion\*'
    dest_Directory ='F:\Chaos\EmotionRecognition\Application\sorted_set'
    sub_Dir = 'F:\Chaos\EmotionRecognition\Download_CK+\extended-cohn-kanade-images\cohn-kanade-images'
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

    for x in glob.glob(main_Directory):
        for subdir in glob.glob(x+"\*"):
            for files in glob.glob(subdir+"\*.txt"):
                file_n = open(files, 'r')
                emotion = int(float(file_n.readline()))
                files=files.replace('.txt','.png')
                files=files.replace('_emotion','')
                files=sub_Dir +files[-31:]
                source_emotion=files
                source_neutral=files[:112]+'01'+files[114:]






                dest_emotion = "%s\%s\%s" %(dest_Directory,emotions[emotion],source_emotion[-21:])
                dest_neutral = "%s\\neutral\%s" %(dest_Directory,source_neutral[-21:])

                #print (dest_neutral)
                #print ("{0} {1}".format(dest_neutral,dest_emotion))

                shutil.copyfile(source_neutral,dest_neutral)
                shutil.copyfile(source_emotion,dest_emotion)


#moveToSorted()
copyFiles()
#reduceSize()
