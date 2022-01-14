import cv2
import numpy as np
import time
import math

from openpose import pyopenpose as op

from sklearn.ensemble import RandomForestClassifier
import joblib

def set_params():
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models"
    params["model_pose"] = "COCO"
#    params["net_resolution"] = "128x64"
    params["net_resolution"] = "256x128"
    return params

def AngleFromVectors(va, vb):
    cosTh = np.dot(va,vb)
    sinTh = np.cross(va,vb)
    return np.arctan2(sinTh,cosTh)

def Len2AdjustData(skldat):
    if skldat[2][2]!=0 and skldat[5][2]!=0:
        l1 = abs(math.sqrt((skldat[2][0]-skldat[1][0])**2+(skldat[2][1]-skldat[1][1])**2)) #point 3
        l2 = abs(math.sqrt((skldat[5][0]-skldat[1][0])**2+(skldat[5][1]-skldat[1][1])**2)) #point 6

        if l1>=l2:
            l = l1
        else:
            l = l2

    elif skldat[2][2]!=0:
        l = math.sqrt((skldat[2][0]-skldat[1][0])**2+(skldat[2][0]-skldat[1][1])**2) #point 3
    else:
        l = math.sqrt((skldat[5][0]-skldat[1][0])**2+(skldat[5][1]-skldat[1][1])**2) #point 6

    return abs(l)

def DataProcessing03(skldat):
    # pair = np.array([[1,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8], [1,9], [9,11], [1,10], [10,12]])
    pair = np.array([[1,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8]])
    pair = pair-1
    
    vectors = np.zeros((len(pair), 2))
    
    l = Len2AdjustData(skldat)
    
    for i in range(len(pair)):
        if skldat[pair[i][0]][2]!=0 and skldat[pair[i][1]][2]!=0:
            vectors[i][0] = (skldat[pair[i][1]][0]-skldat[pair[i][0]][0])/l
            vectors[i][1] = (skldat[pair[i][1]][1]-skldat[pair[i][0]][1])/l
        
    return vectors
    
def DataProcessing04(skldat):
    pair = np.array([[1,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8], [1,9], [9,11], [1,10], [10,12]])
    pair = pair-1
    
    angles = np.zeros((1,5))
    
    for i in range(len(pair)):
        for j in range(i+1, len(pair)):
            if skldat[pair[j][0]][2]!=0 and skldat[pair[j][1]][2]!=0 and skldat[pair[i][0]][2]!=0 and skldat[pair[i][1]][2]!=0:
                if pair[i][0]==pair[j][0]:
                    vectorA = [skldat[pair[i][1]][0]-skldat[pair[i][0]][0], skldat[pair[i][1]][1]-skldat[pair[i][0]][1]]
                    vectorB = [skldat[pair[j][1]][0]-skldat[pair[j][0]][0], skldat[pair[j][1]][1]-skldat[pair[j][0]][1]]
                    angle = AngleFromVectors(vectorA, vectorB)
                    angles = np.vstack((angles, [pair[i][0]+1, pair[i][1]+1, pair[j][0]+1, pair[j][1]+1, angle]))
                    
                elif pair[i][0]==pair[j][1]:
                    vectorA = [skldat[pair[i][1]][0]-skldat[pair[i][0]][0], skldat[pair[i][1]][1]-skldat[pair[i][0]][1]]
                    vectorB = [skldat[pair[j][0]][0]-skldat[pair[j][1]][0], skldat[pair[j][0]][1]-skldat[pair[j][1]][1]]
                    angle = AngleFromVectors(vectorA, vectorB)
                    angles = np.vstack((angles, [pair[i][0]+1, pair[i][1]+1, pair[j][1]+1, pair[j][0]+1, angle]))
                    
                elif pair[i][1]==pair[j][0]:
                    vectorA = [skldat[pair[i][0]][0]-skldat[pair[i][1]][0], skldat[pair[i][0]][1]-skldat[pair[i][1]][1]]
                    vectorB = [skldat[pair[j][1]][0]-skldat[pair[j][0]][0], skldat[pair[j][1]][1]-skldat[pair[j][0]][1]]
                    angle = AngleFromVectors(vectorA, vectorB)
                    angles = np.vstack((angles, [pair[i][1]+1, pair[i][0]+1, pair[j][0]+1, pair[j][1]+1, angle]))
                    
                elif pair[i][1]==pair[j][1]:
                    vectorA = [skldat[pair[i][0]][0]-skldat[pair[i][1]][0], skldat[pair[i][0]][1]-skldat[pair[i][1]][1]]
                    vectorB = [skldat[pair[j][0]][0]-skldat[pair[j][1]][0], skldat[pair[j][0]][1]-skldat[pair[j][1]][1]]
                    angle = AngleFromVectors(vectorA, vectorB)
                    angles = np.vstack((angles, [pair[i][1]+1, pair[i][0]+1, pair[j][1]+1, pair[j][0]+1, angle]))
                    
            else:
                if pair[i][0]==pair[j][0]:
                    angles = np.vstack((angles, [pair[i][0]+1, pair[i][1]+1, pair[j][0]+1, pair[j][1]+1, 0]))
                    
                elif pair[i][0]==pair[j][1]:
                    angles = np.vstack((angles, [pair[i][0]+1, pair[i][1]+1, pair[j][1]+1, pair[j][0]+1, 0]))
                    
                elif pair[i][1]==pair[j][0]:
                    angles = np.vstack((angles, [pair[i][1]+1, pair[i][0]+1, pair[j][0]+1, pair[j][1]+1, 0]))
                    
                elif pair[i][1]==pair[j][1]:
                    angles = np.vstack((angles, [pair[i][1]+1, pair[i][0]+1, pair[j][1]+1, pair[j][0]+1, 0]))
                
    return angles[1:]

def DataProcessing05(sktdata):
    #with X
    pair = np.array([[9, 10], [3, 6]])
    pair = pair-1

    angles = np.zeros((1,5))

    for i in range(len(pair)):
        if sktdata[pair[i,0]][2]!=0 and sktdata[pair[i,1]][2]!=0:
            vectorA = [sktdata[pair[i,1]][0]-sktdata[pair[i,0]][0], sktdata[pair[i,1]][1]-sktdata[pair[i,0]][1]]
            vectorB = [50, 0]
            angle = AngleFromVectors(vectorA, vectorB)
            angles = np.vstack((angles, [pair[i][0]+1, pair[i][1]+1, 'x', 'x', angle]))

        else:
            angles = np.vstack((angles, [pair[i][0]+1, pair[i][1]+1, 'x', 'x', 0]))

    # print(angles[1:])
    # print(len(angles[1:]))
    # test
    return angles[1:]

def UpperBodyData(skldat):
    keypoints = np.array([[1,8],[15,18]])
    keypoints = keypoints-1

    chk1 = True
    for i in range(len(keypoints)):
        if chk1:
            new_skldat = skldat[keypoints[i, 0]:keypoints[i, 1]+1]
            chk1 = False
        else:
            new_skldat = np.vstack((new_skldat, skldat[keypoints[i, 0]:keypoints[i, 1]+1]))

    return np.array(new_skldat)

def DataProcessing(skldat):
    if len(skldat)>1:
        skldat = UpperBodyData(skldat)
        dataProc3 = DataProcessing03(skldat)
        dataProc4 = DataProcessing04(skldat)[:,-1]
        dataProc5 = DataProcessing05(skldat)[:,-1]

        dataProc = np.array([])
        for i in range(len(dataProc3)):
            dataProc = np.append(dataProc, dataProc3[i,0])
            dataProc = np.append(dataProc, dataProc3[i,1])

        dataProc = np.append(dataProc, dataProc4)
        dataProc = np.append(dataProc, dataProc5)

    else:
        dataProc = []

    return dataProc

def SkeletonData1stPerson(datum, opWrapper, img):
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if not datum.poseKeypoints is None:
        if len(datum.poseKeypoints)==1:
            if datum.poseKeypoints[0][0,2]!=0 and datum.poseKeypoints[0][1,2]!=0 and (datum.poseKeypoints[0][2,2]!=0 or datum.poseKeypoints[0][6,2]!=0):
                skldat = datum.poseKeypoints[0]
                img_out = datum.cvOutputData

            else:
                skldat = []
                img_out = img

        else:
            ikey = -1
            miniL = 0

            for k in range(len(datum.poseKeypoints)):
                if datum.poseKeypoints[k][0,2]!=0 and datum.poseKeypoints[k][1,2]!=0:
                    if datum.poseKeypoints[k][14,2]!=0:
                        l1 = math.sqrt(math.pow((datum.poseKeypoints[k][0,0]-datum.poseKeypoints[k][14,0]),2)+math.pow((datum.poseKeypoints[k][0,1]-datum.poseKeypoints[k][14,1]),2))
                    else:
                        l1 = 0

                    if datum.poseKeypoints[k][15,2]!=0:
                        l2 = math.sqrt(math.pow((datum.poseKeypoints[k][0,0]-datum.poseKeypoints[k][15,0]),2)+math.pow((datum.poseKeypoints[k][0,1]-datum.poseKeypoints[k][15,1]),2))
                    else:
                        l2 = 0

                    if l1>l2:
                        l = l1
                    else:
                        l = l2

                    if l>=miniL:
                        ikey = k
                        miniL = l

            if ikey>=0:
                skldat = datum.poseKeypoints[ikey]
                img_out = datum.cvOutputData
            else:
                img_out = img
                skldat = []

    else:
        img_out = img
        skldat = []

    return skldat, img_out

def PostureClassification(postures, feat, model, prob=0.5):
    predictions = model.predict_proba(feat.reshape(1, -1))[0]
#    print(predictions[2], predictions[4])
    if max(predictions)>=prob:
        return postures[int(np.argmax(predictions))]
    else:
        return "unknow"

def SetCameraSize(cap, width=1280, height=720):
    cap.set(3, width)
    cap.set(4, height)
    return cap, width, height

def main():
    params = set_params()

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    rf = joblib.load("models/modelDL.joblib")
    postures = ['Happy', 'Sleep', 'Work', 'Think']

    try:
        cap = cv2.VideoCapture(0)
        cap, wid, hih = SetCameraSize(cap)

    except e:
        print('Error: Don\'t found the camera.')
        raise e

    while(cap.isOpened()):
        tic = time.time()
        ret, img = cap.read()

        if not ret:
            break

        skldat, img_out = SkeletonData1stPerson(datum, opWrapper, img)
        feat = DataProcessing(skldat)
        if len(feat)>1:
            posture = PostureClassification(postures, feat, rf)
            img_out = cv2.putText(img_out, "Posture: {}".format(posture), (10, hih-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            print("posture is", posture, ": fps is %.2f"%(1/(time.time()-tic)))
        else:
            print("fps is %.2f"%(1/(time.time()-tic)))

        img_out = cv2.putText(img_out, "FPS: %.2f"%(1/(time.time()-tic)), (wid-170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('test', img_out)

#        print("posture is", posture, ": fps is %.2f"%(1/(time.time()-tic)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
