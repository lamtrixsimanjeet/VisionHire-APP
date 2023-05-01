import datetime
from csv import writer 
import cv2 
import mediapipe as mp
import math
import os 
import time
import torch
import cv2
import pandas as pd
import numpy as np
import _thread
#DataFrame to store data for Plot.py to plot
df1 = pd.DataFrame(columns=['Face_Count','EAR','MAR'])
#Used for Perclos calculation
FRAME_COUNTER_RESET_VALUE = 45

#Threading to call Plot.py
def plot():
	os.system('python Plot.py')
    
#Defining Face Landmark Detection Module
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

#Basic Constants For Further Calculation Reference
flag=1
unique_faces = {}
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
minThreshold = 40
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark coordinates
states = ['alert', 'drowsy']
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# Declaring FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
model_lstm_path = 'models\clf_lstm_jit6.pth'
model = torch.jit.load(model_lstm_path)
model.eval()

#Function to Verify face

def check2(im,minW,minH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()   # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("CandidateDetails"+os.sep+"CandidateDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5,minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
    for(x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 100:
            aa = df.loc[df['Id'] == Id]['Name'].values
            confstr = "  {0}%".format(round(100 - conf))
            tt = str(Id)+"-"+aa
        else:
            Id = '  Unknown  '
            tt = str(Id)
            confstr = "  {0}%".format(round(100 - conf))         
       # if (100-conf) > 67:
        verify=1
        if (100-conf) < minThreshold:
            verify=0
            
            cv2.putText(im, str(tt), (x+5,y-5), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        if (100-conf) > minThreshold:
            cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
        elif (100-conf) > 50:
            cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
        else:
            cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)
        return verify

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    
    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)
    
    # drawing Eyes Shape on mask with white color 
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv2.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position 

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
    elif max_index==1:
        pos_eye = 'CENTER'
    elif max_index ==2:
        pos_eye = 'LEFT'
    else:
        pos_eye="Closed"
    return pos_eye
def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2])**2).sum())**0.5

# if detects more than one face then prints fraud
def check(image):
    mp_face_detection = mp.solutions.face_detection

    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    face_detection_results = face_detection.process(image)
    count=0
    if face_detection_results.detections:
        count= len(face_detection_results.detections)
        return count


def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    ''' Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    '''
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def run_face_mp1(image):
    ''' Get face landmarks using the FaceMesh MediaPipe model. 
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :return: Feature 1 (Eye), Feature 2 (Mouth), Feature 3 (Pupil), \
        Feature 4 (Combined eye and mouth feature), image with mesh drawings
    '''
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #mp_drawing.draw_landmarks(image, faceLms, mp_face_mesh.FACEMESH_CONTOURS)
           
            h, w, c = image.shape
            cx_min=  w
            cy_min = h
            cx_max= cy_max= 0
            for id, lm in enumerate(faceLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx<cx_min:
                    cx_min=cx
                if cy<cy_min:
                    cy_min=cy
                if cx>cx_max:
                    cx_max=cx
                if cy>cy_max:
                    cy_max=cy
            cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        # for face_landmarks in results.multi_face_landmarks:
        #             mp_drawing.draw_landmarks(
        #             image=image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_TESSELATION,
        #             landmark_drawing_spec=drawing_spec,
        #             connection_drawing_spec=drawing_spec)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image

def run_face_mp(image,CEF_COUNTER, TOTAL_BLINKS,CLOSED_EYES_FRAME,frame_counter,num_blinks,perclos):
    ''' Get face landmarks using the FaceMesh MediaPipe model. 
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :return: Feature 1 (Eye), Feature 2 (Mouth), Feature 3 (Pupil), \
        Feature 4 (Combined eye and mouth feature), Feature 5 (Total Blink Count and Perclos)
    '''
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    decay = 0.9 # for smoothning
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #(Total Blink Count and Perclos)
    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(image, results, False)
        ratio = blinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)

        if ratio >5.5:
            CEF_COUNTER +=1
            cv2.putText(image,'BLINK', (int(0.57*image.shape[1]), int(0.2*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        else:
            if CEF_COUNTER>CLOSED_EYES_FRAME:
                TOTAL_BLINKS +=1
                #num_blinks=num_blinks*decay+(1-decay)
                num_blinks+=1
                CEF_COUNTER=0
        cv2.putText(image,'Total Blinks:'+str(TOTAL_BLINKS), (int(0.47*image.shape[1]), int(0.12*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        right_coords = [mesh_coords[p] for p in RIGHT_EYE]
        left_coords = [mesh_coords[p] for p in LEFT_EYE]
        crop_right, crop_left = eyesExtractor(image, right_coords, left_coords)
        
        eye_position1 = positionEstimator(crop_right)
        
        for faceLms in results.multi_face_landmarks:
            #mp_drawing.draw_landmarks(image, faceLms, mp_face_mesh.FACEMESH_CONTOURS)
            #my change
            
            h, w, c = image.shape
            cx_min=  w
            cy_min = h
            cx_max= cy_max= 0
            for id, lm in enumerate(faceLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx<cx_min:
                    cx_min=cx
                if cy<cy_min:
                    cy_min=cy
                if cx>cx_max:
                    cx_max=cx
                if cy>cy_max:
                    cy_max=cy
            cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        # for face_landmarks in results.multi_face_landmarks:
        #             mp_drawing.draw_landmarks(
        #             image=image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_TESSELATION,
        #             landmark_drawing_spec=drawing_spec,
        #             connection_drawing_spec=drawing_spec)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000
        
    frame_counter += 1
    if frame_counter == FRAME_COUNTER_RESET_VALUE:
        frame_counter = 0
        perclos = num_blinks / FRAME_COUNTER_RESET_VALUE
        num_blinks=num_blinks*decay
    return ear, mar, puc, moe, image,eye_position1,CEF_COUNTER, TOTAL_BLINKS,CLOSED_EYES_FRAME,frame_counter,num_blinks,perclos

def calibrate(calib_frame_count=25):
    ''' Perform clibration. Get features for the neutral position.
    :param calib_frame_count: Image frames for which calibration is performed. Default Vale of 25.
    :return: Normalization Values for feature 1, Normalization Values for feature 2, \
        Normalization Values for feature 3, Normalization Values for feature 4
    '''
    ears = []
    mars = []
    pucs = []
    moes = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        if image is None:
            continue
        #detect(image)
        cap.set(3,640)
        cap.set(4,480)
        # minW=0.1*cap.get(3)
        # minH=0.1*cap.get(4)
        ear, mar,puc, moe, image= run_face_mp1(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)

        cv2.putText(image, "Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
        [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

def get_classification(input_data):
    ''' Perform classification over the facial  features.
    :param input_data: List of facial features for 20 frames
    :return: Alert / Drowsy state prediction
    '''
    model_input = []
    model_input.append(input_data[:5])
    model_input.append(input_data[3:8])
    model_input.append(input_data[6:11])
    model_input.append(input_data[9:14])
    model_input.append(input_data[12:17])
    model_input.append(input_data[15:])
    model_input = torch.FloatTensor(np.array(model_input))
    preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
    return int(preds.sum() >= 5)

def infer(ears_norm, mars_norm, pucs_norm, moes_norm):
    ''' Perform inference.
    :param ears_norm: Normalization values for eye feature
    :param mars_norm: Normalization values for mouth feature
    :param pucs_norm: Normalization values for pupil feature
    :param moes_norm: Normalization values for mouth over eye feature. 
    '''
    i=0
    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0
    print('time,EAR',  file=open('file.csv', 'w'))
    frame_counter = 0
    blink_frame_counter = 0
    num_blinks = 0
    CLOSED_EYES_FRAME = 3
    
    decay = 0.9 # use decay to smoothen the noise in feature values
    _thread.start_new_thread(plot,())
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    
    global df1
    label = None
    input_data = []
    frame_before_run = 0
    perclos = 0

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        cap.set(3,640)
        cap.set(4,480)
        # minW=0.1*cap.get(3)
        # minH=0.1*cap.get(4)
        # check2(image,minW,minH)    
        count=check(image)
        
        ear, mar, puc, moe, image,look,CEF_COUNTER, TOTAL_BLINKS,CLOSED_EYES_FRAME,frame_counter,num_blinks,perclos = run_face_mp(image,CEF_COUNTER, TOTAL_BLINKS,CLOSED_EYES_FRAME,frame_counter,num_blinks,perclos)
        
        mp_drawing = mp.solutions.drawing_utils 
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        if ear != -1000:
            ear = (ear - ears_norm[0])/ears_norm[1]
            mar = (mar - mars_norm[0])/mars_norm[1]
            puc = (puc - pucs_norm[0])/pucs_norm[1]
            moe = (moe - moes_norm[0])/moes_norm[1]
            if ear_main == -1000:
                ear_main = ear
                mar_main = mar
                puc_main = puc
                moe_main = moe
            else:
                ear_main = ear_main*decay + (1-decay)*ear
                mar_main = mar_main*decay + (1-decay)*mar
                puc_main = puc_main*decay + (1-decay)*puc
                moe_main = moe_main*decay + (1-decay)*moe
        else:
            ear_main = -1000
            mar_main = -1000
            puc_main = -1000
            moe_main = -1000
        
        if len(input_data) == 20:
            input_data.pop(0)
        input_data.append([ear_main, mar_main, puc_main, moe_main])

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            label = get_classification(input_data)
            print ('got label ', label)
        
        cv2.putText(image, "EAR: %.2f" %(ear_main), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "MAR: %.2f" %(mar_main), (int(0.27*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "PUC: %.2f" %(puc_main), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "MOE: %.2f" %(moe_main), (int(0.77*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image,look, (int(0.77*image.shape[1]), int(0.13*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image,"PERCLOS: %.4f" %(perclos), (int(0.02*image.shape[1]), int(0.2*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        #df2 = df1.append({'EAR':ear_main, 'MAR':mar_main, 'PUC':puc_main, 'MOE':moe_main, 'Eye_Position':look, 'Face_Count':count}, ignore_index=True)
        #pd.concat([df1, df2])
        print(str(frame_counter)+',' +str(ear_main),  file=open('file.csv','a')) 
        
        # df1.loc[len(df1.index)] = [ear_main, mar_main, puc_main, moe_main, look, count]
        df1.loc[len(df1.index)] = [i,ear_main,mar_main]
        df1.to_csv('result.csv',index=False)
        i+=1
        if(count is None):
            count=1
        if count>1:

            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (50, 150)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            image = cv2.putText(image, 'FRAUD', org, font, 
                               fontScale, color, thickness, cv2.LINE_AA)
        if label is not None:
            if label == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.putText(image, "%s" %(states[label]), (int(0.02*image.shape[1]), int(0.12*image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()
    cap.release()
#-------------------------
def recognize_attendence():

    i=0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        i+=1
        if not success:
            print("Ignoring empty camera frame.")
            continue
        cap.set(3,640)
        cap.set(4,480)
        minW=0.1*cap.get(3)
        minH=0.1*cap.get(4)
        verifycheck=check2(image,minW,minH)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q") or verifycheck==0:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    if verifycheck:
        print ('Starting calibration. Please be in neutral state')
        time.sleep(1)
        ears_norm, mars_norm, pucs_norm, moes_norm = calibrate()

        print ('Starting main application')
        time.sleep(1)
        infer(ears_norm, mars_norm, pucs_norm, moes_norm)

        face_mesh.close()
        
    else:
        print ('Cannot start Face not matching')

    
