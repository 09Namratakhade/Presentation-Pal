

import os

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from django.shortcuts import render
import mediapipe as mp
from collections import deque

def index(request):
    # Replace 'Presentation' with the actual path to your Presentation folder
    presentation_folder = 'Presentation'

    # Initialize an empty list to store folder names
    folder_list = []

    # Get a list of directories within the Presentation folder
    try:
        folders = os.listdir(presentation_folder)

        # Filter out only directories (excluding files)
        for folder in folders:
            folder_path = os.path.join(presentation_folder, folder)
            if os.path.isdir(folder_path):
                folder_list.append(folder)

    except FileNotFoundError:
        # Handle the case when the Presentation folder does not exist
        return "Presentation folder not found."

    # Now folder_list contains the names of folders within the Presentation folder
    # You can do whatever you want with the folder_list, for example, return it as part of the response
    context = {
        'FolderDets': folder_list,
    }
    return render(request, 'home/template/index.html', context)

def execPresentation(request,Fname):
    # Variables
    width, height = 1280, 720
    folderPath = 'Presentation/'+Fname

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Get list of images
    pathImages = sorted(os.listdir(folderPath), key=len)
    print(pathImages)

    # Variables
    imgNumber = 0
    hs, ws = int(120 * 1), 213
    gestureThreshold = 300
    buttonPressed = False
    buttonCounter = 0
    buttonDelay = 30
    annotations = [[]]
    annotationNumber = -1
    annotationStart = False

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        # Import images
        success, img = cap.read()
        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        # Resize the image to fit the screen
        imgCurrent = cv2.resize(imgCurrent, (width+250, height+70))

        hands, img = detector.findHands(img)

        if hands and buttonPressed is False:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            print(fingers)
            lmList = hand['lmList']

            # Constrain value for easier drawing
            indexFinger = lmList[8][0], lmList[8][1]
            xVal = int(np.interp(lmList[8][0], [width // 2, w], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))

            indexFinger = xVal, yVal
            # Gesture 1
            # Thumb = Backward
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                if imgNumber > 0:
                    imgNumber -= 1
                    buttonPressed = True

            # Gesture 2
            # Pinki Finger = Forward
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    buttonPressed = True

            # Gesture 3
            # Show Pointer = Second and Index Finger
            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            # Gesture 4
            # DrawPointer
            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                annotations[annotationNumber].append(indexFinger)
            else:
                annotationStart = False

            # Gesture 5
            # Erase
            if fingers == [0, 1, 1, 1, 0]:
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True

        # Button Pressed Iterations
        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay:
                buttonCounter = 0
                buttonPressed = False

        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                if j != 0:
                    cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

            # Adding webcam image in slide
            imgSmall = cv2.resize(img, (ws, hs))
            h, w, _ = imgCurrent.shape
            imgCurrent[0:hs, w - ws:w] = imgSmall

        cv2.imshow("Image", img)
        cv2.imshow("Slides", imgCurrent)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


    # Replace 'Presentation' with the actual path to your Presentation folder
    presentation_folder = 'Presentation'

    # Initialize an empty list to store folder names
    folder_list = []

    # Get a list of directories within the Presentation folder
    try:
        folders = os.listdir(presentation_folder)

        # Filter out only directories (excluding files)
        for folder in folders:
            folder_path = os.path.join(presentation_folder, folder)
            if os.path.isdir(folder_path):
                folder_list.append(folder)

    except FileNotFoundError:
        # Handle the case when the Presentation folder does not exist
        return "Presentation folder not found."

    # Now folder_list contains the names of folders within the Presentation folder
    # You can do whatever you want with the folder_list, for example, return it as part of the response
    context = {
        'FolderDets': folder_list,
    }
    return render(request, 'home/template/index.html', context)
#end of presentation cont


#virtual board

def index2(request):
    return render(request, 'home/template/index2.html')

def execVBoard(request):
    # Giving different arrays to handle colour points of different colours
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    # These indexes will be used to mark the points in particular arrays of specific colours
    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    # The kernel to be used for dilation purpose
    kernel = np.ones((5,5),np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Here is code for Canvas setup
    paintWindow = np.zeros((471,636,3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        # Read each frame from the webcam
        ret, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
        frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
        frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
        frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
        frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # Post-process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0,255,0),-1)
            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    paintWindow[67:,:,:] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Draw lines of all the colors on the canvas and frame
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()
    return render(request,'home/template/index2.html')



#image zoomer
def index3(request):
    folder_path = 'images'  # Replace 'path/to/your/images/folder' with the actual path to your 'images' folder
    files_list = os.listdir(folder_path)

    # Now 'files_list' contains the names of all the files in the 'images' folder
    # You can use this list as needed in your function

    # For example, you can print the list of files:
    print(files_list)
    folder_list = files_list
    context = {
        'FolderDets': folder_list,
    }
    return render(request, 'home/template/index3.html',context)


def get_file_extension(folder_path, fName):
    full_path = os.path.join(folder_path, fName)
    _, extension = os.path.splitext(full_path)
    return extension


def execVZoomer(request,Fname):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1200)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8)
    startDist = None
    scale = 0
    cx, cy = 500, 500

    images_folder = "images"  # Replace this with the actual path to your "images" folder

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)
        img_path = os.path.join(images_folder, Fname)
        img1 = cv2.imread(img_path)
        img1 = cv2.resize(img1, (350, 250))  # Resize img1 to match the ROI size

        if len(hands) == 2:
            if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
                lmList1 = hands[0]["lmList"]
                lmList2 = hands[1]["lmList"]

                if startDist is None:
                    # Point 8 is the tip of the index finger
                    x1, y1 = lmList1[8][1], lmList1[8][2]
                    x2, y2 = lmList2[8][1], lmList2[8][2]
                    length, info, img = detector.findDistance([lmList1[8][0], lmList1[8][1]],
                                                              [lmList2[8][0], lmList2[8][1]], img)
                    startDist = length

                length, info, img = detector.findDistance([lmList1[8][0], lmList1[8][1]],
                                                          [lmList2[8][0], lmList2[8][1]],
                                                          img)
                scale = int((length - startDist) // 2)
                cx, cy = info[4:]
                print(scale)

        else:
            startDist = None
        try:
            h1, w1, _ = img1.shape
            newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
            img1 = cv2.resize(img1, (newW, newH))

            img[cy - newH // 2: cy + newH // 2, cx - newW // 2: cx + newW // 2] = img1
        except:
            pass

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    folder_path = 'images'  # Replace 'path/to/your/images/folder' with the actual path to your 'images' folder
    files_list = os.listdir(folder_path)

    # Now 'files_list' contains the names of all the files in the 'images' folder
    # You can use this list as needed in your function

    # For example, you can print the list of files:
    print(files_list)
    folder_list = files_list
    context = {
        'FolderDets': folder_list,
    }
    return render(request, 'home/template/index3.html', context)

