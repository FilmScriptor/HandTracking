import cv2
import mediapipe as mp



class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #hand detection model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)

       #draw hand points
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        #send RGB img to hands object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #return result of hands object in RGB
        #print(results.multi_hand_landmarks)

        #extract information of each hand, draw 21 dots on each hand and connect them
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                               self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        return img

    #find function
    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

        return lmList


