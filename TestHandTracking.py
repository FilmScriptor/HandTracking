import cv2
import time
import HandTrackingModule as htm


def main():

    # write out frame rate
    pTime = 0  # previous time
    cTime = 0  # current time

    # using camera
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        success, img = cap.read()
        img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[10])
            print(lmList[5])

        cTime = time.time()  # current time
        fps = 1 / (cTime - pTime)  # fps time calculate
        pTime = cTime  # previous time become current time

        # display on screen, on screen location, font, scale, color, thickness
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Input', img)
        c = cv2.waitKey(1)


if __name__ == "__main__":
    main()