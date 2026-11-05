import cv2, mediapipe as mp
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ok, img = cam.read()
    if not ok:
        print("no frame")
        continue
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            draw.draw_landmarks(img, h, mp.solutions.hands.HAND_CONNECTIONS)
            lm = h.landmark
            count = 0
            for t in [8, 12, 16, 20]:
                if lm[t].y < lm[t-2].y:
                    count += 1
            if lm[4].x < lm[3].x:
                count += 1
            cv2.putText(img, str(count), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    cv2.imshow("cam", img)
    if cv2.waitKey(1) == 27: break
cam.release()
cv2.destroyAllWindows()