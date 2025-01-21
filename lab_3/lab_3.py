import cv2
import time
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(1)

prev_time = 0

def is_eye_closed(eye_region):
    (x, y, w, h) = eye_region
    eye_roi = gray[y:y + h, x:x + w]
    
    _, threshold_eye = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
    
    white_area = np.sum(threshold_eye == 255)
    
    if white_area < (w * h * 0.4):  
        return True  
    return False 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        eye_closed = False
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            if is_eye_closed((x + ex, y + ey, ew, eh)):
                eye_closed = True
                cv2.putText(frame, "Open your eyes!", (x + ex, y + ey + eh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smile) == 0:
            cv2.putText(frame, "Smile!", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 255), 2)

        if eye_closed:
            cv2.putText(frame, "Eyes are closed!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    current_time = time.time()
    time_difference = current_time - prev_time
    fps = 1 / time_difference if time_difference > 0 else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
