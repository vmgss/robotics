import cv2
import numpy as np

video_path = '/Users/vm.gss/Desktop/homeworks/labs_OR/robotics/lab_2/catball.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка при открытии видео!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([15, 100, 100])
    upper_color = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame, f'({cx}, {cy})', (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Object not found', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()