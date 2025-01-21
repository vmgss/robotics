import cv2
import numpy as np

def rgb_to_hsv(r, g, b):
    rgb = np.uint8([[[r, g, b]]]) 
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV) 
    return hsv[0][0] 

color1_rgb = (56, 39, 72) 
color2_rgb = (80, 57, 50)  

color1_hsv = rgb_to_hsv(*color1_rgb)
color2_hsv = rgb_to_hsv(*color2_rgb)

print(f"Color 1 HSV: {color1_hsv}")
print(f"Color 2 HSV: {color2_hsv}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка при открытии камеры!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    lower_color1 = np.array([color1_hsv[0]-20, 50, 50]) 
    upper_color1 = np.array([color1_hsv[0]+20, 255, 255]) 

    lower_color2 = np.array([color2_hsv[0]-20, 50, 50])  
    upper_color2 = np.array([color2_hsv[0]+20, 255, 255])  
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)

    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
