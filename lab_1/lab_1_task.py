import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "lab_1/margo.jpg"
image = cv2.imread(image_path)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
kernel = np.array([[0,-1, 0],[-1, 5,-1],[0,-1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)
edges_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
edges_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.magnitude(edges_x, edges_y)
edges = cv2.convertScaleAbs(edges)
combined = cv2.addWeighted(blurred_image, 0.5, edges, 0.5, 0)
combined = cv2.addWeighted(combined, 0.5, sharpened, 0.5, 0)

def show_images(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('Выделение границ')
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости ')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('Комбинированное изображение')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

show_images(image, blurred_image, edges, sharpened, combined)
