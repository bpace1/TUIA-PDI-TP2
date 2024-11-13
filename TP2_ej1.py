import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'data')

monedas: np.ndarray =  cv2.imread(os.path.join(DATA_PATH,'monedas.jpg'))
monedas_hsv: np.ndarray = cv2.cvtColor(monedas, cv2.COLOR_BGR2HSV)
monedas_gray: np.ndarray = cv2.cvtColor(monedas_hsv, cv2.COLOR_HSV2GRAY)

plt.imshow(monedas, cmap='gray')
plt.show()

monedas_blur: np.ndarray = cv2.GaussianBlur(monedas_gray, (5, 5), 0)

detected_circles = cv2.HoughCircles(monedas_blur,  
                   cv2.HOUGH_GRADIENT, 1.2, 100, param1 = 100, 
               param2 = 150, minRadius = 70, maxRadius = 200)     


if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    
    # Copy the image to avoid drawing directly on the original
    monedas_with_circles = monedas_blur.copy()

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        
        # Draw the circumference of the circle.
        cv2.circle(monedas_with_circles, (a, b), r, (0, 255, 0), 2)
        
        # Draw a small circle (of radius 5) to show the center.
        cv2.circle(monedas_with_circles, (a, b), 5, (0, 0, 255), 3)


# Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
monedas_with_circles_rgb = cv2.cvtColor(monedas_with_circles, cv2.COLOR_BGR2RGB)

# Display the result using Matplotlib
plt.imshow(monedas_with_circles_rgb)
plt.axis("off")  # Hide the axes
plt.title("Detected Circles")
plt.show()