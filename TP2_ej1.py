import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'data')

monedas: np.ndarray =  cv2.imread(os.path.join(DATA_PATH,'monedas.jpg'))
monedas_gray: np.ndarray = cv2.cvtColor(monedas, cv2.COLOR_BGR2GRAY)
plt.imshow(monedas_gray, cmap='gray')
plt.show()

monedas_blur: np.ndarray = cv2.GaussianBlur(monedas_gray, (3, 3), 0)


detected_circles = cv2.HoughCircles(monedas_blur,  
                   cv2.HOUGH_GRADIENT, 1.2, 90, param1 = 65, 
               param2 = 170, minRadius = 70, maxRadius = 200)     

print(f'Circles detected: {detected_circles}')

if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    
    # Copy the image to avoid drawing directly on the original
    monedas_with_circles = monedas_blur.copy()

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        
        cv2.circle(monedas_with_circles, (a, b), r, (0, 0, 0), -1)  # Interior negro

        # Draw the circumference of the circle.
        cv2.circle(monedas_with_circles, (a, b), r, (0, 255, 0), 2)
        
        # Draw a small circle (of radius 5) to show the center.
        cv2.circle(monedas_with_circles, (a, b), 5, (0, 0, 255), 3)


# Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
monedas_with_circles_rgb = cv2.cvtColor(monedas_with_circles, cv2.COLOR_BGR2RGB)

# Display the result using Matplotlib
plt.imshow(monedas_with_circles_rgb, cmap='gray')
plt.axis("off")  # Hide the axes
plt.title("Detected Circles")
plt.show()

# Aplicar un umbral para binarizar la imagen
_, monedas_negras = cv2.threshold(monedas_with_circles_rgb, 1, 255, cv2.THRESH_BINARY)

print(monedas_negras)

# Mostrar la imagen binarizada
plt.imshow(monedas_negras, cmap='gray')
plt.axis('off')  # Desactivar los ejes
plt.title("Binarized Image (0 and 1)")
plt.show()


mask = monedas_negras == 255
monedas_selected = monedas_gray.copy()
monedas_selected[mask] = 0  # Establecer los valores de monedas_selected a 0 donde la máscara es True

# Mostrar la imagen final con solo los píxeles seleccionados de monedas_gray
plt.imshow(monedas_selected, cmap='gray')
plt.axis('off')  # Desactivar los ejes
plt.title("Filtered Image from Monedas Gray")
plt.show()

# Ver la matriz de monedas_selected (opcional)
print(np.unique(monedas_selected))







