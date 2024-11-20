import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

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
  
  #      cv2.circle(monedas_with_circles, (a, b), r, (0, 255, 0), 2)
        
        # Draw a small circle (of radius 5) to show the center.
 #       cv2.circle(monedas_with_circles, (a, b), 5, (0, 0, 255), 3)


# Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
#monedas_with_circles_rgb = cv2.cvtColor(monedas_with_circles, cv2.COLOR_BGR2RGB)

# Display the result using Matplotlib
plt.imshow(monedas_with_circles, cmap='gray')
plt.axis("off")  # Hide the axes
plt.title("Detected Circles")
plt.show()

# Aplicar un umbral para binarizar la imagen
_, monedas_negras = cv2.threshold(monedas_with_circles, 2, 255, cv2.THRESH_BINARY_INV)

print(monedas_negras)

# Mostrar la imagen binarizada
plt.imshow(monedas_negras, cmap='gray')
plt.axis('off')  # Desactivar los ejes
plt.title("Binarized Image (0 and 1)")
plt.show()

mask = monedas_negras == 0
monedas_selected = monedas.copy()
monedas_selected[~mask] = 0  # Establecer los valores de monedas_selected a 0 donde la máscara es True

# Mostrar la imagen final con solo los píxeles seleccionados de monedas_gray
plt.imshow(monedas_selected, cmap='gray')
plt.axis('off')  # Desactivar los ejes
plt.title("Filtered Image from Monedas Gray")
plt.show()

monedas_recortes_gray = cv2.cvtColor(monedas_selected, cv2.COLOR_BGR2GRAY)
plt.imshow(monedas_recortes_gray, cmap='gray')
plt.axis('off')  # Desactivar los ejes
plt.title("Filtered Image from Monedas Gray")
plt.show()


monedas_resized: np.ndarray =  cv2.resize(monedas_negras, (1366, 768))

totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(monedas_resized, 4, cv2.CV_32S)


for i in range(1, totalLabels): 
    
      # Area of the component 
    area = values[i, cv2.CC_STAT_AREA]  
      
    if (area > 140) and (area < 400): 
        # Create a new image for bounding boxes 
        new_img=monedas.copy() 
          
        # Now extract the coordinate points 
        x1 = values[i, cv2.CC_STAT_LEFT] 
        y1 = values[i, cv2.CC_STAT_TOP] 
        w = values[i, cv2.CC_STAT_WIDTH] 
        h = values[i, cv2.CC_STAT_HEIGHT] 
          
        # Coordinate of the bounding box 
        pt1 = (x1, y1) 
        pt2 = (x1+ w, y1+ h) 
        (X, Y) = centroid[i] 
          
        # Bounding boxes for each component 
        cv2.rectangle(new_img,pt1,pt2, 
                      (0, 255, 0), 3) 
        cv2.circle(new_img, (int(X), 
                             int(Y)),  
                   4, (0, 0, 255), -1) 
  
        # Create a new array to show individual component 
        component = np.zeros(monedas_recortes_gray.shape, dtype="uint8") 
        componentMask = (label_ids == i).astype("uint8") * 255
  
        # Apply the mask using the bitwise operator 
        component = cv2.bitwise_or(component,componentMask) 
        output = cv2.bitwise_or(output, componentMask) 
          
        # Show the final images 
        imshow(output) 