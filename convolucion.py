"""
por Abhisek Jana
codigo tomado de https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
codigo main modificado por Vanessa Méndez
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def conv(fragment, kernel):
    
    f_fila, f_colum = fragment.shape
    k_fila, k_colum = kernel.shape 
    result = 0.0
    
    for fila in range(f_fila):
        for colum in range(f_colum):
            result += fragment[fila,colum] *  kernel[fila,colum]
            
    return result

def convolucion(imagen, kernel):
    
    imagen_fila, imagen_colum = imagen.shape
    kernel_fila, kernel_colum = kernel.shape
   
    img2 = np.zeros(imagen.shape)
   
    for fila in range(imagen_fila):
        for colum in range(imagen_colum):
                img2[fila, colum] = conv(
                                    imagen[fila:fila + kernel_fila, 
                                    colum:colum + kernel_colum],kernel)
             
    plt.imshow(img2, cmap='gray')
    plt.title("{}X{} Kernel".format(kernel_fila, kernel_colum))
    plt.show()
 
    return img2

#main
img = cv2.imread("imagen.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
cv2.imshow('Imagen', gray)
kernel = np.ones((3,3))
resultado = convolucion(gray,kernel)