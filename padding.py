"""
por Abhisek Jana
codigo tomado de https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
codigo main modificado por Vanessa Méndez
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
 
 
def convolucion_padding(imagen, kernel, average=False, verbose=False):
    if len(imagen.shape) == 3:
        print("Found 3 Channels: {}".format(imagen.shape))
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size: {}".format(imagen.shape))
    else:
        print("Image Shape: {}".format(imagen.shape))
 
    print("Kernel Shape: {}".format(kernel.shape))
 
    if verbose:
        plt.imshow(imagen, cmap='gray')
        plt.title("Imagen")
        plt.show()
 
    imagen_fila, imagen_colum = imagen.shape
    kernel_fila, kernel_colum = kernel.shape
 
    img2 = np.zeros(imagen.shape)
 
    pad_altura = int((kernel_fila - 1) / 2)
    pad_ancho = int((kernel_colum - 1) / 2)
 
    imagen_padding = np.zeros((imagen_fila + (2 * pad_altura), imagen_colum + (2 * pad_ancho)))
 
    imagen_padding[pad_altura:imagen_padding.shape[0] - pad_altura, pad_ancho:imagen_padding.shape[1] - pad_ancho] = imagen
 
    if verbose:
        plt.imshow(imagen_padding, cmap='gray')
        plt.title("Padding")
        plt.show()
 
    for fila in range(imagen_fila):
        for colum in range(imagen_colum):
            img2[fila, colum] = np.sum(kernel * imagen_padding[fila:fila + kernel_fila, colum:colum + kernel_colum])
            if average:
                img2[fila, colum] /= kernel.shape[0] * kernel.shape[1]
 
    print("Output Image Size: {}".format(img2.shape))
 
    if verbose:
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
resultado = convolucion_padding(gray,kernel)