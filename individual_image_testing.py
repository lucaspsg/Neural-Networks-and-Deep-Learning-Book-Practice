import pickle
import cv2
import numpy as np

pixels = np.empty((784, 1))

img = cv2.imread("./individual_testing_images/zero.png", 0) 
for i in range (img.shape[0]): 
    for j in range (img.shape[1]): 
        pixels[i*28 + j][0] = (1 - img[i][j]/255) #white background

with open('./net.pkl', 'rb') as inp:
    net = pickle.load(inp)
    test_result = net.feedforward(pixels)
    print(np.argmax(test_result))
