import numpy as np
import viz
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P

saliency = np.zeros((250,250))
for i in range(25):
    for j in range(25):
        saliency[i][j] = 0.1
for i in range(25):
    for j in range(25):
        saliency[25+i][j] = 0.2
for i in range(25):
    for j in range(25):
        saliency[50+i][j] = 0.3
for i in range(25):
    for j in range(25):
        saliency[75+i][j] = 0.4
for i in range(25):
    for j in range(25):
        saliency[100+i][j] = 0.5
for i in range(25):
    for j in range(25):
        saliency[125+i][j] = 0.6
for i in range(25):
    for j in range(25):
        saliency[150+i][j] = 0.7
for i in range(25):
    for j in range(25):
        saliency[175+i][j] = 0.8
for i in range(25):
    for j in range(25):
        saliency[200+i][j] = 0.9
for i in range(25):
    for j in range(25):
        saliency[i+225][j] = 1.0
plt.figure(figsize=(25, 15))
plt.subplot(3, 5, 1)
plt.axis('off')
plt.imshow(saliency, cmap=P.cm.gray, vmin=0, vmax=1)
plt.savefig('output/color_palette.png')