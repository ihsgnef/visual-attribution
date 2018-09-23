import numpy as np
import viz
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P

saliency = np.zeros((100,1100))
for i in range(100):
    for j in range(100):
        saliency[i][j] = 0.0
for i in range(100):
    for j in range(100):
        saliency[i][100+j] = 0.1
for i in range(100):
    for j in range(100):
        saliency[i][200+j] = 0.2
for i in range(100):
    for j in range(100):
        saliency[i][300+j] = 0.3
for i in range(100):
    for j in range(100):
        saliency[i][400+j] = 0.4
for i in range(100):
    for j in range(100):
        saliency[i][500+j] = 0.5
for i in range(100):
    for j in range(100):
        saliency[i][600+j] = 0.6
for i in range(100):
    for j in range(100):
        saliency[i][700+j] = 0.7
for i in range(100):
    for j in range(100):
        saliency[i][800+j] = 0.8
for i in range(100):
    for j in range(100):
        saliency[i][j+900] = 0.9
for i in range(100):
    for j in range(100):
        saliency[i][j+1000] = 1.0

#plt.figure(figsize=(1000, 1000))
x = np.array([0,100,200,300,400,500,600,700,800,900,1000])
my_xticks = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
plt.xticks(x, my_xticks)
#plt.xticks(saliency, [])
#plt.axes.get_xaxis().set_ticks([])
plt.imshow(saliency, cmap=P.cm.gray, vmin=0, vmax=1)
plt.savefig('output/color_palette.png')