import numpy as np
import matplotlib.pyplot as plt

random=np.array([1,0.385,0.196,0.141,0.09,0.057,0.054,0.041,0.037,0.035,0.025])
vanilla_grad=np.array([1,0.629,0.466,0.345,0.253,0.202,0.143,0.091,0.053,0.043,0.025])
grad_x_input=np.array([1,0.603,0.432,0.314,0.227,0.177,0.125,0.082,0.053,0.035,0.025])
smooth_grad=np.array([1,0.641,0.482,0.374,0.299,0.229,0.18,0.122,0.079,0.05,0.025])
integrated_gradients=np.array([1,0.585,0.411,0.31,0.233,0.174,0.132,0.088,0.058,0.039,0.025])
guided_backprop=np.array([1,0.574,0.427,0.342,0.263,0.203,0.156,0.113,0.069,0.049,0.025])
deeplift=np.array([1,0.62,0.4,0.32,0.2,0.16,0.14,0.11,0.073,0.044,0.025])

HOC=np.array([1,0.805,0.607,0.434,0.285,0.188,0.121,0.073,0.053,0.039,0.025])
HOC_no_second=np.array([1,0.55,0.409,0.336,0.273,0.222,0.165,0.116,0.068,0.051,0.025])
HOC_no_composite=np.array([1,0.755,0.56,0.419,0.319,0.206,0.155,0.11,0.071,0.066,0.025])

plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [100,90,80,70,60,50,40,30,20,10,0])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [0,10,20,30,40,50,60,70,80,90,100])
# random = [1] * len(random) -  random
# vanilla_grad = [1] * len(random) -  vanilla_grad
# grad_x_input = [1] * len(random) -  grad_x_input
# smooth_grad = [1] * len(random) -  smooth_grad
# guided_backprop = [1] * len(random) -  guided_backprop
# integrated_gradients = [1] * len(random) -  integrated_gradients
# deeplift = [1] * len(random) -  deeplift
# HOC = [1] * len(random) -  HOC
# HOC_no_second = [1] * len(random) -  HOC_no_second
# HOC_no_composite = [1] * len(random) -  HOC_no_composite

# random.reverse()
# vanilla_grad.reverse()
# grad_x_input.reverse()
# smooth_grad.reverse()
# integrated_gradients.reverse()
# guided_backprop.reverse()
# deeplift.reverse()
# HOC.reverse()
# HOC_no_second.reverse()
# HOC_no_composite.reverse()
plt.plot(random.astype(float))
plt.plot(vanilla_grad.astype(float))
plt.plot(grad_x_input.astype(float))
plt.plot(smooth_grad.astype(float))
plt.plot(integrated_gradients.astype(float))
plt.plot(guided_backprop.astype(float))
plt.plot(deeplift.astype(float))
plt.plot(HOC.astype(float))
plt.plot(HOC_no_second.astype(float))
plt.plot(HOC_no_composite.astype(float))
plt.show()