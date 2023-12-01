# %%
from __future__ import annotations
from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
import lzma
import datetime as dt
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from scipy.optimize import curve_fit
import pytz
from matplotlib.pyplot import cm
from skmpython import staticvars, GaussFit
from matplotlib import animation

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import matplotlib
rc('font',**{'family':'serif','serif':['Times New Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
# %%

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3
for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0
# %%
def get_img_pred(index: int):
    if not 0 <= index < 60000:
        raise RuntimeError
    global images, b_i_h, w_i_h, b_h_o, w_h_o
    img = images[index]
    oimg = img.copy().reshape(28, 28)
    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    return (oimg, o.reshape(10))
# %%
def update(i: int):
    global ax, im, tstrs, pstr, imgline, outline
    index = np.random.randint(0, 60000) # int(input("Enter a number (0 - 59999): "))
    img, o = get_img_pred(index)
    pstr.set_text('This is %d'%(np.argmax(o)))
    im.set_data(img.reshape(28, 28))
    imgline.set_ydata(img.flatten())
    outline.set_xdata(o[::-1])
    for idx, val in enumerate(o):
        col = 'k'
        if val == o.max():
            col = 'r'
        tstrs[idx].set_text('%d: %.2f%%, '%(idx, (val * 100)))
        tstrs[idx].set_color(col)
    return [im]
    # plt.show()

def num_article(index: int):
    if not 0 <= index < 10:
        raise RuntimeError('%d invalid'%(index))
    if index == 1 or index == 8:
        return 'an'
    return 'a '
    

# %%
fig, ax = plt.subplots(2, 3, squeeze=True, figsize=(6.8, 4.6), dpi=600, gridspec_kw={'width_ratios': [20, 5, 1], 'height_ratios': [2, 6]})
img, o = get_img_pred(0)
# fig.suptitle(tstr, size=8)
im = ax[1, 0].imshow(img, cmap='Greys')
ax[1, -1].set_ylim(-1, 11)
ax[1, -1].set_xlim(0, 100)
ax[1, -1].axis('off')
ax[1, 0].axis('off')
ax[0, 1].axis('off')
ax[0, 2].axis('off')
imgline, = ax[0, 0].plot(img.flatten(), color='k')
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_xlim(0, 784)
ax[0, 0].set_title('Input vector')
outline, = ax[1, 1].plot(o[::-1], range(9, -1, -1), color='b', marker='o', ls = '-.')
ax[1, 1].set_ylim(0, 9)
ax[1, 1].set_xlim(0, 1)
ax[1, 1].set_ylabel('Output vector')
# ax[1, 0].set_xlabel('This is %d'%(np.argmax(o)))
pstr = ax[1, 0].text(img.shape[0] // 2, img.shape[1] + 2, 'This is %d'%(np.argmax(0)))
tstrs = []
for idx, val in enumerate(o):
    col = 'k'
    if val == o.max():
        col = 'r'
    tstrs.append(ax[1, -1].text(0, 9 - idx, '%d: %.2f%%'%(idx, val * 100), color = col))
anim = animation.FuncAnimation(fig, update, interval=1000, frames=60, blit=True)
anim.save('test2.mp4')
plt.show()

# %%
