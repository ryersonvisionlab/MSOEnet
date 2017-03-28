import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


x = np.random.uniform(0,2*np.pi,10000)
hist, bin_edges = np.histogram(x, bins=50, range=(0, 2*np.pi))
unity_hist = hist / float(hist.sum())
widths = bin_edges[:-1] - bin_edges[1:]
ax = plt.subplot(111, polar=True)
cm = plt.cm.get_cmap('hsv')
bars = plt.bar(bin_edges[:-1], unity_hist, width=widths)
for bar in bars:
    color = cm(bar._x/(2*np.pi))
    bar.set_facecolor(color)
plt.show()
