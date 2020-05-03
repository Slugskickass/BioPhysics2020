import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import microscPSF as msPSF

mp = msPSF.m_params
pz = np.arange(0, 2.01, 0.1)
dxy = 0.05
xy_size = 50

stack = msPSF.gLXYZParticleScan(mp, dxy, xy_size, pz)
images = []
for I in range(np.shape(stack)[0]):
    images.append(Image.fromarray(stack[I, :, :]))

images[0].save('PSF.tiff', save_all=True, append_images=images[1:])
