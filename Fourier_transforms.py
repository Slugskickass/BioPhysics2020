import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from random import randint
from numpy import unravel_index

def load_an_image(file_name, frame_number):
    im = Image.open(file_name)
    if frame_number >= im.n_frames:
        print('Requested frame number too high')
        imarray = 0
    else:
        im.seek(frame_number)
        imarray = np.array(im)
    return(imarray)

def generate_PSF(NA, pixel_size, wavelength, image_sizex, image_sizey):
    fwhm = ((1/2.355) * wavelength/(2*NA))/ pixel_size # https://mathworld.wolfram.com/GaussianFunction.html
    xo = image_sizex/2
    yo = image_sizey/2
    x = np.linspace(0, image_sizex, image_sizex + 1)
    y = np.linspace(0, image_sizey, image_sizey + 1)
    X, Y = np.meshgrid(x,y)
    gaussian = np.exp( -1 * ((((xo - X)**2)/fwhm**2) + ((yo - Y)**2)/fwhm**2))
    return(gaussian)

def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def random_generate(sizex, sizey, number):
    frame = np.zeros((sizex+1, sizey+1))
    for pos in range(number):
        frame[randint(0, sizex), randint(0, sizey)] = 1
    return(frame)

def get_max_position(frame):
    position = np.argmax(frame)
    X, Y = unravel_index(position, np.shape(frame))
    return(X, Y)

if __name__ == '__main__':
# The first thing we are going to do is load in an image there are a number
# of images in each file, remember that python starts at 0
# Each pixel is 51 nm
    file_name = '/Users/Ashley/PycharmProjects/BioPhysics2020/Data/Stack.tif'
    data = load_an_image(file_name, 0)
    frame_shape = np.shape(data)
    plt.subplot(1, 2, 1)
    plt.imshow(data)


# We are now going to take a fourier transform of the data
    data_f = np.fft.fft2(data)
# The fft is shiffted in to the centre of the image, this is more for
# us than the computer you can remove the line below if you want to
    data_f = np.fft.ifftshift(data_f)
    plt.subplot(1, 2, 2)
    plot_spectrum(data_f)
    plt.show()

# Lets generate a point spread function and look at the OTF
    NA = 1.49  # The NA of the lens this should be between 1.49 to 0.1
    pixel_size = 51
    wavelength = 680
    image_sizex = frame_shape[0]
    image_sizey = frame_shape[1]
# Here we generate the PSF for this I am just using a gaussian.
    PSF = generate_PSF(NA, pixel_size, wavelength, image_sizex, image_sizey)
    plt.subplot(2, 2, 1)
    plt.title('PSF')
    plt.imshow(PSF)
# A line plot
    plt.subplot(2, 2, 3)
    X, Y = get_max_position(PSF)
    plt.plot(PSF[:, Y])

# Now we generate the OFT from this PSF
# The OFT is a FFT of the PSF
    OTF = np.fft.fft2(PSF)
    OTF = np.fft.ifftshift(OTF)
    plt.subplot(2, 2, 2)
    plot_spectrum(OTF)
    plt.title('OFT')
    plt.subplot(2, 2, 4)
# Plot a cross section through the OTF
    X, Y = get_max_position(np.real(OTF))
    plt.plot(np.abs(OTF[:, Y]))
    plt.show()

# Now lets use the OTF
# first lets generate a random array of molecules.
    number_of_molecules = 1000
    frame = random_generate(frame_shape[1], frame_shape[0], number_of_molecules)
    plt.subplot(1, 2, 1)
# Show the single molecules
    plt.imshow(frame)
# Take the FFT of the image, we now have it in frequency space
    frame_f = np.fft.ifftshift(np.fft.fft2(frame))
# Multiply the image by the OTF, this acts as a convolution, or a frequency based filter
    frame_f_OTF = frame_f * (OTF) # you can replace (OFT) with (OTF > (np.max(OTF)* .8)) if you want a sharper filter
    blured_f = np.fft.ifft2(np.fft.ifftshift(frame_f_OTF))
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(blured_f))
    plt.show()
