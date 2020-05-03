import numpy as np
import matplotlib.pyplot as plt

number_pixels = 1000
number_points = 100

edn = 2                                 # electron gain
RN_e = 3                                # read noise electrons
RN = RN_e/edn                           # read noise DN
FW_e = 10**5                            # full well in electrons
FW = FW_e/edn                           # full well in DN
SCALE = number_points / np.log10(FW)    # scale
PN = 0.02                               # fixed patter noise
T = 300                                 # temperature
k1 = 8.62*10**-5                        # Boltzman const
DFM = 0.5                               # Thermal figure of merit
DN = 0.30                               # Dark FPN
PA = (8*10**-4)**2                     # Pixel area
t = 0.3                                # intergration time
Eg = 1.1557-(7.021 * 10**-4 * T**2)/(1108 + T)
DARK_e = t * 2.55*10**15 * PA * DFM * T**1.5 * np.exp(-Eg/(2 * k1 * T))
DARK = DARK_e / edn




LUX = np.zeros((number_points))


C = np.random.normal(0, 1, (number_pixels))              # Random number FPN
F = np.random.normal(0, 1, (number_pixels))              # Random Number Dark FPN

SIG1 = np.zeros((number_pixels, number_points))
SIG2 = np.zeros((number_pixels, number_points))
SIG3 = np.zeros((number_pixels, number_points))
SIG4 = np.zeros((number_pixels, number_points))
SIG5 = np.zeros((number_pixels, number_points))
for frame_number in range(number_points):
    signal = 10**((frame_number+1)/SCALE)
    A = np.random.normal(0, 1, (number_pixels, number_points))
    B = np.random.normal(0, 1, (number_pixels, number_points))
    D = np.random.normal(0, 1, (number_pixels, number_points))

    read = RN * A[:, frame_number]
    shot = (signal / edn)**0.5 * B[:, frame_number]
    FPN = signal * PN * C
    Dshot = (DARK / edn)**0.5 * D[:, frame_number]
    DFPN = DARK * DN * F

    SIG1[:, frame_number] = signal + read + shot + FPN + Dshot + DFPN
    SIG2[:, frame_number] = signal + read + shot + FPN + Dshot
    SIG3[:, frame_number] = signal + read + shot + Dshot
    SIG4[:, frame_number] = signal + read + shot + FPN
    SIG5[:, frame_number] = signal + read + shot

SIGNAL = np.mean(SIG1, axis=0)
NOISE1 = np.std(SIG1, axis=0)
NOISE2 = np.std(SIG2, axis=0)
NOISE3 = np.std(SIG3, axis=0)
NOISE4 = np.std(SIG4, axis=0)
NOISE5 = np.std(SIG5, axis=0)

plt.loglog(SIGNAL, NOISE4)
plt.show()
