from matplotlib import pyplot as plt
import numpy as np
from skimage import io
import scipy as sc
import cv2
import math





def generateOmega(size, key=True):
    if key:
        np.random.seed(0)
    Omega = np.random.normal(size=size)
    return Omega


def alg(C, Omega, alpha, new_path):
    f = sc.fft.fft2(C)
    f_re = np.real(f)
    f_re0 = f_re.copy()
    f_im = np.imag(f)
    f_re[:128, :128] = (f_re[:128, :128].flatten() + alpha * Omega[:16384]).reshape(128, 128)
    f_re[:128, 384:] = (f_re[:128, 384:].flatten() + alpha * Omega[16384:32768]).reshape(128, 128)
    f_re[384:, :128] = (f_re[384:, :128].flatten() + alpha * Omega[32768:49152]).reshape(128, 128)
    f_re[384:, 384:] = (f_re[384:, 384:].flatten() + alpha * Omega[49152:]).reshape(128, 128)

    fW = f_re + 1j * f_im
    CW = np.real(sc.fft.ifft2(fW)).astype(np.uint8)
    io.imsave(new_path, CW)

    new_CW = io.imread(new_path)
    new_f = sc.fft.fft2(new_CW)
    new_f_re = np.real(new_f)

    new_Omega = np.zeros(65536)
    new_Omega[:16384] = (new_f_re[:128, :128].flatten() - f_re0[:128, :128].flatten()) / alpha
    new_Omega[16384:32768] = (new_f_re[:128, 384:].flatten() - f_re0[:128, 384:].flatten()) / alpha
    new_Omega[32768:49152] = (new_f_re[384:, :128].flatten() - f_re0[384:, :128].flatten()) / alpha
    new_Omega[49152:] = (new_f_re[384:, 384:].flatten() - f_re0[384:, 384:].flatten()) / alpha

    p = (np.sum(Omega * new_Omega)) / (np.sqrt(np.sum(np.power(Omega, 2))) * np.sqrt(np.sum(np.power(new_Omega, 2))))
    PSNR = cv2.PSNR(C, new_CW)
    wsnr = WSNR(C, new_CW)
    return p, PSNR, wsnr


if __name__ == '__main__':
    path = "bridge.tif"
    new_path = "new_bridge1.tif"
    C = io.imread(path)
    size = int(C.shape[0] * C.shape[1]/4)
    Omega = generateOmega(size)
    alpha_list = list()
    p_list = list()
    PSNR_list = list()
    wsnr_list = list()
    alpha = 1000
    alpha_plus = 500
    while alpha < 40000:
        p, PSNR, wsnr = alg(C, Omega, alpha, new_path)
        p_list.append(round(p, 4))
        PSNR_list.append(round(PSNR, 4))
        alpha_list.append(alpha)
        wsnr_list.append(round(wsnr/8, 4))
        alpha += alpha_plus

    max_p = max(p_list)
    max_ind = p_list.index(max_p)
    alpha_best = alpha_list[max_ind]

    print('alpha = ', alpha_best)
    print('p = ', max_p)
    print('PSNRP = ', PSNR_list[max_ind])
    print('WSNRP = ', wsnr_list[max_ind])

    fig1 = plt.figure()
    plt.plot(alpha_list, p_list)
    plt.xlabel("alpha")
    plt.ylabel("p")
    #plt.show()

    #Ложное обнаружение
    p__list = list()
    x_list = list()
    for i in range(100):
        O = generateOmega(size, False)
        p_, _, _ = alg(C, O, alpha_best, new_path)
        p__list.append(p_)
        x_list.append(i)

    fig2 = plt.figure()
    plt.plot(x_list, p__list)
    #plt.show()











