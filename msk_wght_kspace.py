# -*- coding: utf-8 -*-
"""
k-space weighting and masking for denoising of MRI image without blurring or losing contrast, 
as well as for brightening of the objects in the image with simultaneous noise reduction
(for Agilent FID data).

Created on Mon Nov 21 2022
Last modified on Thu Nov 24 2022

@author: Beata Wereszczyńska
"""
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import cv2

def msk_wght_kspace(path, number_of_slices, picked_slice, weight_power, contrast):
    """
    k-space weighting and masking for denoising of MRI image without blurring or losing contrast, 
    as well as for brightening of the objects in the image with simultaneous noise reduction
    (for Agilent FID data).
    Input:
        .fid folder location: path [str],
        total number of slices in the MRI experiment: number_of_slices [int],
        selected slice number: picked_slice [int],
        exponent in the signal weighting equation: weight_power[float]
        restoring contrast: contrast [bool] (1 for denoising, 0 for brightening).
    """
    
    # import k-space data
    echoes = ng.agilent.read(dir=path)[1]
    kspace = echoes[picked_slice - 1 : echoes.shape[0] : number_of_slices, :]  # downsampling to one slice
    del path, echoes, number_of_slices, picked_slice
    
    # k-space weighting
    if contrast:
        kspace_weighted = kspace * np.power(abs(kspace), 3*weight_power)
        # contrasting
        a = kspace_weighted[abs(kspace_weighted) > abs(np.max(kspace_weighted))/6]
        a = a / np.power(abs(a), weight_power)
        kspace_weighted[abs(kspace_weighted) > abs(np.max(kspace_weighted))/6] = a
        title = 'Denoised image'
        del a
    else:
        kspace_weighted = kspace * np.power(abs(kspace), weight_power)
        title = 'Brightened image'
    del contrast, weight_power
    
    # k-space masking
    r = int(kspace.shape[0]/2)
    mask = np.zeros(shape=kspace.shape)
    cv2.circle(img=mask, center=(r,r), radius = r, color =(1,0,0), thickness=-1)
    kspace_weighted = np.multiply(kspace_weighted, mask)
    del mask, r
    
    # normalization
    kspace_weighted = kspace_weighted / (np.max(abs(kspace_weighted)) / np.max(abs(kspace)))
        
    # reconstructing the original image
    ft1 = np.fft.fft2(kspace)                 # 2D FFT
    ft1 = np.fft.fftshift(ft1)                # fixing problem with corner being center of the image
    ft1 = np.transpose(np.flip(ft1, (1,0)))   # matching geometry with VnmrJ-calculated image (still a bit shifted)
    
    # reconstructing denoised image
    ft2 = np.fft.fft2(kspace_weighted)        # 2D FFT
    ft2 = np.fft.fftshift(ft2)                # fixing problem with corner being center of the image
    ft2 = np.transpose(np.flip(ft2, (1,0)))   # matching geometry with VnmrJ-calculated image (still a bit shifted)
    
    # visualization
    plt.rcParams['figure.dpi'] = 600
    plt.subplot(141)
    plt.title('Original k-space', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(kspace), cmap=plt.get_cmap('gray'), vmax=int(np.mean(abs(kspace))*7))
    plt.subplot(142)
    plt.title('Modified k-space', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(kspace_weighted), cmap=plt.get_cmap('gray'), vmax=int(np.mean(abs(kspace))*7))
    plt.subplot(143)
    plt.title('Original image', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(ft1), cmap=plt.get_cmap('gray'))
    plt.subplot(144)
    plt.title(title, fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(ft2), cmap=plt.get_cmap('gray'))
    plt.tight_layout(pad=0, w_pad=0.2, h_pad=0)
    plt.show()
    del title
    
    # return data
    return kspace, kspace_weighted, ft1, ft2


def main():
    path = 'mems_20190406_02.fid'     # .fid folder location [str]
    number_of_slices = 384            # total number of slices in the imaging experiment [int]
    picked_slice = 119                # selected slice number [int]
    weight_power = 0.09               # exponent in the signal weighting equation [float]
    contrast = 1                      # restoring contrast [bool]

    # running calculations and retrieving the results
    k, kw, ft1, ft2 = msk_wght_kspace(path, number_of_slices, picked_slice, weight_power, contrast)
    
    # creating global variables to be available after the run completion
    global MRI_k
    MRI_k = k
    global MRI_kw
    MRI_kw = kw
    global MRI_ft1
    MRI_ft1 = ft1
    global MRI_ft2
    MRI_ft2 = ft2


if __name__ == "__main__":
    main()
