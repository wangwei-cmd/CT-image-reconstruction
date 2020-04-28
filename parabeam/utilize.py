
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq,fftshift
from functools import partial
from scipy.interpolate import interp1d
from skimage.transform import radon,iradon,rotate
import pydicom
import tensorflow as tf
from skimage.measure import compare_psnr
import math
import glob
import os
import matplotlib.pyplot as plt
class CT_uitil:
    def __init__(self,img_size,theta=None,filter="ramp"):
        self.img_size=img_size
        if theta is None:
            theta=np.arange(180)
            self.theta = theta * np.pi / 180.0
        else:
            self.theta = theta * np.pi / 180.0
        self.filter=filter
        self.pad_width, self.diagonal = self.shape_radon()
        self.sin_size = [img_size[0], self.diagonal, len(self.theta), img_size[3]]
        self.fourier_filter=self.get_fourier_filter()
        self.index_w=self.make_cor_rotate()

    def get_fourier_filter(self):
        img_shape=self.sin_size[1]
        size = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
        filter_name=self.filter
        filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
        if filter_name not in filter_types:
            raise ValueError("Unknown filter: %s" % filter)
        n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                            np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2

        fourier_filter = 2 * np.real(fft(f))  # ramp filter
        if filter_name == "ramp":
            pass
        elif filter_name == "shepp-logan":
            # Start from first element to avoid divide by zero
            omega = np.pi * fftfreq(size)[1:]
            fourier_filter[1:] *= tf.sin(omega) / omega
        elif filter_name == "cosine":
            freq = np.linspace(0, np.pi, size, endpoint=False)
            cosine_filter = tf.signal.fftshift(tf.sin(freq))
            fourier_filter *= cosine_filter
        elif filter_name == "hamming":
            fourier_filter *= tf.signal.fftshift(np.hamming(size))
        elif filter_name == "hann":
            fourier_filter *= tf.signal.fftshift(np.hanning(size))
        elif filter_name is None:
            fourier_filter[:] = 1
        fourier_filter=fourier_filter[:, np.newaxis]
        fourier_filter = np.expand_dims(np.transpose(fourier_filter, [1, 0]).astype(np.complex128), 0)
        fourier_filter = np.expand_dims(fourier_filter, 0)
        return fourier_filter

    # @tf.function
    def iradon(self, radon_image, output_size=None,interpolation="linear"):
        shape = self.sin_size
        fourier_filter = self.fourier_filter
        theta=self.theta
        angles_count = len(theta)
        if angles_count != shape[2]:
            raise ValueError("The given ``theta`` does not match the number of "
                             "projections in ``radon_image``.")

        img_shape = shape[1]
        if output_size is None:
            # If output size not specified, estimate from input radon image
            output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))

        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
        pad_width = ((0, 0), (0, projection_size_padded - img_shape), (0, 0), (0, 0))
        img = tf.pad(radon_image, pad_width, mode='constant', constant_values=0)
        # fourier_filter = get_fourier_filter(projection_size_padded, filter)
        projection = tf.signal.fft(tf.cast(tf.transpose(img, [0, 3, 2, 1]), tf.complex128)) * fourier_filter
        radon_filtered = tf.math.real(tf.signal.ifft(projection)[:, :, :, :img_shape])
        #
        radon_filtered = tf.transpose(radon_filtered, [3, 2, 0, 1])
        radon_filtered = tf.cast(radon_filtered, tf.float64)

        # Reconstruct image by interpolation
        reconstructed = tf.zeros((tf.shape(radon_image)[0], output_size, output_size, tf.shape(radon_image)[3]))
        reconstructed = tf.cast(reconstructed,tf.float64)
        radius = output_size // 2
        xpr, ypr = np.mgrid[:output_size, :output_size] - radius
        x = np.arange(img_shape) - img_shape // 2

        thetad = tf.cast(theta, tf.float64)
        # for col, angle in dd:
        for i in range(len(theta)):
            col, angle = radon_filtered[:, i, :, :], thetad[i]
            t = ypr * tf.math.cos(angle) - xpr * tf.math.sin(angle)
            temp = tf.gather(col, tf.cast(tf.math.ceil(t), tf.int32) + img_shape // 2)
            temp1 = tf.gather(col, tf.cast(tf.math.floor(t), tf.int32) + img_shape // 2)
            w = t - tf.math.floor(t)
            w = tf.expand_dims(w, -1)
            w = tf.expand_dims(w, -1)
            w = tf.broadcast_to(w, tf.shape(temp))
            temp2 = w * temp + (1 - w) * temp1
            temp3 = tf.transpose(temp2, [2, 0, 1, 3])
            reconstructed += temp3
        return reconstructed * np.pi / (2 * angles_count)

    # @tf.function
    def radon(self, img):
        input_shape=self.img_size
        # assert tf.constant(input_shape)==tf.shape(img)
        theta=self.theta
        numAngles = len(theta)
        pad_width, diagonal = self.pad_width, self.diagonal
        img=tf.cast(img,tf.float64)
        img1 = tf.pad(img, pad_width, mode='constant', constant_values=0)
        # sinogram = np.zeros((input_shape[0], diagonal, len(theta), input_shape[3]))
        pp=[]
        for n in range(numAngles):
            rotated = self.imrotate(img1, n)
            # sinogram[:, :, n, :] = tf.reduce_sum(rotated, axis=1)
            pp.append(tf.reduce_sum(rotated, axis=1))
        # pp=np.array(pp)
        pp=tf.stack(pp)
        sinogram = tf.transpose(pp,[1,2,0,3])
        return sinogram

    def shape_radon(self):
        # numAngles = len(theta)
        # shape = tf.shape(img)
        # shape1 = tf.cast(shape, tf.float32)
        shape=shape1 = self.img_size
        diagonal = np.sqrt(2) * np.max([shape1[1],shape1[2]])
        pad = [np.ceil(diagonal - shape1[1]), np.ceil(diagonal - shape1[2])]
        # pad = tf.cast(pad, tf.int32)
        pad = np.array(pad).astype(np.int32)
        new_center = [(shape[1] + pad[0]) // 2, (shape[2] + pad[1]) // 2]
        old_center = [shape[1] // 2, shape[2] // 2]
        pad_before = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
        pad_width = [(0, 0), (pad_before[0], pad[0] - pad_before[0]), (pad_before[1], pad[1] - pad_before[1]), (0, 0)]
        # img1 = np.pad(img, pad_width, mode='constant', constant_values=0)
        assert pad[0]+shape[1]==pad[1]+shape[2]
        pad_width = np.array(pad_width).astype(np.int32)
        return pad_width, pad[0] + shape[1]

    def imrotate(self, img, theta_i):
        index11, index12, index21, index22, w11, w12, w21, w22 = self.index_w[theta_i]
        img1 = tf.cast(tf.transpose(img, [1, 2, 3, 0]),tf.float64)
        f11, f12, f21, f22 = tf.gather_nd(img1, index11), tf.gather_nd(img1, index12), tf.gather_nd(img1,
                                                                                                    index21), tf.gather_nd(
            img1, index22)
        bilinear = w11 * tf.transpose(f11, [2, 1, 0]) + w12 * tf.transpose(f12, [2, 1, 0]) + w21 * tf.transpose(f21,
                                                                                                                [2, 1,
                                                                                                                 0]) + w22 * tf.transpose(
            f22, [2, 1, 0])
        rotate = tf.reshape(bilinear,
                            [tf.shape(bilinear)[0], tf.shape(bilinear)[1], tf.shape(img)[1], tf.shape(img)[2]])
        rotate = tf.transpose(rotate, [0, 2, 3, 1])

        return rotate

    def cor_rotate(self, theta_i):
        theta=self.theta[theta_i]
        cos = math.cos(theta)
        sin = math.sin(theta)
        ux=uy=self.diagonal
        semicorx = math.floor(ux / 2)
        semicory = math.floor(uy / 2)
        x = np.arange(ux) - semicorx
        y = np.arange(uy) - semicory
        XY = np.meshgrid(x, y)
        X, Y = XY[0], XY[1]
        sx = (cos * Y - sin * X) + semicorx
        sy = (sin * Y + cos * X) + semicory
        sx = np.reshape(sx, [-1])
        sy = np.reshape(sy, [-1])
        x1 = np.floor(sx)
        x2 = x1+1
        y1 = np.floor(sy)
        y2 = y1+1
        # index = np.stack([sx, sy], 1)
        index11 = np.stack([x1, y1], 1).astype(np.int32)
        index12 = np.stack([x1, y2], 1).astype(np.int32)
        index21 = np.stack([x2, y1], 1).astype(np.int32)
        index22 = np.stack([x2, y2], 1).astype(np.int32)
        w11 = ((x2 - sx) * (y2 - sy)).astype(np.float64)
        w12 = ((x2 - sx) * (sy - y1)).astype(np.float64)
        w21 = ((sx - x1) * (y2 - sy)).astype(np.float64)
        w22 = ((sx - x1) * (sy - y1)).astype(np.float64)
        return index11, index12, index21, index22, w11, w12, w21, w22

    def make_cor_rotate(self):
        cor=[]
        theta=self.theta
        for i in range(len(theta)):
            cor.append(self.cor_rotate(i))
        return cor


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # udir = '/home/wangwei/ct-compare/CT_image/Pancreas-CT/PANCREAS_0002//11-24-2015-PANCREAS0002-Pancreas-23046/Pancreas-63502'
    udir='E:/CT_image/Pancreas-CT/PANCREAS_0002//11-24-2015-PANCREAS0002-Pancreas-23046/Pancreas-63502'
    # udir='/home/wangwei/ct-compare/CPTAC-LUAD/CPTAC-LUAD'
    dd = glob.glob(udir + "/**/*.dcm", recursive=True)
    dd.sort()
    L = len(dd)
    L = min(5000, L)
    f = []
    for i in range(L):
        name = dd[i]
        dc = pydicom.dcmread(name)
        temp = dc.pixel_array
        temp = temp.astype(np.float32)
        divider = np.max(temp) - np.min(temp)
        if divider == 0:
            print('divider being zero: index ', i)
            pass
        temp = (temp - np.min(temp)) / divider
        f.append(temp)
    f = np.array(f)
    f = np.expand_dims(f, -1)
    cen=f.shape[1]//2
    # f=f[0:10,:,:,:]
    batch = 2
    M = N = 256
    LL=M//2
    f = f[0:batch, cen-LL:cen+LL, cen-LL:cen+LL, :]*255
    angles = 180
    theta = np.linspace(0, 180, angles, endpoint=False)

    s = radon(f[0, :, :, 0], circle=False,theta=theta)
    rf = iradon(s, theta)
    pp = compare_psnr(f[0, :, :, 0], rf, np.max(f[0, :, :, 0]))

    shape=f.shape
    ct=CT_uitil([0,shape[1],shape[2],0],theta=theta)
    s1=ct.radon(f)
    ss=tf.expand_dims(tf.expand_dims(s,0),-1)
    rf1=ct.iradon(ss)
    # rf1 = ct.iradon(s1)
    pp1=tf.image.psnr(f[:1,:,:,:],rf1,np.max(f))
    print('f_shap:', f.shape)
    print('s-s1:', np.sum(np.abs(s - s1[0, :, :, 0])))
    print('rf-rf1:', np.sum(np.abs(rf-rf1[0, :, :, 0])))
    print('f-rf:', np.sum(np.abs(f[0, :, :, 0] - rf)))
    print('f-rf1:', np.sum(np.abs(f[0,:,:,0] - rf1[0, :, :, 0])))
    print('psnr beween f and rf:', pp)
    print('psnr beween f1 and rf1:',pp1.numpy())
    print('debug')





