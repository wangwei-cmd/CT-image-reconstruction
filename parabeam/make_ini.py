import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import radon,iradon
from make_sin_noise import add_sin_noise
from utilize import CT_uitil
import os
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt


def make_ini(u_img,angles,udir):
    # u_img = readimage(dir, ux, uy)
    # np.save(udir + '/npy/' + '/u_CT_img_'+str(np.max(u_img)), u_img)
    print('shape of u_img:', u_img.shape)
    print('maximum of u_img:', np.max(u_img))
    # np.save(udir + '/npy/' + '/u_CT_img_no_scale', u_img)
    theta = np.linspace(0, 180, angles, endpoint=False)
    ini_u_img = np.zeros(u_img.shape)
    temp = radon(u_img[0, :, :, 0], theta=theta, circle=False)
    shape = temp.shape
    ct_sin_img = np.zeros([u_img.shape[0], shape[0], shape[1], u_img.shape[3]])
    ct_sin_img_noisy = np.zeros([u_img.shape[0], shape[0], shape[1], u_img.shape[3]])

    var=0.5
    inter=54
    iter=list(range(0,u_img.shape[0],inter))
    ct = CT_uitil(u_img.shape, theta)
    for i in range(len(iter)):
        ct_sin_img[iter[i]:iter[i]+inter] = ct.radon(u_img[iter[i]:iter[i]+inter]).numpy()
        # ct_sin_img_noisy[iter[i]:iter[i]+inter] = add_sin_noise(ct_sin_img[iter[i]:iter[i]+inter],var=var)
        ct_sin_img_noisy[iter[i]:iter[i]+inter] = ct_sin_img[iter[i]:iter[i]+inter]  #%no noise
        ini_u_img[iter[i]:iter[i]+inter] = ct.iradon(ct_sin_img_noisy[iter[i]:iter[i]+inter]).numpy()
        print(i)


    # np.save(udir + '/npy//512x512/' + '/ini,angle=' + str(angles) + '_no_scale_' + '_' + str(var),
    #         ini_u_img)
    # np.save(udir + '/npy//512x512/' + '/f,angle=' + str(angles) + '_no_scale_' + '_' + str(var),
    #         ct_sin_img)
    # np.save(udir + '/npy//512x512/' + '/f_noisy,angle=' + str(angles) + '_no_scale_' + '_' + str(var),
    #         ct_sin_img_noisy)

    np.save(udir + '/ini,angle=' + str(angles) + '_no_scale_' + '_' + str(var),
            ini_u_img)
    np.save(udir  + '/f,angle=' + str(angles) + '_no_scale_' + '_' + str(var),
            ct_sin_img)
    np.save(udir + '/f_noisy,angle=' + str(angles) + '_no_scale_' + '_' + str(var),
            ct_sin_img_noisy)

    print('save_complete')
    print('min of ct_sin_img_noisy:', np.min(ct_sin_img_noisy))
    psnr=np.zeros([1,u_img.shape[0]])
    psnr1=np.zeros([1,u_img.shape[0]])
    for i in range( u_img.shape[0]):
        psnr[0,i]=compare_psnr(u_img[i],ini_u_img[i],np.max(u_img[i]))
        # psnr1[0, i] = compare_psnr(ct_sin_img[i], ct_sin_img_noisy[i], np.max(ct_sin_img[i]))
    print('psnr:',psnr)
    print('psnr1:', psnr1)

    # print(tf.image.psnr(u_img, ini_u_img, np.max(u_img)).numpy())
    # print(tf.image.psnr(ct_sin_img, ct_sin_img_noisy, np.max(ct_sin_img)).numpy())

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # set='train'
    set = 'test'
    udir = 'E:\CT_image\AMP\sparse_angles\\'+set+'/npy'
    ux, uy = 512, 512
    # u_img=CT.make_CT(udir,ux,uy)
    # u_img=np.load(udir+'/npy/'+'u_CT_img_test_no_scale.npy')
    u_img = np.load(udir + '/u_CT_img_no_scale.npy')
    if set=='train':
        L=1000
    if set=='test':
        L=500
    L = np.minimum(L, len(u_img))
    u_img=u_img[0:L]
    np.save(udir + '//u_CT_img_no_scale', u_img)
    # print('shape of u_img:', u_img.shape)
    angles = 60
    make_ini(u_img, angles,udir)

