import newmodel as net4
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2 as cv

def interp(f,xp,x):
    # f_img=f[:,:,::2,:]
    f_img=f
    shape=f.shape
    L=len(x)
    f_interp = np.zeros(shape=[shape[0],shape[1],L,shape[3]])
    idL = np.where(x <= xp[0])[0]
    idR = np.where(x >= xp[-1])[0]
    xx = x[idL[-1] + 1:idR[0]]
    id = np.searchsorted(xp, xx)
    L = xx - xp[id - 1]
    R = xp[id] - xx
    w1 = R / (L + R)
    w2 = 1 - w1
    val1 = f_img[:, :, id - 1, :]
    val2 = f_img[:, :, id, :]
    val1 = val1.transpose([0, 1, 3, 2])
    val2 = val2.transpose([0, 1, 3, 2])
    temp = val1 * w1 + val2 * w2
    f_interp[:, :, idL[-1] + 1:idR[0], :] = temp.transpose([0, 1, 3, 2])
    for i in idL:
        f_interp[:, :, i, :] = f_img[:, :, 0, :]
    for j in idR:
        f_interp[:, :, j, :] = f_img[:, :, -1, :]
    return f_interp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
Aangles=180
AT = np.load('AT_' + str(Aangles) + '_512x512' + '.npz')
val = AT['name1'].astype('float32')
index = AT['name2']
shape = AT['name3']
AT = tf.sparse.SparseTensor(index, val, shape)
# AT = tf.cast(AT, tf.float32)
theta=np.linspace(0, 180, Aangles, endpoint=False)
ckpt='./512x512/weights'+'/new_model_lambda=0.5'
batch=5
Model=net4.make_model_3(AT,(725, 180),(512, 512))
udir = "./test/"
angles=60
superangles=180
f_noisy_img = np.load(udir + '/f_noisy,angle=' + str(angles) + '_no_scale__0.5.npy')
f_noisy_img=interp(f_noisy_img, np.linspace(0, 180, angles, endpoint=False), np.linspace(0, 180, superangles, endpoint=False))
L=500
f_noisy_img=f_noisy_img[0:L]
f_noisy=tf.cast(f_noisy_img,tf.float32)
def inimodel( f_noisy, Model,ckpt=ckpt):
        _ = Model(f_noisy[0:1])
        Model.load_weights(ckpt)


def evaluate(f_noisy, batch, Model):
    _ = inimodel(f_noisy, Model)
    prediction = np.zeros([tf.shape(f_noisy).numpy()[0],512,512,1])
    iter = list(range(0, tf.shape(f_noisy).numpy()[0], batch))
    for i in range(len(iter)):
        prediction[iter[i]:iter[i] + batch] = Model(f_noisy_img[iter[i]:iter[i] + batch])[
            1].numpy()
        print(i)
    return prediction

prediction=evaluate(f_noisy,batch,Model)
ii=np.random.randint(0,L)
print('show figure:',ii)
plt.imshow(f_noisy[ii,:,:,0],cmap='gray')
plt.figure()
plt.imshow(prediction[ii,:,:,0],cmap='gray')
plt.show()

# vy=np.load(udir+ 'u_CT_img_no_scale.npy')
# vy=vy[0:L]
# vy=tf.cast(vy,tf.float32)
# pp=tf.image.psnr(tf.cast(prediction,tf.float32),vy,tf.reduce_max(prediction)).numpy()
# qq=tf.image.ssim(tf.cast(prediction,tf.float32),vy,tf.reduce_max(prediction)).numpy()
# print('average psnr:',tf.reduce_mean(pp).numpy())
# print('average ssim:',tf.reduce_mean(qq).numpy())