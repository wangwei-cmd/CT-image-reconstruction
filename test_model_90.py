import red_cnn as net0
import DD_Net_tf2 as net1
import fbpconv as net2
import Our_model_1 as net3
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Aangles=180
AT = np.load('AT_' + str(Aangles) + '_512x512' + '.npz')
val = AT['name1'].astype('float32')
index = AT['name2']
shape = AT['name3']
AT = tf.sparse.SparseTensor(index, val, shape)
# AT = tf.cast(AT, tf.float32)
theta=np.linspace(0, 180, Aangles, endpoint=False)

pre='./512x512/weights'
ckpt=[pre+'/red_cnn',pre+'/DD_NET',pre+'/fbpconv',
      pre+'/our_model_1_lambda=0.5']
j=2
if j==0:
    batch=4
else:
    batch=5
Model=[net0.make_model(batch),net1.make_model(batch),net2.make_model(batch),
       net3.make_model_3(AT,(725, 180),(512, 512))]

udir = "./test/"

angles=60
superangles=180
vx=np.load(udir+'ini,angle='+str(angles)+'_no_scale__0.5.npy')
# f=np.load(udir+ '/f,angle=' + str(superangles) + '_no_scale__0.5.npy')
f_noisy_img = np.load(udir + '/f_noisy,angle=' + str(angles) + '_no_scale__0.5.npy')
f_noisy_img=interp(f_noisy_img, np.linspace(0, 180, angles, endpoint=False), np.linspace(0, 180, superangles, endpoint=False))
vy=np.load(udir+ 'u_CT_img_no_scale.npy')
# M = np.max(np.max(vx, 1), 1)
# M = np.reshape(M, [np.shape(M)[0], 1, 1, 1])
L=500
vx=vx[0:L]
# f=f[0:L]
f_noisy_img=f_noisy_img[0:L]
vy=vy[0:L]
shape=vy.shape
vx=tf.cast(vx,tf.float32)
# f=tf.cast(f,tf.float32)
f_noisy=tf.cast(f_noisy_img,tf.float32)
vy=tf.cast(vy,tf.float32)
def inimodel(j,f_noisy,vx,Model):
       if j in[3,4]:
              _ = Model[j](f_noisy[0:1])
              Model[j].load_weights(ckpt[j])
       else:
              _ = Model[j](vx[0:1])
              Model[j].load_weights(ckpt[j])
       print('load model completely')
def evaluate(j,f_noisy,vx,batch,shape,Model):
       _=inimodel(j, f_noisy, vx, Model)
       prediction = np.zeros(shape)
       iter=list(range(0,shape[0],batch))
       for i in range(len(iter)):
              if j in [3]:
                     # _ = Model[j](f_noisy[0:1])
                     # Model[j].load_weights(ckpt[j])
                     M = np.max(np.max(vx[iter[i]:iter[i] + batch], 1), 1)
                     M = np.reshape(M, [np.shape(M)[0], 1, 1, 1])
                     prediction[iter[i]:iter[i] + batch] = Model[j](f_noisy_img[iter[i]:iter[i] + batch] / M * 255)[1].numpy()
                     prediction[iter[i]:iter[i]+batch]= prediction[iter[i]:iter[i] + batch] *M/255
              else:
                     M = np.max(np.max(vx[iter[i]:iter[i] + batch], 1), 1)
                     M = np.reshape(M, [np.shape(M)[0], 1, 1, 1])
                     prediction[iter[i]:iter[i] + batch] = Model[j](vx[iter[i]:iter[i] + batch]/ M * 255).numpy()
                     prediction[iter[i]:iter[i] + batch] = prediction[iter[i]:iter[i] + batch] * M / 255
              print(i)
       return prediction

def saveimages(prediction,vx,vy,j):
    pre='./512x512/pic/'
    name=[pre+'red_cnn/',pre+'DD_NET/',pre+'FBP_conv/',pre+'our/']
    L=len(prediction)
    for i in range(L):
        cv.imwrite(name[j]+str(i)+'.png', prediction[i, :, :, 0]/np.max(prediction[i, :, :, 0])*255)
        # cv.imwrite('./512x512/pic/noisy/'+str(i)+'.png',vx[i,:,:,0].numpy()/np.max(vx[i,:,:,0].numpy())*255)
        # cv.imwrite('./512x512/pic/label/'+str(i)+'.png',vy[i,:,:,0].numpy()/np.max(vy[i,:,:,0].numpy())*255)



def savepsnr(pp,qq,j):
    pre = './512x512/psnr/'
    name = [pre + 'red_cnn', pre + 'DD_NET', pre + 'FBP_conv', pre + 'our',pre+'our']
    np.savez(name[j],psnr=pp,ssim=qq)

prediction=evaluate(j,f_noisy,vx,batch,shape,Model)
ii=np.random.randint(0,L)
print('show figure:',ii)
plt.imshow(prediction[ii,:,:,0],cmap='gray')
plt.figure()
plt.imshow(vy[ii,:,:,0],cmap='gray')
plt.figure()
plt.imshow(vx[ii,:,:,0],cmap='gray')
plt.show()
pp=tf.image.psnr(tf.cast(prediction,tf.float32),vy,tf.reduce_max(prediction)).numpy()
pp1=tf.image.psnr(vx,vy,tf.reduce_max(vx)).numpy()
qq=tf.image.ssim(tf.cast(prediction,tf.float32),vy,tf.reduce_max(prediction)).numpy()
qq1=tf.image.ssim(vx,vy,tf.reduce_max(vy)).numpy()
print('denoise psnr:',tf.reduce_mean(pp).numpy())
print('original psnr:',tf.reduce_mean(pp1).numpy())
print('denoise ssim:',tf.reduce_mean(qq).numpy())
print('original ssim:',tf.reduce_mean(qq1).numpy())
saveimages(prediction,vx,vy,j)
savepsnr(pp,qq,j)