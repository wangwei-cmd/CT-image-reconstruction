import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from skimage.transform import radon,iradon,rotate
from scipy.fftpack import fft, ifft, fftfreq, fftshift
class sinLayer(tf.keras.Model):
    def __init__(self, AT,s_shape = (725, 360),out_size=(512,512)):
        super(sinLayer, self).__init__()
        self.AT = AT
        self.s_shape=s_shape
        self.out_size = out_size
        w_b=w_bfunction(np.pi,np.linspace(-np.floor(s_shape[0]/2),s_shape[0]-np.floor(s_shape[0]/2)-1,s_shape[0]))
        self.w_b =w_b.astype('float32')
        self.sinLayer=[]
        self.sinLayer_1 = []

        self.ctLayer=[]
        self.ctLayer_1 = []
        ###sinLayer###
        self.sinLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='sinconv1', activation=tf.nn.relu))
        self.M1=4
        for layers in range(1,self.M1+1):
            self.sinLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='sinconv%d' % layers, use_bias=False))
            self.sinLayer.append(tf.keras.layers.BatchNormalization())
            self.sinLayer.append(tf.keras.layers.ReLU())
        self.sinLayer.append(tf.keras.layers.Conv2D(1, 5, name='sinconv6',padding='same'))
        ###CTLayer###
        self.ctLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='ctconv1', activation=tf.nn.relu))
        self.M2=5
        for layers in range(1,self.M2+1):
            self.ctLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='ctconv%d' % layers, use_bias=False))
            self.ctLayer.append(tf.keras.layers.BatchNormalization())
            self.ctLayer.append(tf.keras.layers.ReLU())
        self.ctLayer.append(tf.keras.layers.Conv2D(1, 5, name='ctconv6', padding='same'))

    def decode(self, sin_fan):
        # AT, alpha, h, w_c=self.AT,self.alpha,self.h,self.w_c
        AT, w_b = self.AT, self.w_b
        sin_fan = tf.transpose(sin_fan, perm=[0, 2, 1, 3])
        # cos_alpha = tf.math.cos(alpha)
        s_fan_shape = sin_fan.shape
        batch = tf.shape(sin_fan)[0]
        sin_fan1 = tf.reshape(sin_fan, [-1, s_fan_shape[2], 1])
        filter_s_fan = tf.nn.conv1d(sin_fan1, tf.expand_dims(tf.expand_dims(w_b, -1), -1), stride=1, padding='SAME')
        # filter_s_fan1=tf.reshape(filter_s_fan,s_fan_shape)
        filter_s_fan2 = tf.reshape(filter_s_fan, [batch, -1])
        filter_s_fan2 = tf.transpose(filter_s_fan2)
        rf = tf.sparse.sparse_dense_matmul(AT, filter_s_fan2)
        rf = tf.transpose(rf)
        rf = tf.reshape(rf, [batch, 512, 512, 1])
        return 4 * rf

    # @tf.function
    def call(self, inputs):
        de_sin = self.sinLayer[0](inputs)
        pp = de_sin
        for i in range(1, self.M1 + 1):
            for j in range(0, 3):
                de_sin = self.sinLayer[3 * i + j - 2](de_sin)
            pp = de_sin + pp
        de_sin = self.sinLayer[3 * self.M1 + 1](pp/self.M1) + inputs

        fbp = self.decode(de_sin)

        outputs = self.ctLayer[0](fbp)
        qq = outputs
        for i in range(1, self.M2 + 1):
            for j in range(0, 3):
                outputs = self.ctLayer[3 * i + j - 2](outputs)
            qq = qq + outputs
        outputs = self.ctLayer[3 * self.M2 + 1](qq/self.M2) + fbp
        return [de_sin, outputs, fbp]

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

def u_function(s):
    u=np.zeros(s.shape)
    index_1=np.where(s==0)[0]
    u[index_1]=1/2
    index=np.where(s!=0)[0]
    v=s[index]
    u[index]=(np.cos(v)-1)/(v**2)+np.sin(v)/v
    return u
def w_bfunction(b,s):
    return u_function(b*s)*(b**2)/(4*np.pi**2)

def decode(sin_fan,AT,w_b):
        # AT, alpha, h, w_c=self.AT,self.alpha,self.h,self.w_c
        sin_fan=tf.transpose(sin_fan,perm=[0,2,1,3])
        # cos_alpha = tf.math.cos(alpha)
        s_fan_shape =sin_fan.shape
        batch=tf.shape(sin_fan)[0]
        sin_fan1 = tf.reshape(sin_fan, [-1, s_fan_shape[2], 1])
        filter_s_fan = tf.nn.conv1d(sin_fan1, tf.expand_dims(tf.expand_dims(w_b,-1),-1), stride=1, padding='SAME')
        # filter_s_fan1=tf.reshape(filter_s_fan,s_fan_shape)
        filter_s_fan2 = tf.reshape(filter_s_fan, [batch, -1])
        filter_s_fan2 = tf.transpose(filter_s_fan2)
        rf = tf.sparse.sparse_dense_matmul(AT, filter_s_fan2)
        rf = tf.transpose(rf)
        rf = tf.reshape(rf, [batch, 512, 512, 1])
        return 4*rf




def make_model_3(AT,s_shape=(725, 360),out_size=(512,512)):
    CT=sinLayer(AT,s_shape,out_size)
    inputs = tf.keras.Input(shape=(s_shape[0],s_shape[1],1))
    [de_sin, outputs, fbp]=CT(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[de_sin, outputs, fbp])
    return model


def train(epoch, udir,batch, theta, iternum, restore=0, ckpt='./weights/CT_tf2_4'):
    angles = np.shape(theta)[0]

    s_shape = (725, 180)
    out_size = (512, 512)
    AT = np.load('AT_' + str(180) + '_512x512' + '.npz')

    data = np.load('train' + '_fan2para.npz')
    u_img = data['u'].astype('float32')
    f_noisy_img = data['f_noisy'].astype('float32')
    f_img = data['f'].astype('float32')
    # u_ini = data['ini_u'].astype('float32')
    # M = np.max(np.max(u_ini, 1), 1)
    # M=np.reshape(M, [np.shape(M)[0],1,1,1])
    # u_img=u_img/M*255
    # f_noisy_img=f_noisy_img/M*255
    # f_img=f_img/M*255
    # u_ini=u_ini/M*255

    val = AT['name1'].astype('float32')
    index = AT['name2']
    shape = AT['name3']
    # del u_ini

    AT = tf.sparse.SparseTensor(index, val, shape)
    # AT = tf.cast(AT, tf.float32)
    del val
    del index

    # u_img = np.load(udir + 'u_CT_img_no_scale.npy')
    print('shape of u_img:', u_img.shape)
    # f_noisy_img = np.load(udir + '/f_noisy,angle=' + str(180) + '_ no_scale__0.5.npy')

    # f_img=np.load(udir + '/f,angle=' + str(180) + '_ no_scale__0.5.npy')
    # ini_u_img = np.load(udir + '/ini,angle=' + str(angles) + '_255.0_0.002.npy')


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Model = make_model(batch, AT,w_b,s_shape,out_size)
    # Model=sinLayer(AT,s_shape,out_size)
    Model=make_model_3(AT,s_shape,out_size)
    tf.keras.backend.clear_session()
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)

    # u_img = tf.cast(u_img, tf.float32)
    # f_noisy_img = tf.cast(f_noisy_img, tf.float32)
    # f_img=tf.cast(f_img, tf.float32)

    N=tf.shape(u_img)[0]
    vx=f_noisy_img[N-5:N]
    vy=[f_img[N-5:N],u_img[N-5:N]]
    train_data = tf.data.Dataset.from_tensor_slices((u_img[0:N-5],f_img[0:N-5], f_noisy_img[0:N-5])).shuffle(tf.cast(N-5,tf.int64)).batch(batch)
    # _ = Model(vx[0:1])
    if restore == 1:
        # call the build function in the layers since do not use tf.keras.Input
        ##maybe move the functions in build function to _ini_ need not do this
        _=Model(vx[0:1])
        Model.load_weights(ckpt)
        print('load weights, done')
    for i in range(epoch):
        for iter, ufini in enumerate(train_data):
            u,f, f_noisy = ufini
            # Loss, m1, m2, m3 = train_step(f_noisy, Model, [f, u], loss, psnr, optimizer, vx, vy,epochnum=i)
            Loss, m1, m2, m3 = train_step(f_noisy, Model, [f, u], loss_1, psnr, optimizer, vx, vy, epochnum=i)
            print(iter, "/", i, ":", Loss.numpy(),
                  "psnr_f_fnoisy:", m1.numpy(),
                  "psnr1", [m2[0].numpy(), m2[1].numpy(), m2[2].numpy()],
                  ###psnr of f and f_noisy, u and fbp, u and reconstructe,respectively
                  'psnr3:', [m3[0].numpy(), m3[1].numpy(), m3[2].numpy()]
                  )

        if i%2==0:
            Model.save_weights(ckpt)
    # Model.compile(optimizer=optimizer, loss=[loss], metrics=[psnr])
    # Model.fit(x, y, batch_size=batch, epochs=epoch, callbacks=[tensorboard_callback],
    #           validation_split=1/80)
    Model.save_weights(ckpt)
    # tf.keras.utils.plot_model(Model, 'multi_input_and_output_model.png', show_shapes=True)


@tf.function
def train_step(inputs, model, labels, Loss, Metric, optimizer,vx,vy,epochnum):
    # if epochnum<1000:
    #     weights = 0.9999
    # else:
    #     weights = 0.0001
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=1)
        loss = Loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # m1 = Metric(labels, inputs)
    m1 = tf.reduce_sum(tf.image.psnr(labels[0], inputs, max_val=tf.reduce_max(labels[0]))) / tf.cast(tf.shape(inputs)[0],tf.float32)
    m2 = Metric(labels, model(inputs, training=0))
    m3 = Metric(vy, model(vx, training=0))
    return loss, m1, m2, m3


def loss_1(x, y,weights=0.5):
    x0 = tf.cast(x[0], tf.float32)
    x1 = tf.cast(x[1], tf.float32)
    y0 = tf.cast(y[0], tf.float32)
    y1 = tf.cast(y[1], tf.float32)
    shape = tf.cast(tf.shape(x[0]), tf.float32)
    shape1 = tf.cast(tf.shape(x[1]), tf.float32)
    return weights*tf.reduce_sum(tf.math.square(x0 - y0)) / shape[0] / shape[1] / shape[2] / shape[3]\
    +(1-weights)*tf.reduce_sum(tf.math.square(x1 - y1))/shape1[0] / shape1[1] / shape1[2] / shape1[3]
    # return tf.reduce_sum(tf.math.square(x0 - y0)) / shape[0] / shape[1] / shape[2] / shape[3]
    # return tf.reduce_sum(tf.math.square(x1 - y1))/shape1[0] / shape1[1] / shape1[2] / shape1[3]

def psnr(x, y,max_val=255):
    x0 = tf.cast(x[0], tf.float32)
    x1 = tf.cast(x[1], tf.float32)
    y0 = tf.cast(y[0], tf.float32)
    y1 = tf.cast(y[1], tf.float32)
    y2 = tf.cast(y[2], tf.float32)
    batch = tf.cast(tf.shape(x[1])[0], tf.float32)
    psnr1=tf.reduce_sum(tf.image.psnr(x0, y0, max_val=tf.reduce_max(x0))) / batch######psnr of f and de_sin
    psnr2=tf.reduce_sum(tf.image.psnr(x1, y2, max_val=tf.reduce_max(x1))) / batch######psnr of u and fbp
    psnr3 = tf.reduce_sum(tf.image.psnr(x1, y1, max_val=tf.reduce_max(x1))) / batch#####psnr of u and reconstructed
    return [psnr1,psnr2,psnr3]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    iternum = 20
    epoch = 100

    batch = 5
    theta = np.linspace(0, 180, 180, endpoint=False)
    udir = "./train/"
    vdir = "validate"
    # train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./weights/two_stage_4_2')
    train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./512x512/weights/new1_model_lambda=0.5')