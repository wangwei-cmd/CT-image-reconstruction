import tensorflow as tf
import cv2 as cv
import numpy as np
import os
import datetime


def redcnn(in_image, kernel_size=[5, 5], filter_size=96, conv_stride=1, initial_std=0.01):
    # conv layer1
    conv1 = tf.keras.layers.Conv2D(filter_size, kernel_size, conv_stride, padding='valid')(in_image)
    conv1=tf.keras.layers.ReLU()(conv1)
    # conv layer2
    conv2 = tf.keras.layers.Conv2D( filter_size, kernel_size, conv_stride, padding='valid')(conv1)
    conv2 = shortcut_deconv8 = tf.keras.layers.ReLU()(conv2)
    # conv layer3
    conv3 = tf.keras.layers.Conv2D(filter_size, kernel_size, conv_stride, padding='valid')(conv2)
    conv3 = tf.keras.layers.ReLU()(conv3)
    # conv layer4
    conv4 = tf.keras.layers.Conv2D(filter_size, kernel_size, conv_stride, padding='valid')(conv3)
    conv4 = shortcut_deconv6 = tf.keras.layers.ReLU()(conv4)
    # conv layer5
    conv5 = tf.keras.layers.Conv2D(filter_size, kernel_size, conv_stride, padding='valid')(conv4)
    conv5 = tf.keras.layers.ReLU()(conv5)

    """
    decoder
    """
    # deconv 6 + shortcut (residual style)
    deconv6 = tf.keras.layers.Conv2DTranspose(filter_size, kernel_size, conv_stride, padding='valid')(conv5)
    deconv6 += shortcut_deconv6
    deconv6 = tf.keras.layers.ReLU()(deconv6)
    # deconv 7
    deconv7 = tf.keras.layers.Conv2DTranspose(filter_size, kernel_size, conv_stride, padding='valid')(deconv6)
    deconv7 = tf.keras.layers.ReLU()(deconv7)
    # deconv 8 + shortcut
    deconv8 = tf.keras.layers.Conv2DTranspose(filter_size, kernel_size, conv_stride, padding='valid')(deconv7)
    deconv8 += shortcut_deconv8
    deconv8 = tf.keras.layers.ReLU()(deconv8)
    # deconv 9
    deconv9 = tf.keras.layers.Conv2DTranspose(filter_size, kernel_size, conv_stride, padding='valid')(deconv8)
    deconv9 = tf.keras.layers.ReLU()(deconv9)
    # deconv 10 + shortcut
    deconv10 = tf.keras.layers.Conv2DTranspose(1, kernel_size, conv_stride, padding='valid')(deconv9)
    deconv10 += in_image
    output = tf.keras.layers.ReLU()(deconv10)
    return output


def make_model(batch):
    inputs = tf.keras.Input(shape=(None,None,1),batch_size=batch)
    outputs=redcnn(inputs)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

def train(epoch, udir,batch, theta, iternum, restore=0, ckpt='./weights/CT_tf2_4'):
    max_val = 255
    angles = np.shape(theta)[0]
    u_img = np.load(udir + 'u_CT_img_no_scale.npy')
    print('shape of u_img:', u_img.shape)
    # f_img = np.load(udir + '/f,angle=' + str(angles) + '_255.0_0.002.npy')
    ini_u_img = np.load(udir + 'ini,angle=60_no_scale__0.5.npy')

    M = np.max(np.max(ini_u_img, 1), 1)
    M = np.reshape(M, [np.shape(M)[0], 1, 1, 1])
    u_img = u_img / M * 255
    ini_u_img = ini_u_img / M * 255


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
    Model = make_model(batch)
    if restore == 1:
        # call the build function in the layers since do not use tf.keras.Input
        ##maybe move the functions in build function to _ini_ need not do this
        _=Model(ini_u_img[0:1])
        Model.load_weights(ckpt)
        print('load weights, done')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)

    u_img = tf.cast(u_img, tf.float32)
    ini_u_img = tf.cast(ini_u_img, tf.float32)
    N=tf.shape(u_img)[0]
    vx=ini_u_img[N-5:N]
    vy=u_img[N-5:N]
    vx=tf.cast(vx,tf.float32)
    vy = tf.cast(vy, tf.float32)
    train_data = tf.data.Dataset.from_tensor_slices((u_img[0:N-5], ini_u_img[0:N-5])).batch(batch)
    for i in range(epoch):
        for iter, ufini in enumerate(train_data):
            u, ini_u = ufini
            Loss, m1, m2,m3 = train_step(ini_u, Model, u, loss, psnr, optimizer,vx,vy)
            print(iter, "/", i, ":", Loss.numpy(),
                  "psnr1:", m1.numpy(),
                  "psnr2:", m2.numpy(),
                  'psnr3:', m3.numpy()
                  )
        if i%2==0:
            Model.save_weights(ckpt)
    # Model.compile(optimizer=optimizer, loss=[loss], metrics=[psnr])
    # Model.fit(x, y, batch_size=batch, epochs=epoch, callbacks=[tensorboard_callback],
    #           validation_split=1/80)
    Model.save_weights(ckpt)
    # tf.keras.utils.plot_model(Model, 'multi_input_and_output_model.png', show_shapes=True)


@tf.function
def train_step(inputs, model, labels, Loss, Metric, optimizer,vx,vy):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=1)
        loss = Loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    m1 = Metric(labels, inputs)
    m2 = Metric(labels, model(inputs, training=0))
    m3 = Metric(vy, model(vx, training=0))
    return loss, m1, m2, m3


def loss(x, y):
    x1 = tf.cast(x, tf.float32)
    y1 = tf.cast(y, tf.float32)
    shape = tf.cast(tf.shape(x), tf.float32)
    return tf.reduce_sum(tf.math.square(x1 - y1)) / shape[0] / shape[1] / shape[2] / shape[3]


def psnr(x, y,max_val=255):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    batch = tf.cast(tf.shape(x)[0], tf.float32)
    return tf.reduce_sum(tf.image.psnr(x, y, max_val=tf.reduce_max(x))) / batch



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    iternum = 20
    epoch = 200
    batch = 5
    angles = 60
    theta = np.linspace(0, 180, angles, endpoint=False)
    # udir = "/home/wangwei/ct-compare/CPTAC-LUAD//npy/"
    udir = "./train/"
    vdir = "validate"
    train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./512x512/weights/red_cnn')