import tensorflow as tf
import numpy as np
import datetime
import os
import glob


def BN(img):
    # batch_mean, batch_var = tf.nn.moments(img, [0, 1, 2], name='moments')
    # img = tf.nn.batch_normalization(img, batch_mean, batch_var, 0, 1, 1e-3)
    img=tf.keras.layers.BatchNormalization()(img)
    return img

# def conv2d(x, W):
#     tf.keras.layers.Conv2D(64, 3, strides=[1, 1, 1, 1], padding='same')(output)
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x1(x):
    return tf.keras.layers.MaxPool2D([1, 2], strides=[1, 2], padding='same')(x)
    # return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def max_pool_2x2(x):
    # return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.keras.layers.MaxPool2D([2, 2], strides=[2, 2], padding='same')(x)


def max_pool(x, n):
    return tf.keras.layers.MaxPool2D([n, n], strides=[1, 2], padding='same')(x)
    # return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def build_unpool(source, kernel_shape):
    # input_shape = source.get_shape().as_list()
    input_shape=tf.shape(source)
    # return tf.reshape(source,[input_shape[1] * kernel_shape[1], input_shape[2] * kernel_shape[2]])
    return tf.image.resize(source, [input_shape[1] * kernel_shape[1], input_shape[2] * kernel_shape[2]])



def DenseNet(input, growth_rate=16, nb_filter=16, filter_wh=5):
    # shape = input.get_shape().as_list()
    shape=tf.shape(input)
    with tf.name_scope('layer1'):
        input = BN(input)
        input = tf.nn.relu(input)

        # w1_1 = weight_variable([1, 1, shape[3], nb_filter * 4])
        # b1_1 = bias_variable([nb_filter * 4])
        # c1_1 = tf.nn.conv2d(input, w1_1, strides=[1, 1, 1, 1], padding='SAME') + b1_1
        c1_1 = tf.keras.layers.Conv2D(nb_filter * 4, [1,1],1, padding='same')(input)
        ##

        c1_1 = BN(c1_1)
        c1_1 = tf.nn.relu(c1_1)

        # w1 = weight_variable([filter_wh, filter_wh, nb_filter * 4, nb_filter])
        # b1 = bias_variable([nb_filter])
        # c1 = tf.nn.conv2d(c1_1, w1, strides=1, padding='SAME') + b1
        c1 = tf.keras.layers.Conv2D(nb_filter,[filter_wh, filter_wh], 1, padding='same')(c1_1)

    h_concat1 = tf.concat([input, c1], 3)

    with tf.name_scope('layer2'):
        h_concat1 = BN(h_concat1)
        h_concat1 = tf.nn.relu(h_concat1)

        # w2_1 = weight_variable([1, 1, shape[3] + nb_filter, nb_filter * 4])
        # b2_1 = bias_variable([nb_filter * 4])
        # c2_1 = tf.nn.conv2d(h_concat1, w2_1, strides=[1, 1, 1, 1], padding='SAME') + b2_1
        c2_1 = tf.keras.layers.Conv2D(nb_filter * 4, [1, 1], 1, padding='same')(h_concat1)
        ##

        c2_1 = BN(c2_1)
        c2_1 = tf.nn.relu(c2_1)

        # w2 = weight_variable([filter_wh, filter_wh, nb_filter * 4, nb_filter])
        # b2 = bias_variable([nb_filter])
        # c2 = tf.nn.conv2d(c2_1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
        c2 = tf.keras.layers.Conv2D(nb_filter, [filter_wh, filter_wh], 1, padding='same')(c2_1)

    h_concat2 = tf.concat([input, c1, c2], 3)

    with tf.name_scope('layer3'):
        h_concat2 = BN(h_concat2)
        h_concat2 = tf.nn.relu(h_concat2)

        # w3_1 = weight_variable([1, 1, shape[3] + nb_filter + nb_filter, nb_filter * 4])
        # b3_1 = bias_variable([nb_filter * 4])
        # c3_1 = tf.nn.conv2d(h_concat2, w3_1, strides=[1, 1, 1, 1], padding='SAME') + b3_1
        c3_1 = tf.keras.layers.Conv2D(nb_filter * 4, [1, 1], 1, padding='same')(h_concat2)
        ##

        c3_1 = BN(c3_1)
        c3_1 = tf.nn.relu(c3_1)

        # w3 = weight_variable([filter_wh, filter_wh, nb_filter * 4, nb_filter])
        # b3 = bias_variable([nb_filter])
        # c3 = tf.nn.conv2d(c3_1, w3, strides=[1, 1, 1, 1], padding='SAME') + b3
        c3 = tf.keras.layers.Conv2D(nb_filter, [filter_wh, filter_wh], 1, padding='same')(c3_1)

    h_concat3 = tf.concat([input, c1, c2, c3], 3)

    with tf.name_scope('layer4'):
        h_concat3 = BN(h_concat3)
        h_concat3 = tf.nn.relu(h_concat3)

        # w4_1 = weight_variable([1, 1, shape[3] + nb_filter + nb_filter + nb_filter, nb_filter * 4])
        # b4_1 = bias_variable([nb_filter * 4])
        # c4_1 = tf.nn.conv2d(h_concat3, w4_1, strides=[1, 1, 1, 1], padding='SAME') + b4_1
        c4_1 = tf.keras.layers.Conv2D(nb_filter * 4, [1, 1], 1, padding='same')(h_concat3)
        ##

        c4_1 = BN(c4_1)
        c4_1 = tf.nn.relu(c4_1)

        # w4 = weight_variable([filter_wh, filter_wh, nb_filter * 4, nb_filter])
        # b4 = bias_variable([nb_filter])
        # c4 = tf.nn.conv2d(c4_1, w4, strides=[1, 1, 1, 1], padding='SAME') + b4
        c4 = tf.keras.layers.Conv2D(nb_filter, [filter_wh, filter_wh], 1, padding='same')(c4_1)

    return tf.concat([input, c1, c2, c3, c4], 3)

def mix(input_image):
    nb_filter = 16
    # W_conv1 = weight_variable([7, 7, 1, nb_filter])
    # b_conv1 = bias_variable([nb_filter])
    # h_conv1 = (tf.nn.conv2d(input_image, W_conv1, strides=[1, 1, 1, 1],
    #                         padding='SAME') + b_conv1)  # 256*256**(nb_filter)
    h_conv1 =tf.keras.layers.Conv2D(nb_filter, [7, 7], 1, padding='same')(input_image)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME')  # 128*128*(nb_filter)


    D1 = DenseNet(h_pool1, growth_rate=16, nb_filter=nb_filter, filter_wh=5)  # 128*128*(nb_filter*4+nb_filter)

    D1 = BN(D1)
    D1 = tf.nn.relu(D1)
    # W_conv1_T = weight_variable([1, 1, nb_filter + nb_filter * 4, nb_filter])
    # b_conv1_T = bias_variable([nb_filter])
    # h_conv1_T = (
    #         tf.nn.conv2d(D1, W_conv1_T, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_T)  # 128*128*(nb_filter)
    h_conv1_T = tf.keras.layers.Conv2D(nb_filter, [1, 1], 1, padding='same')(D1)

    h_pool1_T = tf.nn.max_pool(h_conv1_T, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')  # 64*64*(nb_filter)

    ##
    D2 = DenseNet(h_pool1_T, growth_rate=16, nb_filter=nb_filter, filter_wh=5)  # 64*64*(4*nb_filter + nb_filter)
    D2 = BN(D2)
    D2 = tf.nn.relu(D2)

    # W_conv2_T = weight_variable([1, 1, nb_filter + nb_filter * 4, nb_filter])
    # b_conv2_T = bias_variable([nb_filter])
    # h_conv2_T = (
    #         tf.nn.conv2d(D2, W_conv2_T, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_T)  # 64*64*(nb_filter)

    h_conv2_T = tf.keras.layers.Conv2D(nb_filter, [1, 1], 1, padding='same')(D2)
    h_pool2_T = tf.nn.max_pool(h_conv2_T, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')  # 32*32*(nb_filter)

    ##
    D3 = DenseNet(h_pool2_T, growth_rate=16, nb_filter=nb_filter, filter_wh=5)  # 32*32*(4*nb_filter + nb_filter)
    D3 = BN(D3)
    D3 = tf.nn.relu(D3)
    # W_conv3_T = weight_variable([1, 1, nb_filter + nb_filter * 4, nb_filter])
    # b_conv3_T = bias_variable([nb_filter])
    # h_conv3_T = (
    #         tf.nn.conv2d(D3, W_conv3_T, strides=[1, 1, 1, 1], padding='SAME') + b_conv3_T)  # 32*32*(nb_filter)

    h_conv3_T = tf.keras.layers.Conv2D(nb_filter, [1, 1], 1, padding='same')(D3)
    h_pool3_T = tf.nn.max_pool(h_conv3_T, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')  # 16*16*(nb_filter)

    ##
    D4 = DenseNet(h_pool3_T, growth_rate=16, nb_filter=nb_filter, filter_wh=5)  # 16*16*(4*nb_filter + nb_filter)
    D4 = BN(D4)
    D4 = tf.nn.relu(D4)
    # W_conv4_T = weight_variable([1, 1, nb_filter + nb_filter * 4, nb_filter])
    # b_conv4_T = bias_variable([nb_filter])
    # h_conv4_T = (
    #         tf.nn.conv2d(D4, W_conv4_T, strides=[1, 1, 1, 1], padding='SAME') + b_conv4_T)  # 16*16*(nb_filter)
    h_conv4_T = tf.keras.layers.Conv2D(nb_filter, [1, 1], 1, padding='same')(D4)

    ##

    # W_conv40 = weight_variable([5, 5, 2 * nb_filter, 2 * nb_filter])
    # b_conv40 = bias_variable([2 * nb_filter])
    # h_conv40 = tf.nn.relu(
    #     tf.nn.conv2d_transpose(tf.concat([build_unpool(h_conv4_T, [1, 2, 2, 1]), h_conv3_T], 3), W_conv40,
    #                            [batch, 64, 64, 2 * nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv40)  # 32*32*40
    h_conv40 = tf.concat([build_unpool(h_conv4_T, [1, 2, 2, 1]), h_conv3_T], 3)
    h_conv40 = tf.keras.layers.Conv2DTranspose(2 * nb_filter,[5,5],strides=1,padding='SAME')(h_conv40)
    h_conv40 = tf.nn.relu(h_conv40)
    batch_mean, batch_var = tf.nn.moments(h_conv40, [0, 1, 2], name='moments')
    h_conv40 = tf.nn.batch_normalization(h_conv40, batch_mean, batch_var, 0, 1, 1e-3)  # 32*32

    # W_conv40_T = weight_variable([1, 1, nb_filter, (2 * nb_filter)])
    # b_conv40_T = bias_variable([nb_filter])
    # h_conv40_T = tf.nn.relu(
    #     tf.nn.conv2d_transpose(h_conv40, W_conv40_T, [batch, 64, 64, nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv40_T)  # 32*32*40
    h_conv40_T = tf.keras.layers.Conv2DTranspose(nb_filter, [1, 1], strides=1, padding='SAME')(h_conv40)
    h_conv40_T = tf.nn.relu(h_conv40_T)
    batch_mean, batch_var = tf.nn.moments(h_conv40_T, [0, 1, 2], name='moments')
    h_conv40_T = tf.nn.batch_normalization(h_conv40_T, batch_mean, batch_var, 0, 1, 1e-3)

    ##
    # W_conv5 = weight_variable([5, 5, 2 * nb_filter, 2 * nb_filter])
    # b_conv5 = bias_variable([2 * nb_filter])
    # h_conv5 = tf.nn.relu(
    #     tf.nn.conv2d_transpose(tf.concat([build_unpool(h_conv40_T, [1, 2, 2, 1]), h_conv2_T], 3), W_conv5,
    #                            [batch, 128, 128, 2 * nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv5)  # 64*64*20
    h_conv5 = tf.concat([build_unpool(h_conv40_T, [1, 2, 2, 1]), h_conv2_T], 3)
    h_conv5 = tf.keras.layers.Conv2DTranspose(2 * nb_filter, [5, 5], strides=1, padding='SAME')(h_conv5)
    h_conv5 = tf.nn.relu(h_conv5)
    batch_mean, batch_var = tf.nn.moments(h_conv5, [0, 1, 2], name='moments')
    h_conv5 = tf.nn.batch_normalization(h_conv5, batch_mean, batch_var, 0, 1, 1e-3)

    # W_conv5_T = weight_variable([1, 1, nb_filter, 2 * nb_filter])
    # b_conv5_T = bias_variable([nb_filter])
    # h_conv5_T = tf.nn.relu(
    #     tf.nn.conv2d_transpose(h_conv5, W_conv5_T, [batch, 128, 128, nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv5_T)  # 64*64*20
    h_conv5_T = tf.keras.layers.Conv2DTranspose(nb_filter, [1, 1], strides=1, padding='SAME')(h_conv5)
    h_conv5_T = tf.nn.relu(h_conv5_T)
    batch_mean, batch_var = tf.nn.moments(h_conv5_T, [0, 1, 2], name='moments')
    h_conv5_T = tf.nn.batch_normalization(h_conv5_T, batch_mean, batch_var, 0, 1, 1e-3)

    ##
    # W_conv6 = weight_variable([5, 5, 2 * nb_filter, 2 * nb_filter])
    # b_conv6 = bias_variable([2 * nb_filter])
    # h_conv6 = tf.nn.relu(
    #     tf.nn.conv2d_transpose(tf.concat([build_unpool(h_conv5_T, [1, 2, 2, 1]), h_conv1_T], 3), W_conv6,
    #                            [batch, 256, 256, 2 * nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv6)
    h_conv6 = tf.concat([build_unpool(h_conv5_T, [1, 2, 2, 1]), h_conv1_T], 3)
    h_conv6 = tf.keras.layers.Conv2DTranspose(2 * nb_filter, [5, 5], strides=1, padding='SAME')(h_conv6)
    h_conv6 = tf.nn.relu(h_conv6)
    batch_mean, batch_var = tf.nn.moments(h_conv6, [0, 1, 2], name='moments')
    h_conv6 = tf.nn.batch_normalization(h_conv6, batch_mean, batch_var, 0, 1, 1e-3)

    # W_conv6_T = weight_variable([1, 1, nb_filter, 2 * nb_filter])
    # b_conv6_T = bias_variable([nb_filter])
    # h_conv6_T = tf.nn.relu(
    #     tf.nn.conv2d_transpose(h_conv6, W_conv6_T, [batch, 256, 256, nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv6_T)  # 64*64*20
    h_conv6_T = tf.keras.layers.Conv2DTranspose(nb_filter, [1, 1], strides=1, padding='SAME')(h_conv6)
    h_conv6_T = tf.nn.relu(h_conv6_T)
    batch_mean, batch_var = tf.nn.moments(h_conv6_T, [0, 1, 2], name='moments')
    h_conv6_T = tf.nn.batch_normalization(h_conv6_T, batch_mean, batch_var, 0, 1, 1e-3)

    # W_conv7 = weight_variable([5, 5, 2 * nb_filter, 2 * nb_filter])
    # b_conv7 = bias_variable([2 * nb_filter])
    # h_conv7 = tf.nn.relu(
    #     tf.nn.conv2d_transpose(tf.concat([build_unpool(h_conv6_T, [1, 2, 2, 1]), h_conv1], 3), W_conv7,
    #                            [batch, 512, 512, 2 * nb_filter], strides=[1, 1, 1, 1],
    #                            padding='SAME') + b_conv7)
    h_conv7 = tf.concat([build_unpool(h_conv6_T, [1, 2, 2, 1]), h_conv1], 3)
    h_conv7 = tf.keras.layers.Conv2DTranspose(2 * nb_filter, [5, 5], strides=1, padding='SAME')(h_conv7)
    h_conv7 = tf.nn.relu(h_conv7)

    # W_conv8 = weight_variable([1, 1, 1, 2 * nb_filter])
    # b_conv8 = bias_variable([1])
    # h_conv8 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv7, W_conv8, [batch, 512, 512, 1], strides=[1, 1, 1, 1],
    #                                             padding='SAME') + b_conv8)
    h_conv8 = tf.keras.layers.Conv2DTranspose(1, [1, 1], strides=1, padding='SAME')(h_conv7)
    h_conv8 = tf.nn.relu(h_conv8)
    return h_conv8

def make_model(batch,ux=None,uy=None):
    inputs = tf.keras.Input(shape=(ux,uy,1),batch_size=batch)
    outputs=mix(inputs)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

def train(epoch, udir,batch, theta, iternum, restore=0, ckpt='./weights/CT_tf2_4'):
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
    Model = make_model(batch,ux=256,uy=256)
    if restore == 1:
        # call the build function in the layers since do not use tf.keras.Input
        ##maybe move the functions in build function to _ini_ need not do this
        _=Model(ini_u_img[0:1])
        Model.load_weights(ckpt)
        print('load weights, done')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)

    u_img=tf.cast(u_img,tf.float32)
    ini_u_img = tf.cast(ini_u_img, tf.float32)
    N=tf.shape(u_img)[0]
    vx=ini_u_img[N-5:N]
    vy=u_img[N-5:N]
    # vx=tf.cast(vx,tf.float32)
    # vy = tf.cast(vy, tf.float32)
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
        if i%10==0:
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    iternum = 20
    epoch = 200
    batch = 5
    angles = 180
    theta = np.linspace(0, 180, angles, endpoint=False)
    udir = "./train/"
    vdir = "validate"
    train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./512x512/weights/DD_NET')