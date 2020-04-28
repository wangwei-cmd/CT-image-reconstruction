
import tensorflow as tf
from collections import OrderedDict
import numpy as np
import datetime
import os


def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [x1_shape[0], x2_shape[1], x2_shape[2], x1_shape[3]]
    x1_crop = tf.slice(x1, offsets, size)
    # return tf.concat([x1_crop, x2], 3)
    return tf.keras.layers.Concatenate(3)([x1_crop, x2])


def create_conv_net(x, channels=1, n_class=1, layers=3, features_root=16, filter_size=3, pool_size=2, Ngpu=1,
                    maxpool=True, summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, [-1, nx, ny, channels])
    in_node = x_image
    # batch_size = tf.shape(x_image)[0]

    # weights = []
    # weights_d = []
    # biases = []
    # biases_d = []
    dw_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    padding = 'same'
    if Ngpu == 1:
        gname = '0'
    else:
        gname = '1'
    # down layers
    with tf.device('/gpu:0'):
        for layer in range(0, layers):
            features = 2 ** layer * features_root
            filters = features
            if layer == 0:
                # w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
                w1_kernel_size=[filter_size, filter_size]
            else:
                # w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)
                w1_kernel_size = [filter_size, filter_size]

            # w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            w2_kernel_size=[filter_size, filter_size]
            # b1 = bias_variable([features])
            # b2 = bias_variable([features])

            # conv = conv2d(in_node, w1, keep_prob, padding)
            # in_node = tf.nn.relu(conv + b1)
            conv=tf.keras.layers.Conv2D(filters,w1_kernel_size,padding=padding)(in_node)
            in_node=tf.keras.layers.ReLU()(conv)

            # conv = conv2d(in_node, w2, keep_prob, padding)
            # in_node = tf.nn.relu(conv + b2)
            conv = tf.keras.layers.Conv2D(filters, w2_kernel_size, padding=padding)(in_node)
            in_node = tf.keras.layers.ReLU()(conv)

            dw_h_convs[layer] = in_node
            # dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
            # convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                if maxpool:
                    in_node = tf.keras.layers.MaxPool2D(pool_size)(dw_h_convs[layer])
                else:
                    in_node = tf.keras.layers.AveragePooling2D(pool_size)(dw_h_convs[layer])

                # pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                # in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers - 1]

    with tf.device('/gpu:0'):
        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            # stddev = np.sqrt(2 / (filter_size ** 2 * features))

            # wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev)
            # bd = bias_variable([features // 2])
            # in_node = tf.nn.relu(deconv2d(in_node, wd, pool_size, padding) + bd)
            in_node = tf.keras.layers.Conv2DTranspose(features, pool_size,strides=2, padding=padding)(in_node)
            in_node = tf.keras.layers.ReLU()(in_node)

            conv = crop_and_concat(dw_h_convs[layer], in_node)

            # w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
            # w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
            # b1 = bias_variable([features // 2])
            # b2 = bias_variable([features // 2])
            filters=features // 2
            w1_kernel_size=[filter_size, filter_size]
            w2_kernel_size=[filter_size, filter_size]

            # conv = conv2d(conv, w1, keep_prob, padding)
            conv=tf.keras.layers.Conv2D(filters, w1_kernel_size, padding=padding)(conv)
            # conv = tf.nn.relu(conv + b1)
            conv=tf.keras.layers.ReLU()(conv)

            # conv = conv2d(conv, w2, keep_prob, padding)
            # in_node = tf.nn.relu(conv + b2)
            conv = tf.keras.layers.Conv2D(filters, w2_kernel_size, padding=padding)(conv)
            in_node =tf.keras.layers.ReLU()(conv)

            # weights.append((w1, w2))
            # weights_d.append((wd))
            # biases.append((b1, b2))
            # biases_d.append((bd))

            # convs.append((conv1, conv2))

            size *= 2
            size -= 4

        # with tf.device('/gpu:1'):
        # Output Map
        # weight = weight_variable([1, 1, features_root, n_class], stddev)
        # bias = bias_variable([n_class])
        # conv = conv2d(in_node, weight, tf.constant(1.0), padding)
        conv=tf.keras.layers.Conv2D(n_class, [1, 1], padding=padding)(in_node)
        # output_map = conv + bias + x_image  # tf.nn.relu(conv + bias)
        output_map = conv + x_image
    return output_map


def make_model(batch,ux=256,uy=256):
    inputs = tf.keras.Input(shape=(ux,uy,1),batch_size=batch)
    outputs=create_conv_net(inputs)
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


# @tf.function
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
    # tf.debugging.set_log_device_placement(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    iternum = 20
    epoch = 200
    batch = 5
    angles = 180
    theta = np.linspace(0, 180, angles, endpoint=False)
    # udir = "/home/wangwei/ct-compare/CPTAC-LUAD//npy/"
    udir = "./train/"
    vdir = "validate"
    train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./512x512/weights/fbpconv')