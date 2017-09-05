# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import tensorflow as tf

os.chdir(os.getcwd())


# {{{ constants

CONTENT_FILE = './image/input/test_input_01.JPG'
STYLE_FILE = './image/style/test_style_01.jpg'

FILTER_CONF = {
    'conv': {
        'height': 3,
        'width': 3
    },
    'maxpool': {
        'height': 2,
        'width': 2
    }
}

NETWORK_CONF = {
    'layer1': {
        'num_channels': 2,
        'num_conv': 2
    },
    'layer2': {
        'num_channels': 4,
        'num_conv': 2
    },
    'layer3': {
        'num_channels': 8,
        'num_conv': 4
    },
    'layer4': {
        'num_channels': 16,
        'num_conv': 4
    },
    'layer5': {
        'num_channels': 32,
        'num_conv': 4
    }
}

wl = 1 / 5.0

ALPHA = 1
BETA = 1000
# constants }}}

# {{{ def read_images_as_jpeg(file):
def read_images_as_jpeg(file):

    """
    JPEG Image Reader

    This function reads the content and style images as JPEG format.
    These image data must be square for now, different height and
    width will be able to supplied for future.

    Args:
        file : str. A path of the image file.

    Returns:
        A tuple. Each Elements are Tensor object of the read images.
    """

    filename_queue = tf.train.string_input_producer([file])
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)

    # Read image is a Tensor object which tf.nn.conv2d cannot handle,
    # so convert it to numpy.ndarray.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess)
    # Returned array's shape will be (height, width, channel).
    image_array_hwc = sess.run(image)
    new_shape = [1]
    new_shape.extend(image_array_hwc.shape)

    # return image_array_chw
    return image_array_hwc.reshape(new_shape)
# def read_images_as_jpeg }}}

# {{{ def calculate_convolutional_layer(x, filter_height, filter_width, output_channels):
def calculate_convolutional_layer(x, filter_height, filter_width, output_channels):

    """
    Executeing a convolutional layer task.

    Args:
        x                     : A tf.Placeholder object for an image data.
        filter_height   (int) : A height of each filters.
        filter_width    (int) : A width of each filters.
        output_channels (int) : A number of channels which must be outputed.

    Returns:
        An activation of an convolutional layer.
    """

    if ((not isinstance(filter_height, int))
        or (not isinstance(filter_width, int))
        or (not isinstance(output_channels, int))):
        raise TypeError('"filter_height" and "filter_width" and "output_channels" '
                        'must be integer.')

    # TODO: 入力画像の縦、横、チャンネル数を属性で取得できるようにする
    # 例） input_channels = x.num_channels
    if isinstance(x, tf.Variable):
        shape = x.get_shape().as_list()
        input_channels = shape[-1]
    else:
        input_channels = int(x.shape[-1])
    filter_value = 1 / float(filter_height * filter_width)
    epsilon = filter_value / 10.0
    W = tf.Variable(
        tf.random_uniform(
            shape=[filter_height,
                   filter_width,
                   input_channels,
                   output_channels],
            minval=filter_value - epsilon,
            maxval=filter_value + epsilon
        )
    )
    h = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.constant(0.1, shape=[output_channels]))
    convoluted_image = tf.nn.relu(h + b)

    return convoluted_image
# calculate_convolutional_layer }}}

# {{{ def calculate_max_pooling_layer(x, ksize, strides):
def calculate_max_pooling_layer(x, ksize, strides):

    """Wrapper function of tf.nn.max_pool.

    Args:
        x       : A Tensor produced by calculate_convolutional_layer.
        ksize   : A list of ints that has length >= 4. The size of
                  the window for each dimension of the input tensor.
        strides : A list of ints that has length >= 4. The stride
                  of the sliding window for each dimension of the
                  input tensor.

    Returns:
        A pooled image.
    """

    pooled_image = tf.nn.max_pool(x,
                                  ksize=ksize,
                                  strides=strides,
                                  padding='SAME')

    return pooled_image
# calculate_max_pooling_layer }}}

# {{{ class FeatureMapHolder(object):
class FeatureMapHolder(object):

    def __init__(self):

        self.conv = []
        self.pool = None

    def set_conv(self, mat):

        self.conv.append(mat)

    def get_conv(self, idx):

        if idx is None:
            raise ValueError('idx is required.')

        if not isinstance(idx, int):
            raise ValueError('idx must be an integer.')

        return self.conv[idx]

    def set_pool(self, mat):

        self.pool = mat

    def get_pool(self):

        return self.pool

    def get(self, type, idx=None):

        if type == 'pool':
            return self.get_pool()
        elif type == 'conv':
            return self.get_conv(idx)
# FeatureMapHolder }}}

# {{{ def apply_vgg_network_unit(x, channels, num_conv):
def apply_vgg_network_unit(x, channels, num_conv):

    """Apply VGG Network From a Convolutional Layer to Max Pooling Layer.

    Table 1 of Simonyan and Zisserman(2015) is separated by 5 parts,
    each parts is from an input data or a pooled data at previous part
    to a maxpool.
    This function provides to apply a that part.
    This will apply recursively.

    Args:
        x (Tensor)     : An input data or A Max pooled data returned by
                         this function.
        channels (int) : A number of channels described at Table 1 of
                         Simonyan and Zisserman(2015).
        num_conv (int) : A number of applying covolutional layers.
                         See Simonyan and Zisserman(2015) for detail.

    Returns:
        A ConvNetProgressHolder object.
    """

    if num_conv < 2:
        raise ValueError('num_conv must be >= 2.')

    feature_maps = FeatureMapHolder()

    conv = calculate_convolutional_layer(
        x=x,
        filter_height=FILTER_CONF['conv']['height'],
        filter_width=FILTER_CONF['conv']['width'],
        output_channels=channels
    )
    feature_maps.set_conv(conv)

    for i in range(1, num_conv):
        conv = calculate_convolutional_layer(
            x=feature_maps.get(type='conv', idx=i-1),
            filter_height=FILTER_CONF['conv']['height'],
            filter_width=FILTER_CONF['conv']['width'],
            output_channels=channels
        )
        feature_maps.set_conv(conv)

    kernel_size = [
        1,
        FILTER_CONF['maxpool']['height'],
        FILTER_CONF['maxpool']['width'],
        1
    ]
    pool = calculate_max_pooling_layer(
        x=feature_maps.get('conv', idx=i-1),
        ksize=kernel_size,
        strides=[1, 2, 2, 1]
    )
    feature_maps.set_pool(pool)

    return feature_maps
# apply_vgg_network_unit }}}

# {{{ def constract_layer_to_extract_feature_map(image):
def constract_layer_to_extract_feature_map(image):

    layers = []

    layer1 = apply_vgg_network_unit(
        x=image,
        channels=NETWORK_CONF['layer1']['num_channels'],
        num_conv=NETWORK_CONF['layer1']['num_conv']
    )
    layers.append(layer1)

    layer2 = apply_vgg_network_unit(
        x=layer1.get(type='pool'),
        channels=NETWORK_CONF['layer2']['num_channels'],
        num_conv=NETWORK_CONF['layer2']['num_conv']
    )
    layers.append(layer2)

    layer3 = apply_vgg_network_unit(
        x=layer2.get(type='pool'),
        channels=NETWORK_CONF['layer3']['num_channels'],
        num_conv=NETWORK_CONF['layer3']['num_conv']
    )
    layers.append(layer3)

    layer4 = apply_vgg_network_unit(
        x=layer3.get(type='pool'),
        channels=NETWORK_CONF['layer4']['num_channels'],
        num_conv=NETWORK_CONF['layer4']['num_conv']
    )
    layers.append(layer4)

    layer5 = apply_vgg_network_unit(
        x=layer4.get(type='pool'),
        channels=NETWORK_CONF['layer5']['num_channels'],
        num_conv=NETWORK_CONF['layer5']['num_conv']
    )
    layers.append(layer5)

    return layers
# constract_layer_to_extract_feature_map }}}

# {{{ def convert_nhwc_to_nchw(image):
def convert_nhwc_to_nchw(image):
    return tf.transpose(image, perm=[0, 3, 1, 2])
# convert_nhwc_to_nchw }}}

# {{{ def flatten_2d(x):

def flatten_2d(image):
    height, width = image.shape.as_list()
    return tf.reshape(image, shape=[height * width])

# flatten_2d }}}

# {{{ def flatten_4d(image):

def flatten_4d(image):
    batch, channel, height, width = image.shape.as_list()
    return tf.reshape(image,
                      shape=[batch, channel, height * width])

# flatten_4d }}}

# {{{ def calculate_gram_matrix(x):
def calculate_gram_matrix(x):

    return tf.matmul(x, tf.transpose(x))

# calculate_gram_matrix }}}

# {{{ def make_placeholder(image):
def make_placeholder(image):

    return tf.placeholder(np.float32, image.shape)

# make_placeholder }}}

# {{{ def construct_layer_of_extracting_content_feature(content):
def construct_layer_of_extracting_content_feature(content):

    feature_raw = constract_layer_to_extract_feature_map(content)
    selected_feature_raw = feature_raw[3].conv[1]
    selected_feature_nchw = convert_nhwc_to_nchw(selected_feature_raw)

    return flatten_4d(selected_feature_nchw)

# construct_layer_of_extracting_content_feature }}}

# {{{ def construct_layer_of_extracting_style_feature(style):
def construct_layer_of_extracting_style_feature(style):
    feature_raw = constract_layer_to_extract_feature_map(style)
    # selected_feature_raw = [
    #     feature_raw[0].conv[0],
    #     feature_raw[1].conv[0],
    #     feature_raw[2].conv[0],
    #     feature_raw[3].conv[0],
    #     feature_raw[4].conv[0]
    # ]
    conv1_1 = feature_raw[0].conv[0]
    conv2_1 = feature_raw[1].conv[0]
    conv3_1 = feature_raw[2].conv[0]
    conv4_1 = feature_raw[3].conv[0]
    conv5_1 = feature_raw[4].conv[0]
    selected_feature_raw = [
        tf.where(tf.is_nan(conv1_1), tf.zeros_like(conv1_1), conv1_1),
        tf.where(tf.is_nan(conv2_1), tf.zeros_like(conv2_1), conv2_1),
        tf.where(tf.is_nan(conv3_1), tf.zeros_like(conv3_1), conv3_1),
        tf.where(tf.is_nan(conv4_1), tf.zeros_like(conv4_1), conv4_1),
        tf.where(tf.is_nan(conv5_1), tf.zeros_like(conv5_1), conv5_1)
    ]
    style_feature = []
    for x in selected_feature_raw:
        nchw = convert_nhwc_to_nchw(x)
        flattened = flatten_4d(nchw)
        style_feature.append(flattened[0])
    return style_feature
# construct_layer_of_extracting_style_feature }}}

# {{{ def calculate_el(synthesize, style):
def calculate_el(synthesize, style):
    n, m = style.shape.as_list()
    gram_synthesize = calculate_gram_matrix(synthesize)
    gram_style = calculate_gram_matrix(style)
    diff = gram_synthesize - gram_style
    el = (1 / (4 * (n**2) * (m**2))) * tf.reduce_sum(diff**2)
    return el
# calculate_el }}}

# {{{ def whiten_ndarray_image(x):
def whiten_ndarray_image(x):

    if not isinstance(x, np.ndarray):
        raise TypeError('x must be a numpy.ndarray.')

    whitening_tensor = tf.image.per_image_standardization(x[0])
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    whitened = sess.run(whitening_tensor)
    return whitened.reshape(x.shape)
# whiten_ndarray_image }}}


if (__name__ == '__main__'):

    # ---------------------------------------------------------
    # Extracting feature maps from the content image.
    # ---------------------------------------------------------

    tmp = read_images_as_jpeg(CONTENT_FILE)
    content_data = whiten_ndarray_image(tmp)
    # content_data = read_images_as_jpeg(CONTENT_FILE)
    content_image = tf.placeholder(np.float32, content_data.shape)
    content_feature_model = construct_layer_of_extracting_content_feature(
        content=content_image
    )
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    content_feature_map = sess.run(content_feature_model,
                                   feed_dict={content_image: content_data})

    # ---------------------------------------------------------
    # Extracting feature maps from the style image.
    # ---------------------------------------------------------

    tmp = read_images_as_jpeg(STYLE_FILE)
    style_data = whiten_ndarray_image(tmp)
    # style_data = read_images_as_jpeg(STYLE_FILE)
    style_image = tf.placeholder(np.float32, style_data.shape)
    style_feature_model = construct_layer_of_extracting_style_feature(
        style=style_image
    )
    sess.run(tf.global_variables_initializer())
    style_feature_map = sess.run(style_feature_model,
                                 feed_dict={style_image: style_data})

    # ---------------------------------------------------------
    # Extracting feature maps from the synthesized image.
    # ---------------------------------------------------------

    batch, height, width, channel = content_data.shape

    # Not Whitening.
    # synthesized_image = tf.Variable(
    #     tf.random_normal([batch, height, width, channel],
    #                      mean=125,
    #                      stddev=30),
    #     trainable=True
    # )
    # synthesized_content_feature = construct_layer_of_extracting_content_feature(
    #     content=synthesized_image
    # )
    # synthesized_style_feature = construct_layer_of_extracting_style_feature(
    #     style=synthesized_image
    # )

    # Whitening.
    synthesized_image = tf.Variable(
        tf.random_normal([height, width, channel]),
        trainable=True
    )
    whitened_image_3d = tf.image.per_image_standardization(synthesized_image)
    whitened_image_4d = tf.reshape(whitened_image_3d,
                                   shape=[batch, height, width, channel])
    synthesized_content_feature = construct_layer_of_extracting_content_feature(
        content=whitened_image_4d
    )
    synthesized_style_feature = construct_layer_of_extracting_style_feature(
        style=whitened_image_4d
    )

    # ---------------------------------------------------------
    # Constructing loss function of contents.
    # ---------------------------------------------------------

    content_feature = tf.placeholder(tf.float32,
                                     shape=content_feature_map.shape)
    content_diff = synthesized_content_feature - content_feature
    x = content_diff[0]
    loss_content = 0.5 * tf.reduce_sum(x**2)

    # ---------------------------------------------------------
    # Constructing loss function of style.
    # ---------------------------------------------------------


    style1 = tf.placeholder(tf.float32,
                            shape=style_feature_map[0].shape)
    style2 = tf.placeholder(tf.float32,
                            shape=style_feature_map[1].shape)
    style3 = tf.placeholder(tf.float32,
                            shape=style_feature_map[2].shape)
    style4 = tf.placeholder(tf.float32,
                            shape=style_feature_map[3].shape)

    loss_style = 0
    loss_style += wl * calculate_el(synthesized_style_feature[0], style1)
    loss_style += wl * calculate_el(synthesized_style_feature[1], style2)
    loss_style += wl * calculate_el(synthesized_style_feature[2], style3)
    loss_style += wl * calculate_el(synthesized_style_feature[3], style4)

    # ---------------------------------------------------------
    # Constructing loss function.
    # ---------------------------------------------------------

    loss = ALPHA * loss_content + BETA * loss_style
    train_step = tf.train.AdamOptimizer().minimize(loss)

    # ---------------------------------------------------------
    # Run training steps.
    # ---------------------------------------------------------

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train_step,
                 feed_dict={
                     content_feature: content_feature_map,
                     style1: style_feature_map[0],
                     style2: style_feature_map[1],
                     style3: style_feature_map[2],
                     style4: style_feature_map[3]
                 }
        )
        if (i + 1) % 100 == 0:
            loss_content_val, loss_style_val, loss_val = sess.run(
                    [loss_content, loss_style, loss],
                     feed_dict={
                         content_feature: content_feature_map,
                         style1: style_feature_map[0],
                         style2: style_feature_map[1],
                         style3: style_feature_map[2],
                         style4: style_feature_map[3]
                     }
            )

            synthesize_val, synthesized_image_val = sess.run(
                    [synthesized_style_feature, synthesized_image],
                     feed_dict={
                         content_feature: content_feature_map,
                         style1: style_feature_map[0],
                         style2: style_feature_map[1],
                         style3: style_feature_map[2],
                         style4: style_feature_map[3]
                     }
            )

            # DEBUG
            print('------------------------------------------------------')
            print('Step              : {}'.format(i+1))
            # print('Style             : {}'.format(style_val))
            # print('Synthesize        : {}'.format(synthesize_val))
            # print('Synthesized image : {}'.format(synthesized_image_val))
            # print('Diff              : {}'.format(diff_val))
            print('Loss Style        : {}'.format(loss_style_val))
            print('Loss Content      : {}'.format(loss_content_val))
            print('Loss              : {}'.format(loss_val))
            print()

