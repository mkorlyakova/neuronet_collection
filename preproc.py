import tensorflow as tf

def preprocess_val(f = '',SIZE = [224,224]):

    #print('ФАЙЛ:',f)
    image_bytes = tf.io.read_file(f)
    rgb = tf.image.decode_image(image_bytes, channels=3)
    #print(rgb.shape)
    # 
    rgb.set_shape((None, None, 3))
    initial_width = tf.shape(rgb)[-3]
    initial_height = tf.shape(rgb)[-2]
    #print(initial_width,initial_height)
    delta_ = tf.abs(tf.subtract(initial_width, initial_height))
    image_central = tf.cond(initial_height > initial_width,
    lambda: rgb[ :, delta_:(initial_width + delta_), :],
    lambda: rgb[ delta_:(initial_height + delta_), :, :])
    #image_central = get_square(rgb, int(delta_ / 2))
    rgb_cropped = tf.image.convert_image_dtype(image_central, dtype=tf.float32)
    rgb_ready = tf.image.resize(rgb_cropped, SIZE)

    #mean, variance = tf.nn.moments(rgbs, axes=[0, 1, 2])

    #images = (rgbs - mean) / (tf.sqrt(variance) + 1e-7)

    return rgb_ready