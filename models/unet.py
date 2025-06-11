import tensorflow as tf
import tensorflow.keras.layers as layers            # type: ignore

def conv_block(input, num_filters):
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input)
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    return c1, p1

def convT_block(input, num_filters, skip_conn):
    u1 = layers.Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(input)
    u1 = layers.concatenate([u1, skip_conn], axis=-1)
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(u1)
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c1)

    return c1

def UNet(input_shape, iou_fn, num_filters):
    inputs = layers.Input(input_shape)

    skip_conn = []

    temp = inputs

    for i in num_filters:
        c, p = conv_block(temp, i)
        skip_conn.append(c)
        temp = p

    temp = layers.Conv2D(num_filters[-1]*2, (3, 3), activation='relu', name='bottleneck1', padding='same')(temp)
    # temp = layers.Conv2D(num_filters[-1]*2, (3, 3), activation='relu', name='bottleneck2', padding='same')(temp)

    for i in num_filters[::-1]:
        c = convT_block(temp, i, skip_conn.pop())
        temp = c

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(temp)

    model = tf.keras.Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[iou_fn])

    return model