import tensorflow as tf

class PatchWiseProcessingLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride, patch_fn, merge_method='avg', **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.patch_fn = patch_fn
        self.merge_method = merge_method

    def call(self, inputs):
        input_shape = tf.shape(inputs)

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.stride[0], self.stride[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches_shape = tf.shape(patches)
        patch_h, patch_w = self.patch_size
        num_patches = patches_shape[1] * patches_shape[2]
        channels = inputs.shape[-1]

        patches = tf.reshape(patches, (input_shape[0], num_patches, patch_h, patch_w, channels))

        processed_patches = tf.map_fn(
            lambda batch: tf.map_fn(self.patch_fn, batch),
            patches
        )

        output = self._stitch_patches(processed_patches, input_shape)

        return output

    def _stitch_patches(self, processed_patches, input_shape):
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        stride_h, stride_w = self.stride
        patch_h, patch_w = self.patch_size

        out = tf.zeros((batch_size, height, width, channels), dtype=processed_patches.dtype)
        weight = tf.zeros((batch_size, height, width, channels), dtype=processed_patches.dtype)

        patches_per_row = (width - patch_w) // stride_w + 1

        for i in range((height - patch_h) // stride_h + 1):
            for j in range((width - patch_w) // stride_w + 1):
                patch_idx = i * patches_per_row + j
                patch = processed_patches[:, patch_idx]

                h_start = i * stride_h
                w_start = j * stride_w

                h_end = h_start + patch_h
                w_end = w_start + patch_w

                out = self._scatter_add(out, patch, h_start, h_end, w_start, w_end)
                weight = self._scatter_add(weight, tf.ones_like(patch), h_start, h_end, w_start, w_end)

        if self.merge_method == 'avg':
            output = tf.math.divide_no_nan(out, weight)
        elif self.merge_method == 'max':
            output = out
        elif self.merge_method == 'min':
            output = out
        else:
            raise ValueError(f"Unsupported merge_method: {self.merge_method}")

        return output

    def _scatter_add(self, tensor, patch, h_start, h_end, w_start, w_end):
        mask = tf.pad(patch, paddings=[
            [0, 0],
            [h_start, tensor.shape[1] - h_end],
            [w_start, tensor.shape[2] - w_end],
            [0, 0]
        ])
        return tensor + mask

def example_patch_fn(patch):
    return patch * 0.5

inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = PatchWiseProcessingLayer(
    patch_size=(4, 4),
    stride=(2, 2),
    patch_fn=example_patch_fn,
    merge_method='avg'
)(x)

x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)

model = tf.keras.Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])