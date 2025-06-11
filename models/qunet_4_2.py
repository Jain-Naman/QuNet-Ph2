import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers # type: ignore

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

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

def PQC1(n_qubits, weight_0, weight_1):

    indices = int(n_qubits/2)

    # U1
    for i in range(indices):
        qml.RX(weight_0[0], wires=2*i)
        qml.RZ(weight_0[1], wires=2*i+1)
        qml.CNOT(wires=[2*i, 2*i+1])
    
    for i in range(1, indices):
        qml.RX(weight_0[0], wires=2*i - 1)
        qml.RZ(weight_0[1], wires=2*i)
        qml.CNOT(wires=[2*i - 1, 2*i])

    # V1
    for i in range(indices):
        qml.CZ(wires=[2*i, 2*i+1])

    indices = int(indices/2)

    # U2
    for i in range(indices):
        qml.RX(weight_1[0], wires=2*(2*i))
        qml.RY(weight_1[1], wires=2*(2*i+1))
        qml.CNOT(wires=[2*(2*i), 2*(2*i+1)])
    
    for i in range(1, indices):
        qml.RX(weight_1[0], wires=2*(2*i - 1))
        qml.RY(weight_1[1], wires=2*(2*i))
        qml.CNOT(wires=[2*(2*i - 1), 2*(2*i)])
    
    # V2
    for i in range(indices):
        qml.CZ(wires=[2*(2*i), 2*(2*i+1)])

def PQC2(n_qubits, weight_0, weight_1):

    indices = int(n_qubits/2)

    # U1
    for i in range(indices):
        qml.RY(weight_0[0], wires=2*i)
        qml.RX(weight_0[1], wires=2*i+1)
        qml.CNOT(wires=[2*i, 2*i+1])
    
    for i in range(1, indices):
        qml.RY(weight_0[0], wires=2*i - 1)
        qml.RX(weight_0[1], wires=2*i)
        qml.CNOT(wires=[2*i - 1, 2*i])

    # V1
    for i in range(indices):
        qml.CZ(wires=[2*i, 2*i+1])

    indices = int(indices/2)

    # U2
    for i in range(indices):
        qml.RZ(weight_1[0], wires=2*(2*i))
        qml.RX(weight_1[1], wires=2*(2*i+1))
        qml.CNOT(wires=[2*(2*i), 2*(2*i+1)])
    
    for i in range(1, indices):
        qml.RZ(weight_1[0], wires=2*(2*i - 1))
        qml.RX(weight_1[1], wires=2*(2*i))
        qml.CNOT(wires=[2*(2*i - 1), 2*(2*i)])
    
    # V2
    for i in range(indices):
        qml.CZ(wires=[2*(2*i), 2*(2*i+1)])

def QuFeX_4_2(input, splits):

    n_qubits = 4

    dev1 = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev1, interface='tf')
    def qcnn1(inputs, weight_0, weight_1):
        
        qml.AngleEmbedding(inputs*np.pi, wires=range(n_qubits), rotation="Y")
        
        PQC1(n_qubits, weight_0, weight_1)
    
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    dev2 = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev2, interface='tf')
    def qcnn2(inputs, weight_0, weight_1):
        
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        qml.AngleEmbedding(inputs*np.pi, wires=range(n_qubits), rotation="Z")
        
        PQC2(n_qubits, weight_0, weight_1)
    
        return [qml.expval(qml.PauliX(wires=i)) for i in range(n_qubits)]

    weight_shapes = {"weight_0": 2, "weight_1": 2}
    qcnn_layer1 = qml.qnn.KerasLayer(qcnn1, weight_shapes, output_dim=n_qubits, name="qcnn1")
    qcnn_layer2 = qml.qnn.KerasLayer(qcnn2, weight_shapes, output_dim=n_qubits, name="qcnn2")

    q_split = tf.split(input, num_or_size_splits = splits, axis=-1)

    channels = n_qubits // splits
    if channels == 0:
        channels = 1

    patch_height = 2
    patch_width = 2

    q_total0, q_total1 = [], []
    for i in range(splits):
        patches = tf.image.extract_patches(
            images = q_split[i],
            sizes = [1, patch_height, patch_width, 1],
            strides = [1, patch_height, patch_width, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID"
        )

        batch_size, new_height, new_width, flattened_patch_size = patches.shape

        patches = layers.Reshape((new_height*new_width, flattened_patch_size))(patches)

        patches = tf.split(patches, num_or_size_splits = new_height*new_width , axis=1)

        patch_q0 = []
        patch_q1 = []
        for j in range(new_height*new_width):
            q0 = qcnn_layer1(patches[j])
            q1 = qcnn_layer2(patches[j])
            map1 = layers.Reshape((patch_height, patch_width, channels, 1))(q0) # CAREFUL HERE
            map2 = layers.Reshape((patch_height, patch_width, channels, 1))(q1) # CAREFUL HERE
            patch_q0.append(map1)
            patch_q1.append(map2)
        
        q_sub0 = tf.concat(patch_q0, axis=-1)
        q_sub1 = tf.concat(patch_q1, axis=-1)
        q_sub0 = layers.Reshape((patch_height, patch_width, channels, new_height, new_width))(q_sub0)
        q_sub1 = layers.Reshape((patch_height, patch_width, channels, new_height, new_width))(q_sub1)

        restored_fm0 = tf.concat(tf.split(q_sub0, new_height, axis=-2), axis=1)
        restored_fm0 = tf.concat(tf.split(restored_fm0, new_width, axis=-1), axis=2)
        restored_fm0 = tf.squeeze(restored_fm0, axis=[-1, -2])

        restored_fm1 = tf.concat(tf.split(q_sub1, new_height, axis=-2), axis=1)
        restored_fm1 = tf.concat(tf.split(restored_fm1, new_width, axis=-1), axis=2)
        restored_fm1 = tf.squeeze(restored_fm1, axis=[-1, -2])

        q_total0.append(restored_fm0)
        q_total1.append(restored_fm1)
    
    q_res0 = layers.concatenate(q_total0, axis=-1, name="q-resultant-0")
    q_res1 = layers.concatenate(q_total1, axis=-1, name="q-resultant-1")
    q_out0 = layers.Add()([input, q_res0])
    q_out1 = layers.Add()([input, q_res1])

    return q_out0, q_out1

def QuNet(input_shape, iou_fn, num_filters, splits):

    inputs = layers.Input(input_shape)

    skip_conn = []

    temp = inputs

    for i in num_filters:
        c, p = conv_block(temp, i)
        skip_conn.append(c)
        temp = p

    q0, q1 = QuFeX_4_2(temp, splits)

    temp = layers.concatenate([q0, q1], axis=-1)

    for i in num_filters[::-1]:
        c = convT_block(temp, i, skip_conn.pop())
        temp = c

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(temp)

    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[iou_fn])

    return model


