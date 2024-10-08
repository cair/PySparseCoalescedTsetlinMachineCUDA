from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar10

animals = np.array([2, 3, 4, 5, 6, 7])

max_included_literals = 32
clauses = 8000
T = int(clauses * 0.75)
s = 10.0
patch_size = 3
resolution = 8
number_of_state_bits_ta = 8
literal_drop_p = 0.0
q=0.1

epochs = 250
ensembles = 5

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
                   dtype=np.uint8)
for z in range(resolution):
    X_train[:, :, :, :, z] = X_train_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution),
                  dtype=np.uint8)
for z in range(resolution):
    X_test[:, :, :, :, z] = X_test_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3 * resolution)).reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3 * resolution)).reshape((X_test.shape[0], -1))

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

f = open("cifar2_%.1f_%d_%d_%d_%.2f_%d_%d.txt" % (
s, clauses, T, patch_size, literal_drop_p, resolution, max_included_literals), "w+")
for ensemble in range(ensembles):
    tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, 3*resolution), (patch_size, patch_size), max_included_literals=max_included_literals, q=q)

    for epoch in range(epochs):
        start_training = time()
        tm.fit(X_train, Y_train, incremental=True, epochs=1)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(X_train) == Y_train).mean()

        print("%d %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, result_train, result_test, stop_training - start_training, stop_testing - start_testing))
        print("%d %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, result_train, result_test, stop_training - start_training, stop_testing - start_testing),
              file=f)
        f.flush()
f.close()
