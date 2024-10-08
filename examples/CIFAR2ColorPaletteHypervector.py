import numpy as np
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from numba import jit

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

scaling = 1.0

hypervector_size = 256

bits = 5

resolution = 8

animals = np.array([2, 3, 4, 5, 6, 7])

ensembles = 5
epochs = 250

examples = 1000

max_included_literals = 32

clauses = 8000
T = int(clauses * 0.75)
s = 10.0

#clauses = 2000
#T = 5000
#s = 1.5

@jit(nopython=True)
def count_nonzero_hypervector(hypervector, encoding, X):
        nonzero_count = 0
        for i in range(X.shape[0]):
                for x in range(X.shape[1]):
                        for y in range(X.shape[2]):
                                hypervector[:] = 0

                                for k1 in range(X[i, x, y, 0] // (256 // resolution)):
                                        for k2 in range(X[i, x, y, 1] // (256 // resolution)):
                                                for k3 in range(X[i, x, y, 2] // (256 // resolution)):
                                                        roll = k1 + 11 * k2
                                                        code = encoding[k3]
                                                        for bit in code:
                                                                hypervector[((bit + roll) % hypervector_size)] = 1
                                nonzero_count += hypervector.sum()
        return np.uint64(nonzero_count)

@jit(nopython=True)
def produce_hypervectors(hypervector, hypervector_size, encoding, X, indptr, indices, data):
        nonzero_count = 0
        for i in range(X.shape[0]):
                indptr[i] = nonzero_count
                for x in range(X.shape[1]):
                        for y in range(X.shape[2]):
                                hypervector[:] = 0
                                for k1 in range(X[i, x, y, 0] // (256 // resolution)):
                                        for k2 in range(X[i, x, y, 1] // (256 // resolution)):
                                                for k3 in range(X[i, x, y, 2] // (256 // resolution)):
                                                        roll = k1 + 11 * k2
                                                        code = encoding[k3]
                                                        for bit in code:
                                                                hypervector[((bit + roll) % hypervector_size)] = 1
                                nonzero = hypervector.nonzero()[0]

                                for bit in nonzero:
                                        indices[nonzero_count] = np.uint64(x * X.shape[2] * hypervector_size + y * hypervector_size + bit)
                                        data[nonzero_count] = 1
                                        nonzero_count += 1
        indptr[X.shape[0]] = nonzero_count

        return nonzero_count

indexes = np.arange(hypervector_size, dtype=np.uint64)
encoding = np.zeros((resolution, bits), dtype=np.uint64)
for i in range(resolution):
        encoding[i] = np.random.choice(indexes, size=(bits))

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org#[0:examples]
X_test_org = X_test_org#[0:examples]
Y_train = Y_train.reshape(Y_train.shape[0])#[0:examples]
Y_test = Y_test.reshape(Y_test.shape[0])#[0:examples]

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

hypervector = np.zeros(hypervector_size, dtype=np.uint64)

print("Training Data")

start_time = time()
nonzero_count_train = count_nonzero_hypervector(hypervector, encoding, X_train_org)
X_train_indptr = np.empty(X_train_org.shape[0]+1, dtype=np.uint64)
X_train_indices = np.empty(nonzero_count_train, dtype=np.uint64)
X_train_data = np.empty(nonzero_count_train, dtype=np.uint32)

print(nonzero_count_train)

produce_hypervectors(hypervector, hypervector_size, encoding, X_train_org, X_train_indptr, X_train_indices, X_train_data)

X_train = csr_matrix((X_train_data, X_train_indices, X_train_indptr))

stop_time = time()

print(stop_time - start_time)

print("Testing Data")

nonzero_count_test = count_nonzero_hypervector(hypervector, encoding, X_test_org)
X_test_indptr = np.empty(X_test_org.shape[0]+1, dtype=np.uint64)
X_test_indices = np.empty(nonzero_count_test, dtype=np.uint64)
X_test_data = np.empty(nonzero_count_test, dtype=np.uint32)

produce_hypervectors(hypervector, hypervector_size, encoding, X_test_org, X_test_indptr, X_test_indices, X_test_data)

X_test = csr_matrix((X_test_data, X_test_indices, X_test_indptr))

print(X_test.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d.txt" % (s, clauses, T, scaling), "w+")
indexes = np.arange(X_train.shape[0], dtype=np.uint32)
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, hypervector_size), (3, 3), max_included_literals=max_included_literals)

        for epoch in range(epochs):
                example_selection = np.random.choice(indexes, size=1000, replace=False)

                start_training = time()
                tm.fit(X_train[example_selection], Y_train[example_selection], epochs=1, incremental=True)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()

                print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, result_test, result_train, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
