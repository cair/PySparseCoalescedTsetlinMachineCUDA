from PySparseCoalescedTsetlinMachineCUDA.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--number-of-clauses", default=2, type=int)
    parser.add_argument("--T", default=16, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--hypervector_size", default=16, type=int)
    parser.add_argument("--hypervector_bits", default=1, type=int)
    parser.add_argument("--noise", default=0.0, type=float)
    parser.add_argument("--number-of-examples", default=1000, type=int)
    parser.add_argument("--max-sequence-length", default=1000, type=int)
    parser.add_argument("--number-of-classes", default=2, type=int)
    parser.add_argument("--max-included-literals", default=1, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

number_of_nodes = np.repeat(args.max_sequence_length, args.number_of_examples).astype(np.uint32)

graphs_train = Graphs(number_of_nodes, number_of_symbols=2, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)
Y_train = np.empty(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
    # Select class
    Y_train[i] = np.random.randint(args.number_of_classes) 

    for j in range(args.max_sequence_length):
        if np.random.randint(2) == 0:
            graphs_train.add_node_feature(i, j, 0)
        else:
            graphs_train.add_node_feature(i, j, 1)

    j = np.random.randint(args.max_sequence_length)
    if Y_train[i] == 0:
        graphs_train.add_node_feature(i, j, 0)
    else:
        graphs_train.add_node_feature(i, j, 1)

Y_train = np.where(np.random.rand(args.number_of_examples) < args.noise, 1 - Y_train, Y_train)  # Adds noise

graphs_train.encode()

print(graphs_train.hypervectors)

graphs_test = Graphs(number_of_nodes, init_with=graphs_train)
Y_test = np.empty(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
    # Select class
    Y_test[i] = np.random.randint(args.number_of_classes) 

    for j in range(args.max_sequence_length):
        if np.random.randint(2) == 0:
            graphs_test.add_node_feature(i, j, 0)
        else:
            graphs_test.add_node_feature(i, j, 1)

    j = np.random.randint(args.max_sequence_length)
    if Y_test[i] == 0:
        graphs_test.add_node_feature(i, j, 0)
    else:
        graphs_test.add_node_feature(i, j, 1)

graphs_test.encode()

print(graphs_test.hypervectors)

graphs_train.Y = Y_train
graphs_test.Y = Y_test

tm = MultiClassGraphTsetlinMachine(args.number_of_clauses, args.T, args.s, (1, args.max_sequence_length, args.hypervector_size), (1, 1), max_included_literals=args.max_included_literals)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))
        print(" AND ".join(l))


start_training = time()
tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
stop_testing = time()

result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

print("%.2f %.2f %.2f %.2f" % (result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print(graphs_train.hypervectors)