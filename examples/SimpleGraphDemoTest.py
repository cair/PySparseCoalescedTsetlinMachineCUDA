from PySparseCoalescedTsetlinMachineCUDA.graph import Graph
from PySparseCoalescedTsetlinMachineCUDA.graph import Graphs
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

graphs_train = Graphs()
Y_train = np.empty(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
    sequence_graph = Graph()
    
    # Select class
    Y_train[i] = np.random.randint(args.number_of_classes) 

    nodes = args.max_sequence_length
    for j in range(nodes):
        sequence_graph.add_node(j)

    j = np.random.randint(nodes)

    if Y_train[i] == 0:
        sequence_graph.add_feature(j, 'A')
    else:
        sequence_graph.add_feature(j, 'B')

    graphs_train.add(sequence_graph)

Y_train = np.where(np.random.rand(args.number_of_examples) < args.noise, 1 - Y_train, Y_train)  # Adds noise

graphs_train.encode(hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)

print(graphs_train.hypervectors)
print(graphs_train.edge_type_id)
print(graphs_train.node_count)

tm = MultiClassGraphTsetlinMachine(args.number_of_clauses, args.T, args.s, (1, args.max_sequence_length, args.hypervector_size), (1, 1), max_included_literals=args.max_included_literals)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_train) == Y_train).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(hypervector_size * 2):
            if tm.ta_action(i, k):
                if k < hypervector_size:
                    l.append(" x%d" % (k))
                else:
                    l.append(" NOT x%d" % (k - hypervector_size))
        print(" AND ".join(l))


start_training = time()
tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100*(tm.predict(graphs_train) == Y_train).mean()
stop_testing = time()

result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print(graphs_train.hypervectors)