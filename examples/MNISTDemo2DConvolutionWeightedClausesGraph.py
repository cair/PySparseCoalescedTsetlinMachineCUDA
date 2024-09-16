from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
from time import time
from MultiClassConvolutionalTsetlinMachine2D import graphs

from keras.datasets import mnist

factor = 10

s = 10.0

T = int(factor*25*100)

ensembles = 10
epochs = 250

patch_size = 10

X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=2500, type=int)
    parser.add_argument("--s", default=1.5, type=float)
    parser.add_argument("--hypervector_size", default=128, type=int)
    parser.add_argument("--hypervector_bits", default=2, type=int)
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

graphs_train = Graphs()
for i in range(X_train.shape[0]):
    if i % 1000 == 0:
        print(i, X_train.shape[0])
    image_graph = Graph() 
    windows = view_as_windows(X_train[i,:,:], (10, 10))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = str(q) + ":" + str(r)
                image_graph.add_node(node_id)
                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    image_graph.add_feature(node_id, str(k))
                image_graph.add_feature(node_id, 'c:'+str(q))
                image_graph.add_feature(node_id, 'r:'+str(r))
    graphs_train.add(image_graph)
graphs_train.encode(hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)

print("Training data produced")

graphs_test = Graphs()
for i in range(X_test.shape[0]):
    image_graph = Graph() 
    windows = view_as_windows(X_test[i,:,:], (10, 10))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = str(q) + ":" + str(r)
                image_graph.add_node(node_id)
                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    image_graph.add_feature(node_id, str(k))
                image_graph.add_feature(node_id, 'c:'+str(q))
                image_graph.add_feature(node_id, 'r:'+str(r))
    graphs_test.add(image_graph)
graphs_test.encode(hypervectors = graphs_train.hypervectors, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)

print(graphs_train.hypervectors)
print(graphs_train.edge_type_id)
print(graphs_train.node_count)

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, int(factor*2000), T,  patch_size), "w+")

for e in range(ensembles):
	tm = MultiClassConvolutionalTsetlinMachine2D(int(factor*2000), T, s, (28, 28, 1), (patch_size, patch_size))

	for i in range(epochs):
	    start_training = time()
	    tm.fit(graphs_train.X, Y_train, epochs=1, incremental=True)
	    stop_training = time()

	    start_testing = time()
	    result_test = 100*(tm.predict(graphs_test.X) == Y_test).mean()
	    stop_testing = time()

	    result_train = 100*(tm.predict(graphs_train.X) == Y_train).mean()

	    print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
	    print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
	    f.flush()
f.close()
