# Copyright (c) 2024 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import numpy as np

import PySparseCoalescedTsetlinMachineCUDA.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.sparse import csr_matrix
import sys

from time import time

g = curandom.XORWOWRandomNumberGenerator() 

class CommonTsetlinMachine():
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			depth=1,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		print("Initialization of sparse structure.")

		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.q = q
		self.max_included_literals = max_included_literals
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.append_negated = append_negated
		self.depth = depth
		self.grid = grid
		self.block = block

		self.graphs_signature_train = np.array([])
		self.graphs_signature_test = np.array([])
		self.encoded_Y = np.array([])
		
		self.ta_state = np.array([])
		self.clause_weights = np.array([])

		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.encode = mod_encode.get_function("encode")
		self.encode.prepare("PPPiiii")

		self.restore = mod_encode.get_function("restore")
		self.restore.prepare("PPPiiii")

		self.produce_autoencoder_examples= mod_encode.get_function("produce_autoencoder_example")
		self.produce_autoencoder_examples.prepare("PPiPPiPPiPPiiii")

		self.initialized = False

	def allocate_gpu_memory(self):
		self.ta_state_gpu = cuda.mem_alloc(self.depth*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs*4)

	def ta_action(self, depth, clause, ta):
		if np.array_equal(self.ta_state, np.array([])):
			self.ta_state = np.empty(self.depth*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
		ta_state = self.ta_state.reshape((self.depth, self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))

		return (ta_state[clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0

	def get_state(self):
		if np.array_equal(self.clause_weights, np.array([])):
			self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
			self.clause_weights = np.empty(self.number_of_outputs*self.number_of_clauses, dtype=np.int32)
			cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
		return((self.ta_state, self.clause_weights, self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_nodes, self.depth, self.number_of_state_bits, self.number_of_ta_chunks, self.append_negated, self.min_y, self.max_y))

	def set_state(self, state):
		self.number_of_outputs = state[2]
		self.number_of_clauses = state[3]
		self.number_of_literals = state[4]
		self.number_of_nodes = state[7]
		self.depth = state[8]
		self.number_of_state_bits = state[9]
		self.number_of_ta_chunks = state[10]
		self.append_negated = state[11]
		self.min_y = state[12]
		self.max_y = state[13]
		
		self.ta_state_gpu = cuda.mem_alloc(self.depth*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		cuda.memcpy_htod(self.ta_state_gpu, state[0])
		cuda.memcpy_htod(self.clause_weights_gpu, state[1])

		self.graphs_signature_train = np.array([])
		self.graphs_signature_test = np.array([])

		self.encoded_Y = np.array([])

		self.ta_state = np.array([])
		self.clause_weights = np.array([])

	def _init(self, graphs):
		self.number_of_features = graphs.hypervector_size
		if self.append_negated:
			self.number_of_literals = self.number_of_features*2
		else:
			self.number_of_literals = self.number_of_features

		if self.max_included_literals == None:
			self.max_included_literals = self.number_of_literals

		self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)

		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define LITERALS %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define S %f
#define THRESHOLD %d
#define Q %f
#define MAX_INCLUDED_LITERALS %d
#define NEGATIVE_CLAUSES %d
#define NUMBER_OF_EXAMPLES %d
""" % (self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.q, self.max_included_literals, self.negative_clauses, graphs.number_of_nodes.shape[0])

		mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
		self.prepare = mod_prepare.get_function("prepare")

		self.allocate_gpu_memory()

		mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
		self.update = mod_update.get_function("update")
		self.update.prepare("PPPiPPPi")

		self.evaluate_update = mod_update.get_function("evaluate")
		self.evaluate_update.prepare("PPiPPi")

		mod_evaluate = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
		self.evaluate = mod_evaluate.get_function("evaluate")
		self.evaluate.prepare("PPiPPi")

		self.initialized = True

	def _init_fit(self, graphs, encoded_Y, incremental):
		if not self.initialized:
			self._init(graphs)
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()
		elif incremental == False:
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

		if not np.array_equal(self.graphs_signature_train, graphs.signature):
			self.graphs_signature_train = graphs.signature

			self.encoded_X_train_gpu = cuda.mem_alloc(graphs.X.nbytes)
			cuda.memcpy_htod(self.encoded_X_train_gpu, graphs.X)

		if not np.array_equal(self.encoded_Y, encoded_Y):
			self.encoded_Y = encoded_Y

			self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

	def _fit(self, graphs, encoded_Y, epochs=100, incremental=False):
		self._init_fit(graphs, encoded_Y, incremental)

		for epoch in range(epochs):
			for e in range(graphs.number_of_nodes.shape[0]):
				class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
				cuda.memcpy_htod(self.class_sum_gpu, class_sum)

				self.evaluate_update.prepared_call(
					self.grid,
					self.block,
					self.ta_state_gpu,
					self.clause_weights_gpu,
					np.int32(graphs.number_of_nodes[e]),
					self.class_sum_gpu,
					self.encoded_X_train_gpu,
					np.int32(e)
				)
				cuda.Context.synchronize()

				self.update.prepared_call(
					self.grid,
					self.block,
					g.state,
					self.ta_state_gpu,
					self.clause_weights_gpu,
					np.int32(graphs.number_of_nodes[e]),
					self.class_sum_gpu,
					self.encoded_X_train_gpu,
					self.encoded_Y_gpu,
					np.int32(e)
				)
				cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score(self, graphs):
		if not self.initialized:
			print("Error: Model not trained.")
			sys.exit(-1)

		if not np.array_equal(self.graphs_signature_test, graphs.signature):
			self.graphs_signature_test = graphs.signature

			self.encoded_X_test_gpu = cuda.mem_alloc(graphs.X.nbytes)
			cuda.memcpy_htod(self.encoded_X_test_gpu, graphs.X)
        
		class_sum = np.zeros((graphs.number_of_nodes.shape[0], self.number_of_outputs), dtype=np.int32)
		for e in range(graphs.number_of_nodes.shape[0]):
			cuda.memcpy_htod(self.class_sum_gpu, class_sum[e,:])

			self.evaluate.prepared_call(
				self.grid,
				self.block,
				self.ta_state_gpu,
				self.clause_weights_gpu,
				np.int32(graphs.number_of_nodes[e]),
				self.class_sum_gpu,
				self.encoded_X_test_gpu,
				np.int32(e)
			)
			cuda.Context.synchronize()

			cuda.memcpy_dtoh(class_sum[e,:], self.class_sum_gpu)

		return class_sum
	
class MultiClassGraphTsetlinMachine(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			dim,
			patch_dim,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			depth=1,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(
			number_of_clauses,
			T,
			s,
			q=q,
			max_included_literals=max_included_literals,
			boost_true_positive_feedback=boost_true_positive_feedback,
			number_of_state_bits=number_of_state_bits,
			append_negated=append_negated,
			depth=depth,
			grid=grid,
			block=block
		)
		self.dim = dim
		self.patch_dim = patch_dim
		self.negative_clauses = 1

	def fit(self, graphs, Y, epochs=100, incremental=False):
		self.number_of_outputs = int(np.max(Y) + 1)
	
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(graphs, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, graphs):
		return self._score(graphs)

	def predict(self, graphs):
		return np.argmax(self.score(graphs), axis=1)
