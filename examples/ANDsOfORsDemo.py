from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np

examples = 10000

#p = [[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9],
#		[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9]]

p = np.array([[0.1, 0.9],
		[0.9, 0.1]])

X_train = np.empty((examples, p.shape[1]), dtype=np.uint32)
Y_train = np.zeros(X_train.shape[0], dtype=np.uint32)

for i in range(examples):
	Y_train[i] = np.random.random() <= 0.5
	print(p.shape[Y_train[i]])
	for k in range(p.shape[Y_train[i]]):
		X_train[i, k] = np.random.random() <= p[Y_train[i], k]

	print(X_train[i,:])
	
average_accuracy = 0.0

for i in range(100):
	tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

	tm.fit(X_train, Y_train, epochs=200)

	print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

	average_accuracy += 100*(tm.predict(X_test) == Y_test).mean()

	print("Average Accuracy:", average_accuracy/(i+1))