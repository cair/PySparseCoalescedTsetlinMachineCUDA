import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine

#target_words = ['masterpiece', 'brilliant', 'comedy', 'scary', 'funny', 'hate', 'love', 'awful', 'terrible']

#target_words = ['awful', 'scary', 'brilliant']

target_words = ['awful', 'terrible', 'lousy', 'abysmal', 'crap', 'outstanding', 'brilliant', 'excellent', 'superb', 'magnificent', 'marvellous', 'truck', 'plane', 'car', 'cars', 'motorcycle',  'scary', 'frightening', 'terrifying', 'horrifying', 'funny', 'comic', 'hilarious', 'witty']

#target_words = ['awful', 'terrible', 'brilliant']

examples = 50000

clause_weight_threshold = 0

context_size = 25

clause_drop_p = 0.0

max_included_literals = 3

number_of_words = 2000

factor = 20
clauses = factor*20
T = factor*40
s = 5.0

NUM_WORDS=10000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

id_to_word = {value:key for key,value in word_to_id.items()}

training_documents = []
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id].lower())

	training_documents.append(terms)

def tokenizer(s):
	return s

vectorizer_X = CountVectorizer(tokenizer=tokenizer, lowercase=False, max_features=NUM_WORDS, binary=True)

X_train_csr = vectorizer_X.fit_transform(training_documents)
X_train_csc = X_train_csr.tocsc()

feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]

# target_words = []
# for word in feature_names:
# 	word_id = vectorizer_X.vocabulary_[word]

# 	target_words.append(word)
# 	if len(target_words) == number_of_words:
# 		break

target_ids_list = []
for target_word in target_words:
	target_ids_list.append(vectorizer_X.vocabulary_[target_word])
target_ids = np.array(target_ids_list)

X_train = lil_matrix((examples, number_of_features), dtype=np.uint32)
Xi = np.zeros(number_of_features, dtype=np.uint32)
Y_train = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	if i % 1000 == 0:
		print(i)
	target_class = np.random.choice(np.arange(target_ids.shape[0]))
	target_rows = X_train_csc.indices[X_train_csc.indptr[target_ids[target_class]]:X_train_csc.indptr[target_ids[target_class]+1]]
	Xi[:] = 0
	for c in range(context_size):
		Xi = np.logical_or(Xi, X_train_csr[np.random.choice(target_rows)].toarray().reshape(-1))
	X_train[i] = Xi
	Y_train[i] = target_class
X_train = X_train.tocsc()
X_train[:,target_ids] = 0

tm = MultiClassTsetlinMachine(clauses, T, s, append_negated=False)

for e in range(40):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	print("\nEpoch #%d\n" % (e+1))

	weights = tm.get_state()[1].reshape((len(target_words), -1))

	print("Clauses\n")

	for j in range(clauses):
		print("Clause #%d " % (j), end=' ')
		for i in range(len(target_words)):
			print("%s:W%d " % (target_words[i], weights[i,j]), end=' ')
		print()

		# l = []
		# for k in range(tm.clause_bank.number_of_literals):
		# 	if tm.get_ta_action(j, k) == 1:
		# 		if k < tm.number_of_features//2:
		# 			l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
		# 		else:
		# 			l.append("¬%s(%d)" % (feature_names[k-tm.number_of_features//2], tm.clause_bank.get_ta_state(j, k)))
		# print(" ∧ ".join(l))

	profile = np.empty((len(target_words), clauses))
	for i in range(len(target_words)):
		profile[i,:] = np.where(weights[i,:] >= clause_weight_threshold, weights[i,:], 0)

	similarity = cosine_similarity(profile)

	print("\nWord Similarity\n")

	for i in range(len(target_words)):
		print(target_words[i], end=': ')
		sorted_index = np.argsort(-1*similarity[i,:])
		for j in range(1, 10):
			print("%s(%.2f) " % (target_words[sorted_index[j]], similarity[i,sorted_index[j]]), end=' ')
		print()

	print("\nTraining Time: %.2f" % (stop_training - start_training))

