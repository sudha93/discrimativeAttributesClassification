from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from numpy import asarray
from numpy import concatenate
from numpy import zeros,random 
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import spatial
import numpy as np

# This method uses word vector concatenation of all 3 words  
# It gets trained on validation set and tested on test set
# The result of this is submitted in the evaluation phase #final submission 
valid_data =  "/home/star/passion/discriminative/DiscriminAtt/training/validation.txt"
test_data = "/home/star/passion/discriminative/DiscriminAtt/test/test_triples.txt"

#validDataList contains the input data of validation set  
#validDataResult contains the output data of validation set 
validDataList = []
validDataResult = []
with open(valid_data) as file:
	for line in file:
		line = line.strip()
		temp = line.split(",")
		validDataList.append(temp[:3])
		validDataResult.append(int(temp[3]))
file.close()

#testDataList contains the input data of validation set  
testDataList = []
with open(test_data) as file:
	for line in file:
		line = line.strip()
		temp = line.split(",")
		testDataList.append(temp)
file.close()

# loading the glove embeddings into memory
# it contains 4 lakh word embeddings
f = open('/home/star/passion/puns/glove.6B/glove.6B.100d.txt')
# storing them in a dictionary with word as key, vector as value
embedding_index = {}
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float') 
	#coefs = values[1:]
	embedding_index[word] = coefs 
f.close() 

# For each valid data sample, take each word's embedding from
# the dict and concatenate them 
x_valid = []
#a = zeros(100)
a = random.rand(100)
# generates random values in the range (0,1) 
for item in validDataList:
	embedding_vector1 = embedding_index.get(item[0])
	if embedding_vector1 is None:
		embedding_vector1 = a 
	embedding_vector2 = embedding_index.get(item[1])
	if embedding_vector2 is None:
		embedding_vector2 = a
	embedding_vector3 = embedding_index.get(item[2])
	if embedding_vector3 is None:
		embedding_vector3 = a
	final = concatenate((embedding_vector1,embedding_vector2,embedding_vector3),axis=0) 
	#final.tolist()
	x_valid.append(final)

x_test = []
for i,item in enumerate(testDataList):
	embedding_vector1 = embedding_index.get(item[0])
	if embedding_vector1 is None:
		embedding_vector1 = a
		print i,'\n' 
	embedding_vector2 = embedding_index.get(item[1])
	if embedding_vector2 is None:
		embedding_vector2 = a
		print i,'\n'
	embedding_vector3 = embedding_index.get(item[2])
	if embedding_vector3 is None:
		embedding_vector3 = a
		print i,'\n'
	final = concatenate((embedding_vector1,embedding_vector2,embedding_vector3),axis=0) 
	#final.tolist()
	x_test.append(final)
	
	
#experiment 
embed = embedding_index.get('up')
summ = 0.0
for item in embed:
	summ = summ + item 
print summ

# model construction, training and testing
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(200,100,50), random_state=1, verbose=True)
print 'Training starts'
clf.fit(x_valid, validDataResult)

predicted = clf.predict(x_test)
est = clf.predict_proba(x_test)
#print predicted
#print y_test 
#result = precision_recall_fscore_support(testDataResult, predicted, average='macro')
# this function outputs precision, recall , F1 in that order
f = open("answer.txt","w")

for i,item in enumerate(testDataList) :
	string = item[0]+','+item[1]+','+item[2]+','+str(predicted[i])
	print >>f, string






















