from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from numpy import asarray
from numpy import concatenate
from numpy import zeros,random
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import spatial
import numpy as np
#from __future__ import print_function
# solving semeval 2018 task 10 using a simple fully connected 
# neural network using cross validation on validation set 
# we are not using training data 
# second method has cosine similarities attached to the word vectors  

valid_data =  "/home/star/passion/discriminative/DiscriminAtt/training/validation.txt"

validDataList = []
validDataResult = []
with open(valid_data) as file:
    for line in file:
        line = line.strip()
        temp = line.split(",")
        validDataList.append(temp[:3])
        validDataResult.append(int(temp[3]))
file.close()

# loading the glove 50 dimensional embeddings into memory
# it contains 4 lakh word embeddings
f = open('/home/star/passion/puns/glove.6B/glove.6B.100d.txt')
# storing them in a dictionary with word as key, vector as value
embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32') 
    embedding_index[word] = coefs 
f.close() 

# For each valid data sample , take each word's embedding from
# the dict and concatenate them 
x = []
#a = zeros(100)
a = random.rand(100)
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
    x.append(final)
#print x 
# concatenating cosine similarities also
xCosine =[]
for item in validDataList:
	embedding_vector1 = embedding_index.get(item[0])
	embedding_vector2 = embedding_index.get(item[1])
	embedding_vector3 = embedding_index.get(item[2])
	if (embedding_vector1 is None or embedding_vector2 is None or embedding_vector3 is None) :
		cos1 = 1.9
		cos2 = 1.8 
	else:
		cos1 = spatial.distance.cosine(np.array(embedding_vector1,dtype=float), np.array(embedding_vector3,dtype=float))
		cos2 = spatial.distance.cosine(np.array(embedding_vector2,dtype=float), np.array(embedding_vector3,dtype=float))
    
	if embedding_vector1 is None:
		embedding_vector1 = a 
	if embedding_vector2 is None:
		embedding_vector2 = a
	if embedding_vector3 is None:
		embedding_vector3 = a
	final = concatenate((embedding_vector1,embedding_vector2,embedding_vector3),axis=0) 
	final = np.append(final,cos1)
	final = np.append(final,cos2)
	xCosine.append(final)
	

 
# using pandas for indexing, as it gets ramdomly split and loses 
# the index..in pandas indexing starts from zero
x_pd = pd.DataFrame(x)     # change it to xCosine if you want the 2nd method 
y = pd.Series(validDataResult)
#print x_pd
#print ('\n'*3)
#print y 

# total no.of samples in validation.txt is 2722
x_train,x_test, y_train,y_test = train_test_split(x_pd,y, test_size = 0.3 ,random_state= 42)
indices = y_test.axes # stores the indices 
#print k[0][1]
#print y_test
#print y_test.index()
# the idea is to collect the indices and print the test output in the form 
# word1 word2 attribute yes/no (basically whether the predicted output is 
# correct or not)

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(200,100,50), random_state=1, verbose=True)
print 'Training starts'
clf.fit(x_train, y_train)
print 'Precision, Recall and F1 score in that order'
predicted = clf.predict(x_test)
est = clf.predict_proba(x_test)
#print predicted
#print y_test 
result = precision_recall_fscore_support(y_test, predicted, average='macro')
# this function outputs precision, recall , F1 in that order
print result 
print 'Accuracy'
acc = accuracy_score(y_test,predicted)
print acc
number = accuracy_score(y_test,predicted,normalize= False)
print number,'out of',len(y_test),'correctly predicted'
print len(y_train)

f = open("result.txt","w")

for i,item in enumerate(predicted) :
	original_index = indices[0][i] 
	sample = validDataList[original_index]
	if predicted[i] == y_test[original_index] :
		print >>f, original_index,sample,y_test[original_index],predicted[i],est[i],' Yes'	 
	else :
		print >>f, original_index,sample,y_test[original_index],predicted[i],est[i],' No'
f.close()
# could change the solver
#could use 100 dim vector
# couls use word embeddings trained on a different corpus 
# could play wround with no.of hidden layers & their dimensions 
# could use cross vaidation or shufflesplit( scikit function) on validation set itself 
# could train on training data and test on validation data 
# could try using keras 
# for 50 vector length, the hidden layer sizes are 100,50,20 ...68%
# for 100....(200,100,50).....72% for test = 0.3 & 73% for test= 0.2 seems good
# for 200....(300,150,50......72%
# for 300....(450,200,50.....almost 73%   seems good too 

# for 100....(200,100,50), test=0.3 , random state =42 , for x, f1 score = 0.74...for xCosine its 0.73


   
