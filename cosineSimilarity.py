from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from numpy import asarray
from numpy import concatenate
from numpy import zeros
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy import spatial 
#from numpy.ndarray import flatten
# This is the method using cosine similarity (MLP model)
# training on train data and testing on validation data 
# and submittimg it for practice submissions 
#2nd method just uses the vaidatipn data split and trains over it 
# tests over test_split of validation data 
# 3rd method uses both word vectors and cosine similarity 
# only using validation data split 

valid_data =  "/home/star/passion/discriminative/DiscriminAtt/training/validation.txt"
train_data =  "/home/star/passion/discriminative/DiscriminAtt/training/train.txt"

validDataList = []
validDataResult = []
with open(valid_data) as file:
	for line in file:
		line = line.strip()
		temp = line.split(",")
        validDataList.append(temp[:3])
        validDataResult.append(int(temp[3]))
file.close()


trainDataList = []
trainDataResult = []
with open(train_data) as file:
    for line in file:
        line = line.strip()
        temp = line.split(",")
        trainDataList.append(temp[:3])
        trainDataResult.append(int(temp[3]))
file.close()

# loading the glove 50 dimensional embeddings into memory
# it contains 4 lakh word embeddings
f = open('/home/star/passion/puns/glove.6B/glove.6B.100d.txt')
# storing them in a dictionary with word as key, vector as value
embedding_index = {}
for line in f:
	values = line.split()
	word = values[0]
	#coefs = asarray(values[1:], dtype='float32') 
	coefs = values[1:]
	embedding_index[word] = coefs 
f.close() 
#print embedding_index
# for each train and valid data sample , calculate its cosine similarity 
# between w1 and A , w2 and A 
# range is [0,2] with 0 being exactly similar & 2 being exactly opposite 
# lesser the vlaue more similar they are 
# we concatenate both cosine similarities and send it as input to MLP

# for the training set 
x_train = []
for item in trainDataList :
    final = []
    embedding_vector1 = embedding_index.get(item[0])
    #print embedding_vector1,'first\n'
    embedding_vector2 = embedding_index.get(item[1])
    #print embedding_vector2,'second\n'
    embedding_vector3 = embedding_index.get(item[2])
    #print embedding_vector3,'third\n'
    if (embedding_vector1 is not None and embedding_vector2 is not None and embedding_vector3 is not None) :
		cos1 = spatial.distance.cosine(np.array(embedding_vector1, dtype=float), np.array(embedding_vector3, dtype=float))
		cos2 = spatial.distance.cosine(np.array(embedding_vector2, dtype=float), np.array(embedding_vector3, dtype=float))
    final.append(cos1)
    final.append(cos2)
    x_train.append(final)

#print x_train
k = 0 	 
x_valid = []
# for the validation set 
for i,item in enumerate(validDataList) :
    list1 = []
    embedding_vector1 = embedding_index.get(item[0])
    embedding_vector2 = embedding_index.get(item[1])
    embedding_vector3 = embedding_index.get(item[2])
    if (embedding_vector1 is None or embedding_vector2 is None or embedding_vector3 is None) :
         cos1 = 1.9
         cos2 = 1.8 
    else :
        cos1 = spatial.distance.cosine(np.array(embedding_vector1,dtype=float), np.array(embedding_vector3,dtype=float))
        cos2 = spatial.distance.cosine(np.array(embedding_vector2,dtype=float), np.array(embedding_vector3,dtype=float))
#print k 
    list1.append(cos1)
    list1.append(cos2)
    x_valid.append(list1)

#first model construction 
clf1 = MLPClassifier(solver='adam', hidden_layer_sizes=(32,16,8,4), random_state=1, verbose=True)
#print 'First model training starts'
clf1.fit(x_train,trainDataResult)
predicted = clf1.predict(x_valid)
#print 'Precision, Recall and F1 score in that order'
result = precision_recall_fscore_support(validDataResult, predicted, average='macro')
#print result 
#print 'Accuracy'
acc = accuracy_score(validDataResult,predicted)
#print acc

x_pd = pd.DataFrame(x_valid)
y = pd.Series(validDataResult)
# second model construction 
x1_train,x_test, y_train,y_test = train_test_split(x_pd,y, test_size = 0.3 ,random_state= 42)
indices = y_test.axes  #storing the indices 
clf2 = MLPClassifier(solver='adam', hidden_layer_sizes=(16,4), random_state=1, verbose=True)
print 'Second model training starts' 
clf2.fit(x1_train,y_train)
predicted = clf2.predict(x_test)
est = clf2.predict_proba(x_test)         #has prob of each class 
#print est,'\n'
#print predicted,'\n' 
print 'Precision, Recall and F1 score in that order'
result = precision_recall_fscore_support(y_test, predicted, average='macro')
print result 
print 'Accuracy'
acc = accuracy_score(y_test,predicted)
print acc

#(32,16,8,4) gives 62 for model 2 
'''
f = open("resultCosine.txt","w")
for i,item in enumerate(predicted) :
	original_index = indices[0][i] 
	sample = validDataList[original_index]
	if predicted[i] == y_test[original_index] :
		print >>f, original_index,sample,y_test[original_index],predicted[i],est[i],' Yes'	 
	else :
		print >>f, original_index,sample,y_test[original_index],predicted[i],est[i],' No'
f.close()

'''





