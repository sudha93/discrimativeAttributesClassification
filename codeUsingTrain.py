
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from numpy import asarray
from numpy import concatenate
from numpy import zeros,random 
from sklearn.model_selection import train_test_split
import pandas as pd


# This is the method using multilayer perceptron 
# Training on train data and testing on validation data 
# and submittimg it for practice submissions 


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
    coefs = asarray(values[1:], dtype='float32') 
    embedding_index[word] = coefs 
f.close() 

# For each valid data sample , take each word's embedding from
# the dict and concatenate them 
x_valid = []
# using a list of random elements for the not found embeddings 
a = random.rand(100)

for i,item in enumerate(validDataList):
    embedding_vector1 = embedding_index.get(item[0])
    if embedding_vector1 is None:
        embedding_vector1 = a 
        print i,'valid\n'
    embedding_vector2 = embedding_index.get(item[1])
    if embedding_vector2 is None:
        embedding_vector2 = a
        print i,'valid\n'
    embedding_vector3 = embedding_index.get(item[2])
    if embedding_vector3 is None:
        embedding_vector3 = a
        print i,'valid\n'
    final = concatenate((embedding_vector1,embedding_vector2,embedding_vector3),axis=0) 
    x_valid.append(final)

x_train = []
for i,item in enumerate(trainDataList):
    embedding_vector1 = embedding_index.get(item[0])
    if embedding_vector1 is None:
        embedding_vector1 = a 
        print i,'train\n'
    embedding_vector2 = embedding_index.get(item[1])
    if embedding_vector2 is None:
        embedding_vector2 = a
        print i,'train\n'
    embedding_vector3 = embedding_index.get(item[2])
    if embedding_vector3 is None:
        embedding_vector3 = a
        print i,'train\n'
    final = concatenate((embedding_vector1,embedding_vector2,embedding_vector3),axis=0) 
    x_train.append(final)

#  Model construction and training 
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(200,100,50), random_state=1, verbose=True)
print 'Training starts'
clf.fit(x_train, trainDataResult)
print 'Precision, Recall and F1 score in that order'
predicted = clf.predict(x_valid)
result = precision_recall_fscore_support(validDataResult, predicted, average='macro')
# This function outputs precision, recall , F1 in that order
print result 
print 'Accuracy'
acc = accuracy_score(validDataResult,predicted)
print acc
number = accuracy_score(validDataResult,predicted,normalize= False)
print number,'out of',len(validDataResult),'correctly predicted'
print len(trainDataResult)

#printing the results to a file in the submission format
f = open("answer.txt","w")

for i,item in enumerate(validDataList) :
	string = item[0]+','+item[1]+','+item[2]+','+str(predicted[i])
	print >>f, string



# gave 0.51 accuracy for 100 size















