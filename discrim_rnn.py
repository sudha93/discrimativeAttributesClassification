import untangle 
import nltk 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray 
from numpy import zeros
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Embedding 
from keras.layers import Flatten 
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers import LSTM, GRU
from sklearn.metrics import f1_score 

# this is solving semeval-2018 task 10 using rnn
# Idea is to concatenate word1 , word2 and attribute and give as input to rnn and 
# make it as sequence classification task 

train_data = "/home/star/passion/discriminative/DiscriminAtt-master/training/train.txt"

valid_data =  "/home/star/passion/discriminative/DiscriminAtt-master/training/validation.txt"
# storing the 3 words in a list in the same order
validDataList = []
validDataResult = []
with open(valid_data) as file:
    for line in file:
        line = line.strip()
        temp = line.split(",")
        validDataList.append(temp[:3])
        validDataResult.append(int(temp[3]))
file.close()
#print len(validDataList) 
#print trainDataResult

trainDataList = []
trainDataResult = []
with open(train_data) as file:
    for line in file:
        line = line.strip()
        temp = line.split(",")
        trainDataList.append(temp[:3])
        trainDataResult.append(int(temp[3]))
file.close()
#print len(trainDataList)
#print len(trainDataResult)
# the README says 17782 samples in train_data, but we got 17547,even after verfication 
sentenceListTrain = []
for item in trainDataList:
    string = ' '.join(item)
    sentenceListTrain.append(string)

sentenceListValid = []
for item in validDataList:
    string = ' '.join(item)
    sentenceListValid.append(string)

# Here we are giving ids to all the words in both train and valid data 
# as we are sending all the sentences to tokenizer 
# we can change this and try 
finalList = sentenceListTrain + sentenceListValid
# startting tokenier 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(finalList)
vocab_size = len(tokenizer.word_index)
#print vocab_size
encoded_sent_train= tokenizer.texts_to_sequences(sentenceListTrain)
encoded_sent_valid = tokenizer.texts_to_sequences(sentenceListValid)

# tokenizer starts ids from 1 , but we want them from zero 
# so we subtract one from all the ids 
for i , item in enumerate(encoded_sent_train):
    for j, subitem in enumerate(item):
        encoded_sent_train[i][j] -= 1

for i , item in enumerate(encoded_sent_valid):
    for j, subitem in enumerate(item):
	encoded_sent_valid[i][j] -= 1
#print encoded_sent_valid

# our inputs to model , all 4 of them should be in the format of numpy arrays 
# we use pad_sequences just so that it ouputs numopy arrays ,though we are
# not actually doing any padding  
x_train = pad_sequences(encoded_sent_train,maxlen=3,padding='pre',truncating= 'post', value= 0.0)
x_test = pad_sequences(encoded_sent_valid,maxlen=3,padding='pre',truncating= 'post', value= 0.0)
#x_train = encoded_sent_train
#print x_train
y_train = asarray(trainDataResult)
#x_test = encoded_sent_valid 
y_test = asarray(validDataResult)
#print y_test
#print vocab_size

# loading the glove 50 dimensional embeddings into memory
# it contains 4 lakh word embeddings
f = open('/home/star/passion/puns/glove.6B/glove.6B.50d.txt')
embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32') 
    embedding_index[word] = coefs 
f.close() 
#print('Loaded %s word vectors.' % len(embedding_index))
# create a weight matrix for words in training docs
#print tokenizer.word_index.items()
embedding_matrix = zeros((vocab_size, 50))
for i, word in enumerate (tokenizer.word_index.items()):
    embedding_vector = embedding_index.get(word[0])
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector 
# here each row corresponds to a word embedding 

# model construction 
model = Sequential()
model.add(Embedding(vocab_size,50,weights = [embedding_matrix], input_length=3,trainable = True))
# I can use trainable = True as well , no harm 
#model.add(Bidirectional(GRU(40,return_sequences = False , stateful = False)))
model.add(GRU(40,return_sequences = False , stateful = False))
#model.add(Flatten())
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=16 , epochs = 25)
loss,score = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)
#print y_pred.round() 
print 'accuracy : ',score*100

print f1_score(y_test, y_pred.round(), labels=None, pos_label=1, average='macro', sample_weight=None) 

# accuracy is 49 0r 50 and f1 score is around 0.44


























