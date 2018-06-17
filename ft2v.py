
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import gensim
import csv
import numpy

model = gensim.models.KeyedVectors.load_word2vec_format('/home/vamshi/GoogleNews-vectors-negative300.bin', binary=True)

s = model['sentence']
t = model['come']
try:			
	u = model['tyuh']
	print('value found')
except KeyError:
	print('oops! cant find it bruh!!')
frv = []
frt = []
with open('/home/vamshi/datavamshifinal.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	finalco = 0;
	for row in reader:
		if finalco < 50000:
			finalco=finalco + 1;
			s = row['Review']
			l = s.split()
			count = 0;
			for w in l:
				if count<100:
					flag = 1
					try:
						v = model[w]

					except KeyError:
						flag  = 0
					if flag==1:
						count=count+1;
						

					if flag==1 and count==1:
						fv =  v
					if flag==1 and count>1:
						fv = fv + v
			p = row['Rating']
			frv.append(fv)
			frt.append(int(p)-1)
			del fv
print('shape of single frv is ', frv[0].shape)
print('len of rating array is ', len(frt))
npfrv = numpy.vstack(frv)
npfrt = numpy.vstack(frt)
print('dimensiosn of npfrv, ', npfrv.shape)
print('dimensiosn of npfrt, ', npfrt.shape)




# Generate dummy data
#import numpy as np
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

x_train = npfrv[:24000]
y_train = npfrt[:24000]
x_test = npfrv[:24000]
y_test = npfrt[:24000]
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(248, activation='relu', input_dim=300))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
			              optimizer=sgd,
				                    metrics=['accuracy'])

model.fit(x_train, y_train,
		epochs=2000,
		batch_size=128)
score, acc = model.evaluate(x_test, y_test, batch_size=128)
print('model score:', score)
print('model accuracy:', acc)

