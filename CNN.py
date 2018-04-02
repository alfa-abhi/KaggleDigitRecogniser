import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
seed = 7
numpy.random.seed(seed)


df_train = pd.read_csv("train.csv")
y_train = df_train.iloc[0:, 0:1].values
X_train = df_train.iloc[0:, 1:].values
X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_test = pd.read_csv('test.csv').values.astype('float32')


X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
print(y_train.shape)
y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

scale = numpy.max(X_train)
X_train /= scale
X_test /= scale

mean = numpy.std(X_train)
X_train -= mean
X_test -= mean


def Deep_CNN_model():
    model = Sequential()
    model.add(Conv2D(30, (7, 7), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model_2 = Deep_CNN_model()
history_2 = model_2.fit(X_train, y_train, epochs=100, batch_size=100)

preds = model_2.predict_classes(X_test, verbose=0)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1, len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)


write_preds(preds, "digitRec.csv")
