import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

x = open(r'/content/இசை_ஜீனியஸ்_ராஜா_ரவி_நடராஜன்.txt', 'rb').read().decode(encoding='utf-8')
x = x.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')

tokens = Tokenizer()
tokens.fit_on_texts([x])
pickle.dump(tokens, open('Ilayaraja_book_tokens.pkl', 'wb'))
dictionary = tokens.word_index

x_n_grams = []
for line in x.split('.'):
    line_tokens = tokens.texts_to_sequences([line])[0]
    for i in range(1, len(line_tokens)):
        n_grams = line_tokens[:i+1]
        x_n_grams.append(n_grams)

max_line_len = max([len(i) for i in x_n_grams])      
training_data = np.array(pad_sequences(x_n_grams, maxlen=max_line_len, padding='pre'))
train_X = training_data[:, :-1]
train_y = training_data[:, -1]      

total_words = len(dictionary) + 1

y = np.array(tf.keras.utils.to_categorical(train_y, num_classes=total_words))  

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_line_len-1)) 
model.add(LSTM(150)) 
model.add(Dense(total_words, activation='softmax'))
model.build(input_shape=(None, max_line_len-1))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, y, epochs=40, verbose=1)

Ilayaraja_book_model = {'model_json': model.to_json(),'model_weights': model.get_weights()}
pickle.dump(Ilayaraja_book_model, open('Ilayaraja_book_model.pkl', 'wb'))
