import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences

x = "தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்த ஒரு அழகிய மாநிலமாகும். இது பல்வேறு கலாச்சார பாரம்பரியங்களையும், செழிப்பான சாகுபடிமுறையையும் கொண்டுள்ளது. தமிழ்நாட்டின் தலைநகரமான சென்னை, தொழில்நுட்பம் மற்றும் கல்வியில் முன்னணி வகிக்கிறது. மாமல்லபுரம், தஞ்சாவூர் பெரிய கோயில் போன்ற வரலாற்று முக்கியத்துவம் வாய்ந்த இடங்கள் சுற்றுலாப் பயணிகளை ஈர்க்கின்றன. தமிழ்நாட்டின் கலை, இலக்கியம் மற்றும் இசை உலகளாவிய புகழ் பெற்றவை"

tokens = Tokenizer()
tokens.fit_on_texts([x])
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
from gensim.models import Word2Vec 
vec = Word2Vec(sentences=[i.split() for i in x.split('.')], vector_size=100, window=5, min_count=1, workers=1) 	
embeds = np.zeros((total_words, 100)) 
for word, i in dictionary.items():  
    if word in vec.wv:        
        embeds[i] = vec.wv[word] 
model.add(Embedding(total_words, 100, input_length=max_line_len - 1, weights=[embeds], trainable=False))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.build(input_shape=(None, max_line_len-1))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, y, epochs=100, verbose=1)

input_text = "தமிழ்நாடு"
predict_next_words= 3

for _ in range(predict_next_words):
    input_token = tokens.texts_to_sequences([input_text])[0]
    input_x = pad_sequences([input_token], maxlen=max_line_len-1, padding='pre')
    predicted = np.argmax(model.predict(input_x), axis=-1) # Greedy. tensorflow.random.categorical(input_token,num_samples=1) for sampling for beam search 
    output_word = ""
    for word, index in tokens.word_index.items():
        if index == predicted:
            output_word = word
            break
    input_text += " " + output_word

print(input_text)
