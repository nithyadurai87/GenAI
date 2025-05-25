import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

x = "தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்த ஒரு அழகிய மாநிலமாகும். இது பல்வேறு கலாச்சார பாரம்பரியங்களையும், செழிப்பான சாகுபடிமுறையையும் கொண்டுள்ளது. தமிழ்நாட்டின் தலைநகரமான சென்னை, தொழில்நுட்பம் மற்றும் கல்வியில் முன்னணி வகிக்கிறது. மாமல்லபுரம், தஞ்சாவூர் பெரிய கோயில் போன்ற வரலாற்று முக்கியத்துவம் வாய்ந்த இடங்கள் சுற்றுலாப் பயணிகளை ஈர்க்கின்றன. தமிழ்நாட்டின் கலை, இலக்கியம் மற்றும் இசை உலகளாவிய புகழ் பெற்றவை"

tokens = Tokenizer()
tokens.fit_on_texts([x])
dictionary = tokens.word_index

x_n_grams = []
y_n_grams = []
window_size = 3
for line in x.split('.'):
    line_tokens = tokens.texts_to_sequences([line])[0]
    for i in range(len(line_tokens) - window_size):
        n_grams = line_tokens[i:i + window_size]
        label = line_tokens[i + window_size]
        x_n_grams.append(n_grams)
        y_n_grams.append(label)

max_line_len = max([len(i) for i in x_n_grams])     
total_words = len(dictionary) + 1

train_X = np.array(pad_sequences(x_n_grams, maxlen=max_line_len, padding='pre'))
y = np.array(tf.keras.utils.to_categorical(y_n_grams, num_classes=total_words))    

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_line_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.build(input_shape=(None, max_line_len-1))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, y, epochs=100, verbose=1)

input_text = "தமிழ்நாடு இந்தியாவின் தெற்கே"
input_token = tokens.texts_to_sequences([input_text])[0]
input_x = pad_sequences([input_token], maxlen=max_line_len-1, padding='pre')
predicted = np.argmax(model.predict(input_x), axis=-1)
for word, index in tokens.word_index.items():
        if index == predicted:
          print (word)
