from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

tokens = pickle.load(open(r'/content/Ilayaraja_book_tokens.pkl', 'rb'))
model_file = pickle.load(open(r'/content/தமிழ்_புத்தகங்கள்_மாடல்.pkl', 'rb'))

model = model_from_json(model_file['model_json'])
model.set_weights(model_file['model_weights'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

input_text = "சுருக்கமாகச்"
predict_next_words= 3

for _ in range(predict_next_words):
    input_token = tokens.texts_to_sequences([input_text])[0]
    input_x = pad_sequences([input_token], maxlen=max_line_len-1, padding='pre')
    predicted = np.argmax(model.predict(input_x), axis=-1) 
    output_word = ""
    for word, index in tokens.word_index.items():
        if index == predicted:
            output_word = word
            break
    input_text += " " + output_word

print(input_text)
