import os
import urllib.request
import zipfile

glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = "glove.6B.zip"
glove_dir = "glove.6B"

if not os.path.exists(glove_dir):
    urllib.request.urlretrieve(glove_url, glove_zip_path)
    with zipfile.ZipFile(glove_zip_path, "r") as zip_ref:
        zip_ref.extractall(glove_dir)

# Convert GloVe to Word2Vec format
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = os.path.join(glove_dir, "glove.6B.100d.txt")
word2vec_output_file = "glove.6B.100d.word2vec.txt"

glove2word2vec(glove_input_file, word2vec_output_file)

# Load model
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)




import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Your paragraph
paragraph = (
    "Periyar was a social reformer in Tamil Nadu. "
    "He founded the Self-Respect Movement. "
    "This movement aimed to promote equality and end caste discrimination. "
    "Today, he is celebrated as a key figure in the fight for social justice and equality in Tamil Nadu"
)

# Sentence splitting
lines = [i for i in paragraph.split('.') if i.strip()]

# Tokenize, lowercase, remove stopwords & punctuation
stop_words = set(stopwords.words('english'))

x = [
    [word.lower() for word in word_tokenize(each_line)
     if word.lower() not in stop_words and word.isalpha()]
    for each_line in lines
]

# Flatten tokens
tokens = [word for sublist in x for word in sublist]

# Words in vocab
tokens_in_vocab = [word for word in tokens if word in model.key_to_index]

print("Words in vocab:", tokens_in_vocab)

# Check availability of words before using them
if 'periyar' in model and 'equality' in model:
    print("\nVector for 'periyar':", model['periyar'])
    print("Similarity between 'self-respect' and 'equality':", model.similarity('self-respect', 'equality'))
    print("Most similar to 'periyar':", model.most_similar(positive=['periyar'], topn=2))
else:
    print("\nSome words not found in GloVe vocabulary.")
