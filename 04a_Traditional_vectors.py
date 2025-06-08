import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')   
from sklearn.feature_extraction.text import CountVectorizer

paragraph = "Periyar was a social reformer in Tamil Nadu. He founded the Self-Respect Movement. This movement aimed to promote equality and end caste discrimination. Today, he is celebrated as a key figure in the fight for social justice and equality in Tamil Nadu."
x = [i for i in paragraph.split('.')]

tokens = CountVectorizer()
vectors = tokens.fit_transform(x)

print(tokens.vocabulary_)
print(vectors.toarray())

# tokens = CountVectorizer(stop_words='english') - NLTK
# tokens = CountVectorizer(ngram_range = (2, 2), stop_words='english') - Ngrams
# Dense vector / distributed representation for sparse issue
