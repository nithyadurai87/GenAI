!pip install gensim
pip install numpy==1.23.5

from gensim.models import word2vec  
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

paragraph = "Periyar was a social reformer in Tamil Nadu. He founded the Self-Respect Movement. This movement aimed to promote equality and end caste discrimination. Today, he is celebrated as a key figure in the fight for social justice and equality in Tamil Nadu"
lines = [i for i in paragraph.split('.')]
x= [[word for word in nltk.word_tokenize(each_line) if word.lower() not in nltk.corpus.stopwords.words('english')] for each_line in lines]

model = word2vec.Word2Vec(x, window=10, vector_size=5, min_count=1, sg=1, sample=1e-3)

print (model.wv.index_to_key)
print (model.wv['Periyar'])
print (model.wv.similarity('Self-Respect', 'equality'))
print (model.wv.most_similar(positive=['Periyar'],topn=2))
#print (model.wv.most_similar(positive=['Peri'],topn=2))
