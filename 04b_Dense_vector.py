from sklearn.preprocessing import LabelEncoder
import numpy as np

paragraph = "Periyar was a social reformer in Tamil Nadu. He founded the Self-Respect Movement. This movement aimed to promote equality and end caste discrimination. Today, he is celebrated as a key figure in the fight for social justice and equality in Tamil Nadu."
x = [i for i in paragraph.split('.')]

l1 = []
for i in x:
  l1.append(LabelEncoder().fit_transform(i.split()))
padded_arrays = [np.pad(i, (0, max(len(i) for i in l1) - len(i)), 'constant', constant_values=99) for i in l1]

print(np.array(padded_arrays))
