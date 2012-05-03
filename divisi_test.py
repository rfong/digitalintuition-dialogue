import divisi2
import numpy as np

assoc_matrix = divisi2.network.conceptnet_assoc('en')
assocU, assocS, assocV = assoc_matrix.normalize_all().svd(k=150)
assocmat = assocU.multiply(np.exp(assocS)).normalize_rows(offset=.00001)

limit = 20

# input terms
terms = ['I', 'feel', 'sick']

vec = divisi2.DenseVector(np.zeros((150,)))
for term, weight in terms:
  if term in assocmat.row_labels:
    vec += assocmat.row_named(term) * weight
similar = assocmat.dot(vec)
top_items = similar.top_items(limit)

print assocmat.row_named('happy')
