import divisi2
import numpy as np

assoc_matrix = divisi2.network.conceptnet_assoc('en')
assocU, assocS, assocV = assoc_matrix.normalize_all().svd(k=150)
assocmat = assocU.multiply(np.exp(assocS)).normalize_rows(offset=.00001)

limit = 20

# parser
# output:
#  doctors, patients = list of sentences (ists of terms)
terms = { 'doctors':[], 'patients':[]}

# weight terms by sqrt of frequency
terms = [(term, 1) for term in terms]


# anjali
vec = divisi2.DenseVector(np.zeros((150,)))
for group in term.items():
  for term, weight in group:
    if term in assocmat.row_labels:
      vec += assocmat.row_named(term) * weight
similar = assocmat.dot(vec)
top_items = similar.top_items(limit)

print assocmat.row_named('happy')
