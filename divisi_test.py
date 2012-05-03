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
#terms = [(term, 1) for term in terms]

allTerms = terms.get('doctors') + terms.get('patients')
freqDict = {}
numUniqueTerms = 0
for term in allTerms:
  if freqDict.has_key(term):
    freqDict[term] += 1
  else:
    freqDict[term] = 1
  numUniqueTerms += 1
#freqDict = {term:(freqDict.get(term)/numUniqueTerms)

# anjali
vec = divisi2.DenseVector(np.zeros((150,)))
for group in term.items():
  for term in group:
    if term in assocmat.row_labels:
      vec += assocmat.row_named(term) * 1.0

happy = assocmat.row_named('happy')
sad = assocmat.row_named('sad')

similarHappy = happy.dot(vec)
similarSad = sad.dot(vec)

print "Happy Similarity: " + str(similarHappy)
print "Sad Similarity: " + str(similarSad)

