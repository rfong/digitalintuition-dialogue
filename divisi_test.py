import os
import divisi2
import numpy as np
import nltk

assoc_matrix = divisi2.network.conceptnet_assoc('en')
assocU, assocS, assocV = assoc_matrix.normalize_all().svd(k=150)
assocmat = assocU.multiply(np.exp(assocS)).normalize_rows(offset=.00001)

# input
input_files = []
for dirname, dirnames, filenames in os.walk('GMIAS_CMU/'):
  for filename in filenames:
    if filename.split('.')[-1] == 'parsed':
      input_files.append( os.path.join(dirname, filename) )
terms = { 'doctors':[], 'patients':[]}
for input_file in input_files:
  fin = open( input_file )
  for line in fin.readlines():
    hash = line.split('#')[-1]
    line_terms = nltk.word_tokenize( line.split('#')[0] )
    if hash[0]=='D':
      terms['doctors'] += line_terms
    elif hash[0]=='P':
      terms['patients'] += line_terms


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


