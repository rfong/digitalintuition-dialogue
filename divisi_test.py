import os
import divisi2
import numpy as np
import nltk
import math

def main():
  assoc_matrix = divisi2.network.conceptnet_assoc('en')
  assocU, assocS, assocV = assoc_matrix.normalize_all().svd(k=150)
  assocmat = assocU.multiply(np.exp(assocS)).normalize_rows(offset=.00001)
  print "finished constructing conceptnet matrix"
  
  # input
  input_files = []
  for dirname, dirnames, filenames in os.walk('GMIAS_CMU/'):
    for filename in filenames:
      if filename.split('.')[-1] == 'parsed':
        input_files.append( os.path.join(dirname, filename) )
  allTerms = { 'doctors':[], 'patients':[]}
  for input_file in input_files:
    fin = open( input_file )
    for line in fin.readlines():
      line = line.strip()
      hash = line.split('#')[-1]
      line_terms = nltk.word_tokenize( line.split('#')[0] )
      if hash[0]=='D':
        allTerms['doctors'] += line_terms
      elif hash[0]=='P':
        allTerms['patients'] += line_terms

  concepts = [line.strip() for line in open('concepts.txt', 'r').readlines()]
  
  # weight terms by function of frequency
  for group, terms in allTerms.iteritems():
    print group
    freqDict = {}
    for term in terms:
      freqDict[term] = freqDict.get(term, 0) + 1
    # let's normalize to keep the final score down

    #freqDict = {term : weight_fn(freqDict.get(term)) / weight_fn(max(freqDict.values())) for term in freqDict.keys()}
    for term in freqDict.keys():
      freqDict[term] = weight_fn(freqDict.get(term)) / weight_fn(max(freqDict.values()))
    # divisi
    vec = divisi2.DenseVector(np.zeros((150,)))
    for term in terms:
      if term in assocmat.row_labels:
        vec += assocmat.row_named(term) * freqDict.get(term)

    for concept in concepts:
      concept_vec = assocmat.row_named(concept)
      print '\t' + concept + ':', str( concept_vec.dot(vec) )

def weight_fn(freq):
  return math.pow(float(freq), 1.0/2)

def exit():
  print 1/0

main()
