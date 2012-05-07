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
      hash = line.split('#')[-1]
      line_terms = nltk.word_tokenize( line.split('#')[0] )
      if hash[0]=='D':
        allTerms['doctors'] += line_terms
      elif hash[0]=='P':
        allTerms['patients'] += line_terms
  
  # weight terms by sqrt of frequency
  for group, terms in allTerms.iteritems():
    print group
    freqDict = {}
    for term in terms:
      freqDict[term] = freqDict.get(term, 0) + 1
    freqDict = {term : math.sqrt(freqDict.get(term)) for term in freqDict.keys()}
  
    # divisi
    vec = divisi2.DenseVector(np.zeros((150,)))
    for term in terms:
      if term in assocmat.row_labels:
        vec += assocmat.row_named(term) * 1.0
    
    vec = divisi2.DenseVector(np.zeros((150,)))
    happy = assocmat.row_named('happy')
    sad = assocmat.row_named('sad')
    
    for term in terms:
      if term in assocmat.row_labels:
        vec += assocmat.row_named(term) * 1.0
    similarHappy = happy.dot(vec)
    similarSad = sad.dot(vec)
    print "Happy Similarity: " + str(similarHappy)
    print "Sad Similarity: " + str(similarSad)

def exit():
  print 1/0

main()
