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
      # todo - fix this so we can handle any hash...
      if hash[0]=='D':
        allTerms['doctors'].append(line_terms)
      elif hash[0]=='P':
        allTerms['patients'].append(line_terms)

  concepts = [line.strip() for line in open('concepts.txt', 'r').readlines()]
  concepts = [ (line.split(';')[0].split(','), line.split(';')[1].split(',')) for line in concepts]
  
  # weight terms by function of frequency  
  for group, terms in allTerms.iteritems():
    print group
    for term in terms:
      freqDict[term] = freqDict.get(term, 0) + 1
    # let's normalize to keep the final score down
    maximum = max(freqDict.values())
    for term in freqDict.keys():
      freqDict[term] = weight_fn(freqDict.get(term)) / weight_fn(maximum)
    # divisi
    vec = divisi2.DenseVector(np.zeros((150,)))
    for term in terms:
      if term in assocmat.row_labels:
        vec += assocmat.row_named(term) * freqDict.get(term)
  
      for concept in concepts:
        concept_vec = np.zeros((150,))
        for c in concept[0]:
          concept_vec += assocmat.row_named(c)
        for c in concept[1]:
          concept_vec += assocmat.row_named(c)
        print '\t' + str(concept) + ':', str( concept_vec.dot(vec) )
      print ''

# not sure what to do here; sqrt is probably better for emotion, inverse sqrt better for medical/technical evaluation
def weight_fn(freq):
  return math.sqrt(freq)

def exit():
  print 1/0

main()
