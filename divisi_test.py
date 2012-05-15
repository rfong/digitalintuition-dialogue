import os
import divisi2
import numpy as np
import nltk
import math
import csv

def main():
  assoc_matrix = divisi2.network.conceptnet_assoc('en')
  assocU, assocS, assocV = assoc_matrix.normalize_all().svd(k=150)
  global assocmat
  assocmat = assocU.multiply(np.exp(assocS)).normalize_rows(offset=.00001)
  print "finished constructing conceptnet matrix"
  
  # clear old output files (if we changed window range, there will be junk)
  os.system( "rm -f `ls | grep -E 'patients|doctors[0-9]+.*'`" )
 
  # input
  input_files = []
  for dirname, dirnames, filenames in os.walk('GMIAS_CMU/'):
    for filename in filenames:
      if filename.split('.')[-1] == 'parsed':
        input_files.append( os.path.join(dirname, filename) )
  allTerms = { 'doctors':[], 'patients':[]}
  maxWindow = -1
  for input_file in input_files:
    fin = open( input_file )
    maxWindow = min( len(fin.readlines()), maxWindow ) if maxWindow > -1 else len(fin.readlines())
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

  global concepts
  concepts_raw = [line.strip() for line in open('concepts.txt', 'r')]
  concepts = [ (line.split(';')[0].split(','), line.split(';')[1].split(',')) for line in concepts_raw]
  # for printing later
  concepts_raw = [ line.replace(',', ' ').replace(';', ' / ') for line in concepts_raw ]

  #maxWindow = min( len(allTerms['doctors']), len(allTerms['patients']) )
  windowStep = minWindow = 5
  #windowSizes = [150]
  windowSizes = range(minWindow, maxWindow, windowStep)
  endSize = (maxWindow/windowStep - 1)*windowStep + minWindow
  for group, terms in allTerms.iteritems():
    print group
    
    for windowSize in windowSizes:
      print '\t' + str(windowSize)
      # weight terms by function of frequency
      freqDict = getWeights(terms, windowSize)

      newfile = open(group+str(windowSize)+'.csv', "w")
      writer = csv.writer(newfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow( concepts_raw )

      # construct first window
      window = []
      first_sentence_len = 0 # since we aggregate all terms in sentences
      if windowSize > len(terms): # sanity check
        window = terms
      else:
        first_sentence_len = len(terms[0])
        for sentence in terms[:windowSize]:
          window += sentence

      # roll through windows
      for i in xrange( 1, len(terms) - windowSize ):
        if len(window)==0:
          continue
        # divisi
        vec = divisi2.DenseVector(np.zeros((150,)))
        for term in window:
          if term in assocmat.row_labels:
            vec += assocmat.row_named(term) * freqDict.get(term)

        similarities = []
        for concept_set in concepts:
          concept_vec = np.zeros((150,))
          for concept in concept_set[0]:
            concept_vec += assocmat.row_named(concept)
          for concept in concept_set[1]:
            concept_vec += assocmat.row_named(concept)
          similarities.append(str(concept_vec.dot(vec)))
        writer.writerow(similarities)
          #print '\t' + str(concept) + ':', str( concept_vec.dot(vec) )
        #print ''

        # next rolling window
        window = window[first_sentence_len:] + terms[i+windowSize]
        first_sentence_len = len(terms[i])

def getWeights(terms, windowSize):
  global assocmat
  global concepts
  # get all frequencies
  freqDict = {}
  for sentence in terms:
    for term in sentence:
      freqDict[term] = freqDict.get(term, 0) + 1
  # let's normalize to keep the final score down
  maximum = max(freqDict.values())
  for term in freqDict.keys():
    freqDict[term] = weight_fn(freqDict.get(term)) / weight_fn(maximum)
  return freqDict

# not sure what to do here; sqrt is probably better for emotion, inverse sqrt better for medical/technical evaluation
def weight_fn(freq):
  return math.sqrt(freq)

def exit():
  print 1/0

main()
