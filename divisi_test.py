import os, sys, math, csv
from optparse import OptionParser
import divisi2
import numpy as np
import nltk

results_dir = 'results/'
corpus_dir = 'GMIAS_CMU/'

def main():
  print "constructing conceptnet matrix..."
  assoc_matrix = divisi2.network.conceptnet_assoc('en')
  assocU, assocS, assocV = assoc_matrix.normalize_all().svd(k=150)
  global assocmat
  assocmat = assocU.multiply(np.exp(assocS)).normalize_rows(offset=.00001)
  
  # clear old output files (if we changed window range, there will be junk)
  os.system( "rm -f `ls | grep -E '" + results_dir + "patients|doctors[0-9]+.*'`" )
 
  print "parsing corpus..."
  input_files = []
  for dirname, dirnames, filenames in os.walk(corpus_dir):
    for filename in filenames:
      if filename.split('.')[-1] == 'parsed':
        input_files.append( os.path.join(dirname, filename) )
  allTerms = { 'doctors':[], 'patients':[]}
  maxWindow = -1
  for input_file in input_files:
    fin = open( input_file )
    maxWindow = min( len(fin.readlines()), maxWindow ) \
        if maxWindow > -1 else len(fin.readlines())
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
  concepts = []
  for line in concepts_raw:
    concept = line.split(';')
    concepts.append( ( \
      concept[0].split(',') if concept[0]!='' else [], \
      concept[1].split(',') if concept[1]!='' else [] \
      ) )
  # for printing to CSV later
  concepts_raw = [ line.replace(',', ' ').replace(';', ' / ') for line in concepts_raw ]

  #maxWindow = min( len(allTerms['doctors']), len(allTerms['patients']) )
  windowStep = minWindow = 5
  windowSizes = range(minWindow, maxWindow, windowStep)
  endSize = (maxWindow/windowStep - 1)*windowStep + minWindow

  for group, terms in allTerms.iteritems():
    print "calculating for %s..."%group
    best_peakiness_avg = {'windowSize':-1, 'peakinesses':[sys.float_info.min]}
    best_peakiness_by_concept = []
    for concept_set in concepts:
      best_peakiness_by_concept.append({'windowSize':-1, 'peakiness':sys.float_info.min})
    
    for windowSize in windowSizes:
      verbose('  ' + str(windowSize))

      # weight terms by function of frequency
      freqDict = get_weights(terms, windowSize)

      newfile = open(results_dir+group+str(windowSize)+'.csv', "w")
      writer = csv.writer(newfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow( concepts_raw )

      # score tracking for each concept
      scores = [ [] for concept_set in concepts ]

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

        scores_this_window = []
        for concept_set in concepts:
          concept_vec = np.zeros((150,))
          for concept in concept_set[0]:
            concept_vec += assocmat.row_named(concept)
          for concept in concept_set[1]:
            concept_vec -= assocmat.row_named(concept)
          score = concept_vec.dot(vec)
          scores_this_window.append(str(score))
          scores[concepts.index(concept_set)].append(score)
        writer.writerow(scores_this_window)
        scores.append(scores_this_window)

        # next rolling window
        window = window[first_sentence_len:] + terms[i+windowSize]
        first_sentence_len = len(terms[i])

      pk = peakiness(scores, windowSize)
      verbose(pk)
      if avg(pk) > avg(best_peakiness_avg['peakinesses']):
        best_peakiness_avg['peakinesses'] = pk
        best_peakiness_avg['windowSize'] = windowSize
      for concept_set in concepts:
        c = concepts.index(concept_set)
        if pk[c] > best_peakiness_by_concept[c]['peakiness']:
          best_peakiness_by_concept[c]['peakiness'] = pk[c]
          best_peakiness_by_concept[c]['windowSize'] = windowSize
          
  print "done."
  print 'best peakiness by average %s'%str(best_peakiness_avg)
  print 'best peakiness by concept'
  for concept_set in concepts_raw:
    print concept_set, str(best_peakiness_by_concept[concepts_raw.index(concept_set)])

def peakiness(L, windowSize):
  global concepts
  peakinesses = []
  for concept_set in concepts:
    c = concepts.index(concept_set)
    # two highest maxima
    peaks = [ (L[c][i], i) for i in find_peak_indices(indexed_list(L[c])) ]
    maxima0 = max(peaks, key=lambda t: t[0])
    peaks.pop( peaks.index(maxima0) )
    maxima1 = max(peaks, key=lambda t: t[0])
    # lowest val between the two peaks
    minima = min(L[c][ min(maxima0[1], maxima1[1])+1 : max(maxima0[1], maxima1[1]) ])
    # done
    peakinesses.append( min(maxima0[0], maxima1[0]) / minima ) # div/0 errors...too lazy to normalize
  return peakinesses

def find_peak_indices(L):
  peaks = []
  prev = L[0][0]
  foundPosDeriv = False
  for (x,i) in L:
    if not foundPosDeriv:
      if sign(x-prev) > 0:
        foundPosDeriv = True
    else:
      if sign(x-prev) < 0:
        peaks.append(i-1)
        foundPosDeriv = False
    if i==L[-1][1] and sign(x-prev) > 0: # ends on + deriv
      peaks.append(i)
    prev = x
  return peaks

def get_weights(terms, windowSize):
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

def avg(L):
  if len(L)==0:
    return None
  return sum(L)/len(L)
def sign(x):
  return cmp(x,0)
def indexed_list(L):
  return zip( L, xrange(len(L)) )

def verbose(str):
  if options.verbose:
    print str
def exit():
  print 1/0

### unit tests ###
class UnitTest:

  def __init__(self):
    self.find_peak_indices()

  def find_peak_indices(self):
    data = [ \
      ([1,0,1,0], [2]),
      ([1,0,1,0], [2]),
      ([0,2,1], [1]),
      ([-1,0,1,2,0], [3]),
      ([3,2,1], []),
      ([3,2,1,2], [3]),
      ([3,2,1,2,0,1], [3,5])
      ]
    for (L,a) in data:
      result = find_peak_indices(indexed_list([ float(x) for x in L ]))
      assert result == a, \
        "find_peak_indices(%s) should output %s, instead outputs %s"%(str(L),str(a),str(result))

###
parser = OptionParser()
parser.add_option("-v", "--verbose",
                  dest="verbose", default=False, action="store_true",
                  help="print more stuff")
parser.add_option("-o", "--optimize",
                  dest="optimize", default=False, action="store_true",
                  help="turn off unit tests and such")
(options, args) = parser.parse_args()

if not options.optimize:
  UnitTest()

main()
