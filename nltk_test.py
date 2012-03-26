import os
import numpy
import nltk

from collections import defaultdict
from nltk.probability import *


def main():
  input_files = [] 
  for dirname, dirnames, filenames in os.walk('GMIAS_CMU/'):
    for filename in filenames:
      if filename.split('.')[-1] == 'parsed':
        input_files.append( os.path.join(dirname, filename) )

  global utterances  # corpus dump
  utterances = []
  for input_file in input_files:
    fin = open( input_file )
    for line in fin.readlines():
      utt = parse_utterance(line)
      if utt != None:
        utterances.append(utt)

  for attr in ['speaker', 'topic', 'speech_act']:
    for tag in get_tag_set(attr):
      print attr, tag
      fd = nltk.FreqDist()
      for utt in get_utts(attr, tag):
        for word in nltk.word_tokenize( utt['text'] ):
          fd.inc(word)
      print dict(fd.items()[:30])


# return set of tags in corpus (utterances) for this utterance attribute
def get_tag_set(attr):
  return set([utt[attr] for utt in utterances])

# return set of utterances with a particular tag
def get_utts(attr, tag):
  return [utt for utt in utterances if utt[attr]==tag]

# return dict of utterance text, speaker, topic tag, speech act tag
def parse_utterance(line):
  line = line.split('#')
  text = ','.join(line[:-1])
  if text == '':
    return None
  utterance = {}
  utterance['text'] = text
  meta = line[-1].split('\n')[0].split(',')
  if len(meta) != 3:
    return None
  utterance['speaker'] = meta[0]
  utterance['topic'] = meta[1]
  utterance['speech_act'] = meta[2]
  if utterance['speaker'] not in ['D', 'P']:
    return None
  return utterance


main()
