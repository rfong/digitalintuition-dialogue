import codecs, re, os
from bs4 import BeautifulSoup

# This works specifically for Anvil encoded tinkrbook files.

curdir = '/Users/havasi/code/digitalintuition-dialogue/Tinkrbk1/'

for curfile in os.listdir(curdir):
    if curfile.split('.')[1] != '.anvil': continue
    # Open the File
    sixteen = codecs.open(curfile, mode='r', encoding="utf-16'")
    data = sixteen.read()
    sixteen.close()

    # Get text out
    xml_parsed = BeautifulSoup(data, "xml")
    text = xml_parsed.get_text()
    text= text.split('pageTurning')[0].strip()
    text = text.replace('UTF-16', '')

    outfile = open(curfile.split('.')[0] + '.cont', 'w')
    for line in text.split('\n'):
        if line.strip() == '':continue
        outfile.write(line + '\n')
    outfile.close()
