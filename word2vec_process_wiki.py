# -*- coding: utf-8 -*-
"""
This script process xml file from wiki using gensim library
Script was slightly modified to adapt to czech characters

source: http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

Czech wiki was downloaded using following link: https://dumps.wikimedia.org/cswiki/20160701/cswiki-20160701-pages-articles-multistream.xml.bz2

Run example:
python3 process_wiki.py cswiki-20160701-pages-articles-multistream.xml.bz2 wiki.cs.text
"""

import logging
import os.path
import sys
 
from gensim.corpora import WikiCorpus
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0
 
    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join([str(x,'utf-8' ) for x in text]) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
 
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
