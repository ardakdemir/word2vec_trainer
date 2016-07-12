from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim
import sys
args = sys.argv
input_file=args[1]
w2v_text_file=args[2]
wv = KeyedVectors.load(input_file)
out =  w2v_text_file
with open(out,'w') as o:
    s = ''
    for i,word in enumerate(wv.wv.vocab):
        s+="{}\t{}\t{}\n".format(i,word,wv[word])
    o.write(s)
