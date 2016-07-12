# nlp_czech_wiki

Word2Vec and Fastext trainers

Pipeline:

- Download latest wikipedia dump for the language

- Preprocess the xml file into .txt

- Train a word2vec and a fasttext model for the predefined dimension

## Run
Whole pipeline for czech-finnish-turkish-hungarian can be  run through (this example train 768-dim vectors):

```
dimension=768
bash word2vec_pipeline.sh ${dimension}
```
