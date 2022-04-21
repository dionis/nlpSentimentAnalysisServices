#!/bin/bash

echo "******************** OPINION PROCESS *******************"
cd opinion_process/nlp
mkdir pretrained
mkdir pretrained/bert
cd ../linguisticrule
python3 -m spacy download en_core_web_md
python3 -m spacy download es_core_news_sm
cd ../../
#TEST DIRECTORY
echo "******************** TEST DIRECTORY *******************"
cd test
mkdir pretrained
mkdir pretrained/bert
mkdir spacy_module
cd spacy_module
python3 -m spacy download en_core_web_md
python3 -m spacy download es_core_news_sm
cd ../../
#ROOT DIRECTORY
echo "********************ROOT DIRECTORY*******************"
mkdir spacy_module
cd spacy_module
python3 -m spacy download en_core_web_md
python3 -m spacy download es_core_news_sm

#https://spacy.io/usage
#python3 -m venv .env
#source .env/bin/activate
#pip install -U pip setuptools wheel
#pip install -U spacy
