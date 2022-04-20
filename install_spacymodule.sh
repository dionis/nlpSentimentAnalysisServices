#!/bin/bash

echo "******************** OPINION PROCESS *******************"
cd opinion_process/nlp
mkdir pretrained
mkdir pretrained/bert
cd ../linguisticrule
python3 -m spacy download en_core_web_sm
python3 -m spacy download es_core_web_sm
cd ../../
#TEST DIRECTORY
echo "******************** TEST DIRECTORY *******************"
cd test
mkdir pretrained
mkdir pretrained/bert
mkdir spacy_module
cd spacy_module
python3 -m spacy download en_core_web_sm
python3 -m spacy download es_core_web_sm
cd ../../
#ROOT DIRECTORY
echo "********************ROOT DIRECTORY*******************"
mkdir spacy_module
cd spacy_module
python3 -m spacy download en_core_web_sm
python3 -m spacy download es_core_web_s
