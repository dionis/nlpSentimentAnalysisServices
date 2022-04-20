import string
import re
import xml
import os
import tarfile
import zipfile
import pickle
import numpy as np
from nltk.stem import LancasterStemmer
from linguisticrule import binliu
from linguisticrule import data_helpers
from linguisticrule import object_definition
from linguisticrule import object_definition as Sentence
from linguisticrule import object_definition as Datasets
from linguisticrule import semeval2014processor as Semeval
from linguisticrule import corpusprocessing

#import fine_grained_sentiment_analysis as func
from linguisticrule import spacy_fine_grained_sentiment_analysis as func

from linguisticrule import L2MN

DATASET_ADDRES = "./linguisticrule/data/deeplearninDataset"

# Preguntar significado de etiquetas B-A I-A

# Cargando lexicon de opiniones
# O_S = func.getOpinionsFromFile('Opinion_Seed.txt')
# O_S = func.getOpinionsFromLexicon()



class Linguistic():
    def __init__(self, load_dataset = False, addess = ""):
        self.Product_Reviews_list = []
        self.aspectDT = set()
        self.sentimentDT = set()

        # Cargando datos del dataset
        # (dataSetDT, vocab) = corpusprocessing.processCorpus('binliucorpus', '')
        self.SPECIAL_TOKEN = "$T$"

        self.FILE_LEXICON_OPINION_MINING = "lexiconLifelonLearning"
        self.use_serialize_lexicon = False
        self.use_serialize = False
        self.doubleProgationAlgorithm = func.DoublePropagation(None, 'en')

        if self.use_serialize_lexicon == True:
            if os.path.exists( self.FILE_LEXICON_OPINION_MINING) and self.use_serialize_lexicon == True:
                fileObjet = open(self.FILE_LEXICON_OPINION_MINING, "rb")
                self.aspectDT = pickle.load(fileObjet)

        if self.use_serialize == False:

            (self.dataSetDTHuLiu, self.vocabHuLiu) = corpusprocessing.processCorpus('huAndLiuLexicon', '')
            (self.dataSetDT, self.vocab) = corpusprocessing.processCorpus('semeval2014', '')

            # Merge Bin Liu Lexicon and labeled dataset Lexicon and tests
            self.option_lexicon = "combine"
            self.dataSetDP = binliu.processingFileChen2014('./linguisticrule/data/KDD2014-Chen-Dataset')
            # Cargando apsectos del dominio etiquetado
            self.aspectDTHuLiu = corpusprocessing.obtenerAspectosDominioEtiquetado(self.dataSetDTHuLiu)
            self.aspectDTDataset = corpusprocessing.obtenerAspectosDominioEtiquetado( self.aspectDT )

            if self.use_serialize_lexicon == False:
                if self.option_lexicon == 'huAndLiu':
                    self.aspectDT = self.aspectDTHuLiu
                elif self.option_lexicon == 'dataset':
                    self.aspectDT = self.aspectDTDataset
                else:
                    self.aspectDT = self.aspectDTDataset
                    if self.aspectDT != None and self.aspectDTHuLiu != None:
                        self.aspectDT |= self.aspectDTHuLiu

            # Cargando todas las oraciones del dataset no etiquetado
            if load_dataset == True:
                for i in self.dataSetDP.document:
                    productReview = func.Product_Review(i[0])
                    for j in i[2]:
                        rs = func.Review_Sent([], j.text)
                        productReview.review_sentences_list.append(rs)
                    self.Product_Reviews_list.append(productReview)


    def createIOB2tagsAspect(self, currentSentence, stemmer):
            """
            Transform in IOB2 tags tags esqueme for result
            In our case, we are interested to determine
            whether a word or chunk is an aspect, so
            we only have "B–A", "I–A" and "O" tags   for the words.
            :param currentSentence: 
            :return: current Sentence with IOB2 data
            """

            #Built a list of tuple when each tuple have
            #(Aspect object, sentence text modified)
            listAspectSenten = []

            aspectList = currentSentence.getAspectTerm()
            sentence = currentSentence.text
            wordInSentence = data_helpers.tokenizer(sentence)
            tokenseInSentence = [ stemmer.stem(iword) for iword in wordInSentence] #Get words stemming is better
            lenSentenceInToken = len(tokenseInSentence)
            sizeCurrentTagList = len(currentSentence.tagList)
            tags = ["B-A", "I-A"]

            for iAspect in aspectList:
                word = iAspect.word
                iAspectToken = data_helpers.tokenizer(word)
                try:
                  indexStart = tokenseInSentence.index(stemmer.stem(iAspectToken[0]))
                except ValueError:
                    #print("Not exist Aspet " + iAspectToken[0] + " in sentence: " + sentence)
                    continue
                if indexStart != -1:  # Exist aspect in sentence
                    tmpWord = tokenseInSentence[indexStart]
                    tokenseInSentence[indexStart] = "B-A"
                    setenceTag = ""

                    for ipos, iToken in enumerate(tokenseInSentence):
                        if iToken in tags:
                            if iToken == tags[0]:
                                setenceTag += " " + self.SPECIAL_TOKEN
                            else:
                                setenceTag += ""
                        else:
                            setenceTag += " " + wordInSentence[ipos]

                    tokenseInSentence[indexStart] = tmpWord
                    listAspectSenten.append((iAspect,str.strip(setenceTag)))

            currentSentence.aspectSentence =  listAspectSenten
            currentSentence.tagList = tokenseInSentence
            return  currentSentence

#Electronic devices datasets
    def writeFile(self, documentsArray, filename):
            with open(filename, "a+", encoding="utf-8") as fileToWrite:
                notAscpet = 0
                positive = negative = neutral = 0
                for idocument in documentsArray:
                    try:
                        if idocument.insertAspect == True and idocument.aspectSentence != None:
                          lisAspect = idocument.aspectSentence
                          if len(lisAspect) > 0:
                              for aspect, text in lisAspect:
                                  # aspect, text =
                                  fileToWrite.write(str(text) + "\n")
                                  fileToWrite.write(aspect.word + "\n")
                                  polarity = "0"
                                  if aspect.wordpolarity == 'positive':
                                      polarity = '1'
                                      positive += 1
                                  elif aspect.wordpolarity == 'negative':
                                      polarity = '0'
                                      negative += 1
                                  elif aspect.wordpolarity == 'neutral':
                                      polarity = '-1'
                                      neutral += 1
                                  fileToWrite.write(polarity + "\n")

                          else:
                              fileToWrite.write(str(idocument.text) + "\n")
                              fileToWrite.write("<END>" + "\n")
                              polarity = "-1"  # None aspect
                              fileToWrite.write(polarity + "\n")
                              #print("No exist opinion and became None <---> exception captured")
                              notAscpet += 1
                              neutral += 1
                    except AttributeError :
                        print("No exist opinion and became None --- exception captured")
                return (positive, negative, neutral)
                # ("Sentences without aspect " + str(notAscpet))
            return True
    def processDataStructure (self, globalTrain, globalValid, listDocument, address, datasetname = 'global'):
        # address = address + os.sep + datasetname
        # if not os.path.exists(address + os.sep + datasetname):
        #     os.mkdir(address)

        directoryAddress = address + os.sep + "process"
        if not os.path.exists(directoryAddress):
            os.mkdir(directoryAddress)

        toWriteIn = directoryAddress + os.sep + datasetname
        lenDocument = len(listDocument)
        shuffle = np.random.permutation(lenDocument)
        positive = negative = neutral = 0
        trainToWrite = [listDocument[ivalue] for ivalue in shuffle[0: int(lenDocument * 0.8)]]
        validToWrite = [listDocument[ivalue] for ivalue in shuffle[int(lenDocument * 0.8):]]
        if not os.path.exists(toWriteIn):
            os.mkdir(toWriteIn)
        positivetmp, negativetmp, neutraltmp =\
        self.writeFile(trainToWrite, toWriteIn + os.sep + datasetname + "Train" + os.extsep + "raw")
        globalTrain = np.concatenate((globalTrain, np.array(trainToWrite)), axis=None)
        print ("")
        positive += positivetmp
        negative += negativetmp
        neutral += neutraltmp
        positivetmp, negativetmp, neutraltmp =\
            self.writeFile(validToWrite, toWriteIn + os.sep + datasetname + "Test" + os.extsep + "raw")
        globalValid = np.concatenate((globalValid, np.array(validToWrite)), axis=None)

        positive += positivetmp
        negative += negativetmp
        neutral += neutraltmp
        print ("============== Domain: " + datasetname + "=========================")
        print("Sentences to train = " + str(len(trainToWrite)))
        print("Sentences to tests = " + str(len(validToWrite)))
        print("Positive Aspect " + str(positive))
        print("Negative Aspect " + str(negative))
        print("Neutral Aspect " + str(neutral))

        return (globalTrain, globalValid)

    def processSimpleDataStructure (self, globalTrain, globalValid, listDocument, address, type):

            directoryAddress = address + os.sep + "process"
            if not os.path.exists(directoryAddress):
                os.mkdir(directoryAddress)

            shuffle = np.random.permutation(len(listDocument))
            trainToWrite = [listDocument[ivalue] for ivalue in shuffle]
            toWriteIn = directoryAddress
            positive = negative = neutral = 0

            if not os.path.exists(toWriteIn):
                os.mkdir(toWriteIn)

            fileName = "globalTrain" if type == "Train" else "globalTest"
            positivetmp, negativetmp, neutraltmp = \
                self.writeFile(trainToWrite, toWriteIn + os.sep + fileName + os.extsep + "raw")
            if globalTrain != None:
                globalTrain = np.concatenate((globalTrain, np.array(trainToWrite)), axis=None)
            print("")
            positive += positivetmp
            negative += negativetmp
            neutral += neutraltmp

            print("============== Domain: " + fileName + "=========================")
            print("Sentences to train = " + str(len(trainToWrite)))
            print("Positive Aspect " + str(positive))
            print("Negative Aspect " + str(negative))
            print("Neutral Aspect " + str(neutral))
            return (globalTrain, globalValid)

    def searchAspectPhrases(self, sentenceList, doubleProgationAlgorithm):
            Q_Addjective_Bound = 2
            K_Noun_Bound = 1
            #We have account if analysis is equal to sentence size
            listSentences = list()
            for inum, iSentence in enumerate(sentenceList[0].review_sentences_list):
                text = iSentence.sent
                doc = doubleProgationAlgorithm.getNLP(text)
                sizeSentence = len(doc)
                aspectPhrases = dict()

                for iAspect in iSentence.pred_target_set:
                    pos = 0
                    for token in doc:
                        posLeft = postRigth = -1
                        if iAspect.token == str(token.text).strip() and not (iAspect.token in aspectPhrases):
                            leftWord = list()
                            rigthWord = list()
                            if pos >= Q_Addjective_Bound:
                              posLeft = pos - 1
                              posibleAddj = doc[posLeft]
                              posLeft -= 1
                              if posibleAddj.tag_ in doubleProgationAlgorithm.JJ:
                                  leftWord.append((posibleAddj, posibleAddj.pos))
                              if posLeft > 0:
                                posibleAddj = doc[posLeft]
                                if posibleAddj.tag_ in doubleProgationAlgorithm.JJ:
                                    leftWord.append((posibleAddj,posibleAddj.pos))
                            if pos < sizeSentence -1:
                                postRigth = pos + K_Noun_Bound
                                possibleNoun = doc[postRigth]
                                if possibleNoun.tag_ in doubleProgationAlgorithm.NN:
                                    rigthWord.append((possibleNoun,possibleNoun.pos))

                            if len(rigthWord) >= 1 or len(leftWord) >= 1:
                                aspectPhrases[iAspect.token ] = (leftWord,rigthWord)

                                #target = func.Target(root_t.text, polarity, root_t.idx)

                        pos += 1

                new_pred_target_set = set()
                for iAspect in iSentence.pred_target_set:
                    if not (iAspect.token in aspectPhrases.keys()):
                        new_pred_target_set.add(iAspect)

                for key, listExpansionPhrase  in aspectPhrases.items():
                    target = None
                    for iValue in iSentence.pred_target_set:
                        if iValue.token == key:
                            target = iValue
                            break
                    if target != None:
                        leftWordList, rigthWordList = listExpansionPhrase
                        for iPossWord in leftWordList:
                            posibleAddj, position = iPossWord
                            newTarget = func.Target(posibleAddj.text + " " + key, target.polarity, position)
                            new_pred_target_set.add(newTarget)

                        for iPossWord in rigthWordList:
                            posibleNoun, position = iPossWord
                            newTarget = func.Target(key + " " + posibleNoun.text , target.polarity, position)
                            newTarget.position = target.position  #Update because the initial aspect word is first in position
                            new_pred_target_set.add(newTarget)

                iSentence.pred_target_set = new_pred_target_set  #Update expasion phrases
                print("Apect analysis end")
            pass
        # doc = nlp("Next week I'll   be in Madrid.")
        # for token in doc:
        #     print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        #         token.text,
        #         token.idx,
        #         token.lemma_,
        #         token.is_punct,
        #         token.is_space,
        #         token.shape_,
        #         token.pos_,
        #         token.tag_
        #     ))


    def buildDatasetDataStructure(self, sentenceList):
            # Create a new corpus (dataset) with un-supervised dataset search
            globalTrain = np.array([])
            globalValid = np.array([])
            stemmer = LancasterStemmer()  # Stemmer getting word stem

            listSentencesGlobal = list()
            for inum, iSentenceList in enumerate(sentenceList):
                listSentences = list()
                for iSentence in iSentenceList.review_sentences_list:
                    currentSentence = object_definition.Sentence()
                    currentSentence.text = iSentence.sent
                    for iAspect in iSentence.pred_target_set:
                        polarity = 'negative'
                        if iAspect.polarity > 0:
                            polarity = 'positive'
                        currentSentence.insertAspectDataCategory('', '', iAspect.token,
                                                                 polarity,
                                                                 '', '', [])

                        currentSentence.insertAspect = True

                    currentSentence = self.createIOB2tagsAspect(currentSentence, stemmer)
                    listSentences.append(currentSentence)
                    listSentencesGlobal.append(currentSentence)
                self.processDataStructure(globalTrain, globalValid, listSentences, DATASET_ADDRES, datasetname = iSentenceList.title)

            #Global dataset
            self.processDataStructure(globalTrain, globalValid, listSentencesGlobal, DATASET_ADDRES)
            return listSentences



    def loadFile(self,Product_Reviews_list):
            # Tomando las 20 primeras oraciones del dataset para probar
            Product_Reviews_list_Reducido = []
            productReviewReducido = func.Product_Review(Product_Reviews_list[0].title)
            for index, review in enumerate(Product_Reviews_list[0].review_sentences_list):
                if index == 15:  #Only take 20 sentences for more fast execution
                    break
                rs = func.Review_Sent([], review.sent)
                productReviewReducido.review_sentences_list.append(rs)
            Product_Reviews_list_Reducido.append(productReviewReducido)

            return Product_Reviews_list_Reducido

    def loadFileEx(self, Product_Reviews_list):
                # Tomando las 20 primeras oraciones del dataset para probar
                Product_Reviews_list_Reducido = []
                productReviewReducido = func.Product_Review("processing")
                for index, review in enumerate(Product_Reviews_list):
                    if index == 15:  # Only take 20 sentences for more fast execution
                        break
                    rs = func.Review_Sent([], review['sentence'])
                    productReviewReducido.review_sentences_list.append(rs)
                Product_Reviews_list_Reducido.append(productReviewReducido)

                return Product_Reviews_list_Reducido



            #Expand possible aspect words in sentence

    def process_text(self, Product_Reviews_list_Reducido):

          #Trasform setencesToProcess in Product_Reviews_list_Reducido

            self.aspectDP, self.sentimentDP = self.doubleProgationAlgorithm.DoubleProp(self.aspectDT, Product_Reviews_list_Reducido)

            self.FILE_SERIALIZE = "dataStructureSerialized"


            fileObjet = None
            #https://wiki.python.org/moin/UsingPickle
            if os.path.exists(self.FILE_SERIALIZE) and self.use_serialize == True:
                fileObjet = open(self.FILE_SERIALIZE, "rb")
                Product_Reviews_list_Reducido = pickle.load(fileObjet)
            else:
                if os.path.exists(self.FILE_SERIALIZE):
                    os.remove(self.FILE_SERIALIZE)
                if os.path.exists(self.FILE_LEXICON_OPINION_MINING):
                    os.remove(self.FILE_LEXICON_OPINION_MINING)
                fileObjet = open(self.FILE_SERIALIZE, "wb")
                pickle.dump(Product_Reviews_list_Reducido,fileObjet)

                fileObjectOMLexicon = open(self.FILE_LEXICON_OPINION_MINING, "wb")
                pickle.dump(self.sentimentDP, fileObjectOMLexicon)

            if fileObjet != None:
                fileObjet.close()

            if fileObjectOMLexicon != None:
               fileObjectOMLexicon.close()

            self.searchAspectPhrases(Product_Reviews_list_Reducido,self.doubleProgationAlgorithm)

            #Built posible dataset
            documentProcess = self.buildDatasetDataStructure(Product_Reviews_list_Reducido)

            rk = set()


            return  self.trasformOutputDataStructure(documentProcess,Product_Reviews_list_Reducido)

    def trasformOutputDataStructure(self, documentProcess, Product_Reviews_list_Reducido):
        outputDataStructure = []
        # print("royal_sentence => ", ival['royal_sentence'])
        # print("sentence => ", ival['sentence'])
        # print("aspect => ", ival['aspect'])
        # print("polarity => ", ival['polarity'])
        # print("predictpolarity => ", ival['predictpolarity'])
        sencente_list = Product_Reviews_list_Reducido[0].review_sentences_list
        for iEndSentence, iInputSente in zip(documentProcess,sencente_list):
            iDictResult = {}
            iDictResult["royal_sentence"]="";
            iDictResult["sentence"] = "";

            #Remember for each aspect there are one sentence
            iDictResult["aspect"] = "";
            iDictResult["polarity"] = "";
            iDictResult["predictpolarity"] = "";
            outputDataStructure.append(iDictResult)

        return outputDataStructure


# for dp in dataSetDP.document:
#     rk_i = L2MN.l2mn(dp, aspectDP, sentimentDP, None)
#     rk = rk | rk_i
#
# for index, dt in enumerate(dataSetDT):
#     (asa, cse) = L2MN.knowMining(rk, dt, aspectDT[index])

print('done')
