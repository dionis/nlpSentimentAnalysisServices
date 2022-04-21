from ..senticnet.senticnet import  SenticNet
import spacy
import os
from spacy import displacy
from nltk.chunk import conlltags2tree
from nltk.corpus import stopwords
from itertools import repeat

from spacy_langdetect import LanguageDetector
from spacy.language import Language


@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()

#Google Traslate API
#https://pypi.org/project/google-trans-new/
#https://pypi.org/project/googletrans/
#
#
#https://pypi.org/project/spacy-langdetect/
#
#Other Language detector in python
#
#https://pypi.org/project/pycld2/
#

from spacy_langdetect import LanguageDetector
import string

def isaNoun(relationKeyTag):
    # if type(relationKeyTag) == 'str':
    if len(relationKeyTag) > 0 and relationKeyTag[0] == 'N':
        return True
    # else:
    #     return False

def isAdjective(relationKeyTag):
    # if type(relationKeyTag) == 'str':
    if len(relationKeyTag) > 0 and relationKeyTag[0] == 'J':
        # Adjetive word
        return True
    elif len(relationKeyTag) > 0 and relationKeyTag[0] == 'R':
        #Adverbial word
        return True
    # else:
    #     return False
    pass

def isInSentinet(sentinect, relationKeyTag):
    try:
        wordInSenctinet = sentinect.concept(relationKeyTag)
        if wordInSenctinet != None:
           return True
        return False
    except:
        return False
    pass


class TextAnalysis():

    def __init__(self):
        self.language = ""
        self.sentences = None
        self.entities = None


def hash_EntiyInFunction(mapForSentence, start_char, end_char):
    for key, values in mapForSentence.items():
         start, end  = values
         if end_char <= end and start <= start_char:
             return key




class LinguisticRuleSpaCy:
    """This class apply five linguistic rules proppose by 
       in "Aspect extraction for opinion mining with a deep convolutional neural
       networks" by 
       
       S. Poria, E. Cambria, and A. Gelbukh, 
       “Aspect extraction for opinion mining with a deep convolutional neural networks,”
       Knowledge-Based Systems, vol. 108, pp. 42–49, 2016.
    
    ."""
    aspectInSencente = []
    sentinect = None
    stopwordAnalizer = None    #NLTK stopwords corpus that contains word lists for many languages

    # Modelo de Ingles: en_core_web_md
    #
    # Modelo del Espanol: es_core_news_md

    def create_lang_detector(nlp, name):
        return LanguageDetector()

    def __init__(self, language, model = None):
        if model != None:
            self.nlp = model
        elif language == None:
            #Load Spacy Module to English Language
            # E:\___Dionis_MO\Articulos\IMPLEMENTACION\SOURCE\Inoid_ABSA_DL\ABSA - PyTorch - master\linguisticrule\en_core_web_md\en_core_web_md - 2.1
            # .0
            path = os.path.abspath('spacy_module/en_core_web_md/en_core_web_md-3.1.0')

            # to get the current working directory
            directory = os.getcwd()

            self.nlpEn = spacy.load(path)
            self.nlpEn .add_pipe( "language_detector",  last=True)
            #self.nlpEn.add_pipe("ner")

            # Load Spacy Module to English Spanish
            pathEs = os.path.abspath('spacy_module/es_core_news_md/es_core_news_md-3.1.0')
            self.nlpEs = spacy.load(pathEs)
            self.nlpEs.add_pipe( "language_detector", last=True)
            #self.nlpEs.add_pipe("ner")
        elif language == "en":
           self.nlp = spacy.load('en_core_web_md')
        elif language == "es":
            self.nlp = spacy.load('es_core_news_md')
        else:
            self.nlp = spacy.load('en')

        self.type =  "Spacy"

        self.sentinect = SenticNet()
        self.stopwordAnalizer = set(stopwords.words('english'))    #OOOOOOOOOOOOOOOOOJJJJJJJJJJJJJJOOOOOOOOOOOOOOOOOOOOOO
                                                              #Set the stopwordAnalyzer to englis. It must be possible

        self.ACOMP = "acomp"#"adjectival complement"                                                   #configure
        self.AUX_VERB = "aux"#"Auxiliary"
        self.ATTR = "attr"  # "attribute"
        self.AMOD = "advmod" #"Adverbial modifier"
        self.ADVMOD = "amod" #"Adjective modifier"
        self.WORD_LEMMA = "lemma" #Word lemma
        self.WORD_DEPS = "deps" #Word dependencies
        self.WORD_TAG = "tag"  # Word POS tag
        self.WORD_ADDRESS = 'address' #Position in Dependency parer list nodes
        self.WORD_TOKEN = "word" #"Royal word in a sentence"
        self.WORD_RELATION = "rel"  # "Dictonary entre in a node on Dependency Parser"
        self.WORD_AUX = 'aux' #An aux (auxiliary) of a clause is a function
                              # word associated with a verbal predicate that expresses
                              # categories such as tense, mood, aspect, voice or evidentiality.


        self.SENT_SUBJECT = "nsubj"  # Sentences subject
        self.SENT_ROOT = 'ROOT'  # Word root in a sentences
        self.SENT_PUNCT = "punct"  # "Depency parser relation to puctuation symbol"
        self.OBJ = 'dobj'     #The "direct object" grammatical relation. The direct object of a verb
                              #is the noun phrase wich is the (accusative)object
        self.COP = "cop"      #The "copula" grammatical relation.
                              #A copula is the relation between the complement of a copular
                              #verb and the copular verb.

        self.COMPOUND = "compound" #A compound modifier of an NP is any noun that serves to modify the head noun.
        super().__init__()

    def initAnalizer(self, text, root, deepLearnigAspect) -> None:
        """
        
        :param nodes: A sentence CoreNLP Sanford Parser Dependecy Tree 
        :param root:  A sentence root in a Dependency Tree 
        :param deepLearnigAspect: A list of word positions result of the deep 
                                 learning algorithm that identifies which are aspects (1) 
        """

        self.isInitAnalizer = True
        self.nodes = self.getDependencyParser(text)
        self.root = root
        self.algDLearningList = deepLearnigAspect
        return self.nodes



    def  getDependencyParser(self, text):
         if self.nlp == None:
             return None
         elif text == "" or text == None:
             return None
         else:
           return self.nlp(text);


    def initialiceAspectList(self):
        """
          Initialice a aspect list if node data estructure is null different
        :return: An empty aspect list or None (if Dependency parser data estructure is null)
        """

        if self.nodes == None:
            return None
        aspectInSentResult = list(repeat(0, len(self.nodes)))  # list result word positions in a sentence
        return aspectInSentResult
    #  that identifies which are aspects

    def r1linguistcRule(self):
        """
          Rule 1 Let a noun h be a subject of a word t ,
           which has an adverbial or adjective modifier present in 
           a large sentiment lexicon, SenticNet. Then mark h as an aspect.  
        :return: A list of word positions in a sentence that identifies which are aspects
        """
        #.algDLearningList CNN list with Aspect in a sentence
        if self.nodes == None:
            return None
        aspectInSentResult = list(repeat(0, len(self.nodes))) #list result word positions in a sentence
                                                                     #  that identifies which are aspects
        rootTagList = self.nodes
        # First get root t word to Rule 1
        #"Verb or principal word in the sentence")
        # listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        # mainWordInSent = listOfWord[self.SENT_ROOT][0]
        # rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]
        # for token in doc:
        #     print("{0}/{1} <--{2}-- {3}/{4}".format(
        #         token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
        for ipos, token in enumerate(rootTagList):
            # Let a noun h be a subject of a word t

            if token.dep_ == self.SENT_SUBJECT:
                #"Nominal subject relation")

                #"Subject relation HEAD")
                #print(self.nodes.nodes[subjectKey])
                #"Morphological tag")
                nodeTag = token.tag_
                #Relation for main subject word
                subListRela = token.children
                #"Relations:")


                # which has an adverbial or adjective modifier present in a large
                # sentiment lexicon, SenticNet
                for token_child in subListRela:
                    #if an adverbial modifier or adjectival modifier
                    if token_child.dep_ == self.AMOD or token_child.dep_ == self.ADVMOD:
                        #"Exist an adverbial or adjectival modifier")
                        # iKeyPossibleAdver = depsDict[iKeyRel][0]
                        # #print(iKeyPossibleAdver)
                        # possibleAdverbialRel = self.nodes.nodes[iKeyPossibleAdver]
                        #print(possibleAdverbialRel)
                        # If Adjectival or modifier
                        wordIs =  token_child.lemma_
                        #print("Search Adverbial or Adjectival " + wordIs + " in Senticnet")
                        try:
                            wordInSenctinet = self.sentinect.concept(wordIs)
                            if wordInSenctinet != None:
                                #Set 1 as Aspect identifier in word position in a sentence
                                aspectInSentResult[ipos] = 1
                                break
                        except:
                            break
                            # print(wordInSenctinet["polarity_value"])
                            # print(wordInSenctinet["polarity_intense"])
                            # print(wordInSenctinet["moodtags"])
                            # print(wordInSenctinet["sentics"])
                            # print(wordInSenctinet["semantics"])

        return aspectInSentResult

    def r2linguisticRule(self):
        """
        Rule 2 Except when the sentence has an auxiliary verb, such as
               is, was, would, should, could , etc., we apply:
               
                Rule 2.1 If the verb t is modified by an adjective or ad-
                         verb or is in adverbial clause modifier relation
                         with another token, then mark h as an aspect.
                         E.g., in “The battery lasts little”, battery is the
                         subject of lasts , which is modified by an adjective 
                         modifier little , so battery is marked as an aspect.

                        Rule 2.2 If t has a direct object, a noun n , not found in
                        SenticNet, then mark n an aspect, as, e.g., in “I
                        like the lens of this camera”.
                
                :return: A list of word positions that identifies which are aspects
        """
        if self.nodes == None:
            return None
        aspectInSentResult = list(repeat(0, len(self.nodes))) #list result word positions in a sentence
                                                                     #  that identifies which are aspects
        rootTagList = self.nodes
        # First get root t word to Rule 1
        #"Verb or principal word in the sentence")
        # listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        # mainWordInSent = listOfWord[self.SENT_ROOT][0]
        # rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]
        # for token in doc:
        #     print("{0}/{1} <--{2}-- {3}/{4}".format(
        #         token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

        # First get root t word to Rule 1
        # "Verb or principal word in the sentence")
        # listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        # mainWordInSent = listOfWord[self.SENT_ROOT][0]
        # rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]

        # Except when the sentence has an auxiliary verb, such as is,
        # was, would, should, could, etc., we apply
        if self.isAuxilaryVerbInSent(rootTagList) == True:
            return None
        else:
            subjectKey = None
            relationKeyObj = None
            subjectKeyObject = None
            self.rule21 = self.rule22 = False
            ruleObject = None
            for ipos, token in enumerate(rootTagList):
                # Let a noun h be a subject of a word t
                # If the verb t is modified by an adjective or adverb or is in adverbial clause
                # modifier relation with another token, then mark h as an aspect.

                    # Let a noun h be a subject of a word t
                if token.dep_ == self.SENT_SUBJECT:
                     # "Nominal subject relation")
                     subjectKey = token.children
                     # "Subject relation HEAD")
                     # print(self.nodes.nodes[subjectKey])
                     # "Morphological tag")
                     subjectKeyObject = token
                     nodeTag = token.tag_
                     verbRelation = token.head
                     verbChildren = verbRelation.children

                     for ichild_verb in verbChildren:
                         if ichild_verb != token:
                             if ichild_verb.dep_ == self.ADVMOD or ichild_verb.dep_ == self.AMOD:
                                 if isAdjective(ichild_verb.tag_):
                                     self.rule21 = True
                                     ruleObject = token
                                     break
                             elif ichild_verb.dep_ == self.OBJ:
                                 wordInRelation = ichild_verb.lemma_
                                 if isaNoun(ichild_verb.tag_) and isInSentinet(self.sentinect, wordInRelation) == False:
                                      self.rule22 = True
                                      ruleObject = ichild_verb
                                      break


                if ruleObject != None:
                    break

            if ruleObject:
                for ipos, token in enumerate(rootTagList):
                    if token == ruleObject:
                       aspectInSentResult[ipos] = 1
                       break;
            else:
                    return None

            return aspectInSentResult

    def r3linguisticRule(self):
        """
        Rule 3 If a noun h is a complement of a couplar verb, then mark
               h as an explicit aspect. E.g., in “The camera is nice”, 
               camera is marked as an aspect.
        :return: A list of word positions that identifies which are aspects
        """
        if self.nodes == None:
            return None
        aspectInSentResult = list(repeat(0, len(self.nodes))) # list result word positions in a sentence
        #  that identifies which are aspects
        rootTagList = self.nodes
        # First get root t word to Rule 1
        # "Verb or principal word in the sentence")
        # listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        # mainWordInSent = listOfWord[self.SENT_ROOT][0]
        # rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]

        subjectKey = None
        relationKeyObj = None
        subjectKeyObject = None
        rule3 = False

        # If the verb t is modified by an adjective or adverb or is in adverbial clause
        # modifier relation with another token, then mark h as an aspect.
        for ipos, token in enumerate(rootTagList):
            # Let a noun h be a subject of a word t
            if token.dep_ == self.ACOMP or token.dep_ == self.COP or token.dep_ == self.ATTR:
                # "Nominal subject relation")
                verbHead = token.head
                if token.dep_ != self.COP and self.isPossibleCopular(verbHead.tag_, verbHead.text) == False:
                    return None
                elif(not isaNoun(token.tag_) and not isAdjective(token.tag_)):
                    return None
                aspectInSentResult[ipos] = 1
                self.rule3 = True
                return aspectInSentResult

        return None

    def isPossibleCopular(self, tag_, text):
        enAuxVerb = ['is','are','was','were','been','be']


        if tag_ == self.AUX_VERB:
            return True
        elif text in enAuxVerb:
            return True
        return False

        pass

    def r4linguisticRule(self, _liguistiRuleSelctAspect):
        """
        Rule 4 If a term marked as an aspect by the CNN or the other
               rules is in a noun-noun compound relationship with 
               another word, then instead form one aspect term composed
               of both of them. E.g., if in “battery life”, “battery” or “life”
               is marked as an aspect, then the whole expression is
               marked as an aspect.
        :return: A list of word positions that identifies which are aspects
        """
        if self.nodes == None:
            return None

        if _liguistiRuleSelctAspect == None and self.algDLearningList == None:
            return None

        aspectInSentResult = list(repeat(0, len(self.nodes)))  # list result word positions in a sentence
        #Analized Deep Learning  Aspect selection
        self.rule4 = False
        sizeDeepLearningSelection = 0
        sizeLingRuleAspectSelect = 0
        rootTagList = self.nodes
        if self.algDLearningList != None and len(self.algDLearningList) > 0:
            sizeDeepLearningSelection = len(self.algDLearningList)
            for ipos, iaspct in enumerate (self.algDLearningList):
                if iaspct == 1:
                    aspectInSentResult[ ipos] = 1

        if _liguistiRuleSelctAspect != None and len(_liguistiRuleSelctAspect) > 0:
            sizeLingRuleAspectSelect = len(_liguistiRuleSelctAspect)
            for ipos, iaspct in enumerate(_liguistiRuleSelctAspect):
                if iaspct == 1:
                    aspectInSentResult[ipos] = 1

        for ipos, token in enumerate( rootTagList):
            head = token.head
            headpos = head.i
            if (self.algDLearningList != None and self.algDLearningList[headpos] == 1) or (_liguistiRuleSelctAspect != None and _liguistiRuleSelctAspect[headpos] == 1): #It is selected as an aspect by Deep Learnig algorithm
                     if token.dep_ == self.COMPOUND and isaNoun(token.tag_):
                        aspectInSentResult[ipos] = 1
                        self.rule4 = True

        if self.rule4 == False:
            return None

        return aspectInSentResult

    def r5linguisticRule(self, _listAspectInSentence):
        """
        Rule 5 The above rules 1–4 improve recall by discovering more
              aspect terms. However, to improve precision, we apply
              some heuristics: e.g., we remove stop-words such as of,
              the, a , etc., even if they were marked as aspect terms by
              the CNN or the other rules.
        
        :return: A list of tuples with word positions that identifies which are aspects
                 and Dependency Parser element with information
        """
        if self.nodes == None or _listAspectInSentence == None:
            return None
        if len(self.nodes) != len(_listAspectInSentence):
            return None

        aspectInSentence = []                                               #Merge in a same list Deep Learnig algorithm
                                                                             #result with linguistic rule result
        if self.algDLearningList != None and len(self.algDLearningList) > 0:
            aspectInSentence = self.algDLearningList

        if len(aspectInSentence) == 0:
            aspectInSentence = _listAspectInSentence
        else:
            for iElement in  range(len(_listAspectInSentence)):
                #Only in element with 0 value in aspectInsente
                if _listAspectInSentence[iElement] == 1:
                    aspectInSentence[iElement] = 1

        resultAspectDependcyAux = zip(self.nodes,aspectInSentence)  #Merge in a tuple Depency Parser info and
                                                                                         #aspect identification
        resultAspectDependcy = []

        for ipos, iElement in enumerate (resultAspectDependcyAux):
            #Delete all stopword element. Use compresion list when we know how accecing to tuples elements
            depencyObject, isAspect = iElement
            if depencyObject != None and  depencyObject.is_stop == True:
                aspectInSentence[ipos] = 0

        return aspectInSentence

    def applyRules(self):
        """
         Apply all rules over nodes and root for getting 
         aspect in a sentence.
         When apply each rule an aspect list is updated
        :return: A dictionary of word positions that identifies which are aspects
        """

        aspectInSentResult = self.initialiceAspectList();
        if aspectInSentResult == None and (self.isInitAnalizer == False or  self.isInitAnalizer == None):
            return None

        #Apply rule 1
        rule1AspectList = self.r1linguistcRule()
        if rule1AspectList == None:
            return rule1AspectList

        aspectInSentResult = self.updateAspectListElemnt(aspectInSentResult,rule1AspectList)

        #Apply rule 2
        rule2AspectList = self.r2linguisticRule()
        # if rule2AspectList == None and not self.iNotAspectInList(aspectInSentResult):
        #     return rule2AspectList
        aspectInSentResult = self.updateAspectListElemnt(aspectInSentResult, rule2AspectList)

        # Apply rule 3
        rule3AspectList = self.r3linguisticRule()
        # if rule3AspectList == None and not self.iNotAspectInList(aspectInSentResult):
        #     return rule3AspectList
        aspectInSentResult = self.updateAspectListElemnt(aspectInSentResult, rule3AspectList)

        # Apply rule 4
        rule4AspectList = self.r4linguisticRule(aspectInSentResult)
        # if rule4AspectList == None and not self.iNotAspectInList(aspectInSentResult):
        #     return rule4AspectList
        aspectInSentResult = self.updateAspectListElemnt(aspectInSentResult, rule4AspectList)

        # Apply rule 5 delete stopwords
        #Add a punct element at last position in aspectInSentResult list

        aspectInSentResult = aspectInSentResult
        aspectInSentTuples = self.r5linguisticRule(aspectInSentResult)

        isAspect = False
        for iValue in aspectInSentTuples:     #Verify there are Aspects
            if iValue == 1:
                isAspect = True
                break

        if not isAspect:
            return None
        #Delete end punctuction elements if appear in last element

        rootTagList = self.nodes


        lastElement = rootTagList[-1]
        if lastElement.is_punct == True:
            rootTagList = rootTagList[:-1]

        #Create dictionary with word and aspect identificador 1/0
        resultDictionary = {}
        allwordDictionary = {}
        for ipos, elemt in enumerate (rootTagList):
            word = elemt.text
            wordPost = elemt.idx  #Word position in a sentences
            if  not elemt.is_stop and not elemt.is_punct and not elemt.is_digit and not elemt.is_space:
              resultDictionary[word] = aspectInSentTuples[ipos]
            if not elemt.is_space:
             allwordDictionary[word] = (aspectInSentTuples[ipos], wordPost)

        return (allwordDictionary, resultDictionary)

    def updateAspectListElemnt(self, aspectInSentResult, ruleAspectList):
        """
        
        :param aspectInSentResult: Aspect list identify when apply each linguistic rule 
        :param ruleAspectList: : Aspect list when apply an specific rule
        :return: An update list when all aspect identify for each linguistic rule 
        """
        if ruleAspectList != None:
            for iValue in range (len(aspectInSentResult)):
                if ruleAspectList[iValue] == 1:
                    aspectInSentResult[iValue] = 1

        return aspectInSentResult

    def iNotAspectInList(self, aspectInSentResult):
        """If there almost and element in aspectInSentResult zero different"""
        return (True if len([n for n in aspectInSentResult if n == 1]) > 0 else False)

    def isAuxilaryVerbInSent(self, rootTagList):
        if rootTagList == None:
            return True
        for token in rootTagList:
            if token.dep_ == self.WORD_AUX:
                return  True
        return False

    def process_text(self, _text):
        #Language detection
        doc = self.nlpEn (_text)
        # document level language detection. Think of it like average language of the document!
        if (doc._.language[0] != 'en'):
            doc = self.nlpEs(_text)
            if (doc._.language[0] != 'es'):
              raise Exception("Unknow document language")
        print(doc._.language)
        # sentence level language detection
        for sent in doc.sents:
            print(sent, sent._.language[0])

    def getSplitterSentences(self, _text):
         if _text == None or _text == "":
             return None
         else:
             doc = self.nlpEn(_text)
             resultObject = TextAnalysis()
             resultObject.sentences = []
             # document level language detection. Think of it like average language of the document!
             if (doc._.language['language'] != 'en'):
                 doc = self.nlpEs(_text)
                 if (doc._.language['language'] != 'es'):
                     raise Exception("Unknow document language")
                 else:
                     resultObject.language = 'es'
             else:
                 resultObject.language = 'en'

             assert doc.has_annotation("SENT_START")
             for sent in doc.sents:
                 print(sent.text)
                 resultObject.sentences.append({'text':sent.text, 'start':sent.start, 'end': sent.end})
             resultObject.spaCyDoc = doc
             return resultObject

    def getLanguage(self, _text):
        if _text == None or _text == "":
            return None
        else:
            doc = self.nlpEn(_text)
            resultObject = TextAnalysis()
            resultObject.sentences = []
            # document level language detection. Think of it like average language of the document!
            if (doc._.language['language'] != 'en'):
                doc = self.nlpEs(_text)
                if (doc._.language['language'] != 'es'):
                    raise Exception("Unknow document language")
                else:
                    resultObject.language = 'es'
            else:
                resultObject.language = 'en'

            return resultObject

    def getEntitiesSentences(self, _text):
         if _text == None or _text == "":
             return None
         else:
             doc = self.nlpEn(_text)
             resultObject = TextAnalysis()
             resultObject.entities = []
             # document level language detection. Think of it like average language of the document!
             if (doc._.language['language']!= 'en'):
                 doc = self.nlpEs(_text)
                 if (doc._.language['language']):
                     raise Exception("Unknow document language")
                 else:
                     resultObject.language = 'es'
             else:
                 resultObject.language = 'en'

             assert doc.has_annotation("SENT_START")
             for ent in doc.ents:
                 print(ent.text, ent.start_char, ent.end_char, ent.label_)
                 resultObject.entities.append(ent)
             return resultObject


    def getAspectCanditateInSenteces(self,_text):
        """

        :param _text:  Sentence texts
        :return:  A list of word in sentences wich can be
                  non, adjetive, aspect lexicon member or using
                  a predictive aspect extraction computational model

                  Each element in list has text, type, pos in sentence,
        """
        # # print ("********* spaCy automatically detects noun-phrases as well  ********")
        if _text == None or _text == "":
            return None
        else:
            doc = self.nlpEn(_text)
            resultObject = TextAnalysis()
            resultObject.aspects = []
            # document level language detection. Think of it like average language of the document!
            if (doc._.language['language'] != 'en'):
                doc = self.nlpEs(_text)
                if (doc._.language['language']):
                    raise Exception("Unknow document language")
                else:
                    resultObject.language = 'es'
            else:
                resultObject.language = 'en'

            #POST TAG LIST
            #https://universaldependencies.org/u/pos/
            #
            #ADJ: adjective
            # ADP: adposition
            # ADV: adverb
            #
            ASPECT_TAGS = ['ADJ','ADV ']
            #https://spacy.io/usage/linguistic-features#pos-tagging
            for token in doc:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                      token.shape_, token.is_alpha, token.is_stop)
                if token.pos_ in ASPECT_TAGS:
                   resultObject.aspects.append(token)

            # Noun chunks
            # https://spacy.io/api/doc#noun_chunks
            # Noun chunks are “base noun phrases” – flat phrases
            # that have a noun as their head.
            # You can think of noun chunks as a noun plus the words
            # describing the noun – for example,
            #     “the lavish green grass” or “the world’s largest
            #     tech fund”.
            # To get the noun chunks in a document, simply iterate
            # over Doc.noun_chunks
            for chunk in doc.noun_chunks:
                print(chunk.text, chunk.label_, chunk.root.text)
                resultObject.aspects_noun.append(chunk)
            resultObject.spacyDoc = doc
        return resultObject


    def getProcessText(self, _text):
         if _text == None or _text == "":
             return None
         else:
             doc = self.nlpEn(_text)
             resultObject = TextAnalysis()
             resultObject.sentences = []
             resultObject.aspects = []
             resultObject.entities = []
             resultObject.aspects_noun = []

             # document level language detection. Think of it like average language of the document!
             if (doc._.language['language'] != 'en'):
                 doc = self.nlpEs(_text)
                 if (doc._.language['language'] != 'es'):
                     print(" Error Language is by:")
                     #print(_text)
                     #raise Exception("Unknow document language")
                     resultObject.language = 'unknow'
                 else:
                     resultObject.language = 'es'
             else:
                 resultObject.language = 'en'

             #assert doc.has_annotation("SENT_START")

             #Spacy split in sentence the text
             mapForSentence = dict()
             for pos, sent in enumerate (doc.sents):
                 print(sent.text)
                 objectSent = doc[sent.start:sent.end]
                 resultObject.sentences.append({'text':sent.text, 'start':objectSent.start_char, 'end': objectSent.end_char, 'entities':[],'aspects':[],'aspects_noun':[]})
                 mapForSentence[pos] = (objectSent.start_char, objectSent.end_char)
             resultObject.spaCyDoc = doc

             #assert doc.has_annotation("SENT_START")
             #Spacy split entities in text
             for ent in doc.ents:
                 print(ent.text, ent.start_char, ent.end_char, ent.label_)
                 sentencePos = hash_EntiyInFunction(mapForSentence, ent.start_char, ent.end_char)
                 resultObject.sentences[sentencePos]['entities'].append(ent)
                 resultObject.entities.append(ent)


             #Associated sentences with aspect
             # POST TAG LIST
             # https://universaldependencies.org/u/pos/
             #
             # ADJ: adjective
             # ADP: adposition
             # ADV: adverb
             #
             ASPECT_TAGS = ['ADJ', 'ADV ']
             # https://spacy.io/usage/linguistic-features#pos-tagging
             interesToken = [ token for token in doc if token.pos_ in ASPECT_TAGS]
             for token in interesToken:
                 print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                       token.shape_, token.is_alpha, token.is_stop)
                 sentencePos = hash_EntiyInFunction(mapForSentence, token.idx, token.idx + len(token.text))
                 resultObject.sentences[sentencePos]['aspects'].append(token)
                 resultObject.aspects.append(token)

             # Noun chunks
             # https://spacy.io/api/doc#noun_chunks
             # Noun chunks are “base noun phrases” – flat phrases
             # that have a noun as their head.
             # You can think of noun chunks as a noun plus the words
             # describing the noun – for example,
             #     “the lavish green grass” or “the world’s largest
             #     tech fund”.
             # To get the noun chunks in a document, simply iterate
             # over Doc.noun_chunks
             for chunk in doc.noun_chunks:
                 print(chunk.text, chunk.label_, chunk.root.text)
                 sentencePos = hash_EntiyInFunction(mapForSentence, chunk.start_char, chunk.end_char)
                 resultObject.sentences[sentencePos]['aspects_noun'].append(chunk)
                 resultObject.aspects_noun.append(chunk)
             resultObject.spacyDoc = doc


             return resultObject

