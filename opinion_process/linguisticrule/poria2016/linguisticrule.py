from senticnet.senticnet import SenticNet
from nltk.corpus import stopwords
from itertools import repeat
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


class LinguisticRule:
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

    def __init__(self, nodes, root, deepLearnigAspect) -> None:
        """
        
        :param nodes: A sentence CoreNLP Sanford Parser Dependecy Tree 
        :param root:  A sentence root in a Dependency Tree 
        :param deepLearnigAspect: A list of word positions result of the deep 
                                 learning algorithm that identifies which are aspects (1) 
        """

        self.nodes = nodes
        self.root = root
        self.algDLearningList = deepLearnigAspect
        self.sentinect = SenticNet()
        self.stopwordAnalizer = set(stopwords.words('english'))    #OOOOOOOOOOOOOOOOOJJJJJJJJJJJJJJOOOOOOOOOOOOOOOOOOOOOO
                                                              #Set the stopwordAnalyzer to englis. It must be possible
                                                              #configure

        self.AMOD = "amod" #"Adverbial modifier"
        self.ADVMOD = "advmod" #"Adjective modifier"
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

    def initialiceAspectList(self):
        """
          Initialice a aspect list if node data estructure is null different
        :return: An empty aspect list or None (if Dependency parser data estructure is null)
        """

        if self.nodes == None:
            return None
        aspectInSentResult = list(repeat(0, len(self.nodes.nodes.keys())))  # list result word positions in a sentence
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
        aspectInSentResult = list(repeat(0, len(self.nodes.nodes.keys()))) #list result word positions in a sentence
                                                                     #  that identifies which are aspects
        rootTagList = self.nodes.root[self.WORD_DEPS].keys()
        # First get root t word to Rule 1
        #"Verb or principal word in the sentence")
        listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        mainWordInSent = listOfWord[self.SENT_ROOT][0]
        rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]

        for iKey in rootTagList:
            # Let a noun h be a subject of a word t
            if iKey == self.SENT_SUBJECT:
                #"Nominal subject relation")
                subjectKey = self.nodes.root[self.WORD_DEPS][iKey][0]
                #"Subject relation HEAD")
                #print(self.nodes.nodes[subjectKey])
                #"Morphological tag")
                nodeTag = self.nodes.nodes[subjectKey][self.WORD_TAG]
                #Relation for main subject word
                depsDict = self.nodes.nodes[subjectKey][self.WORD_DEPS]
                #"Relations:")
                subListRela = depsDict.keys()
                # print(subListRela)
                # print(depsDict.values())


                # which has an adverbial or adjective modifier present in a large
                # sentiment lexicon, SenticNet
                for iKeyRel in subListRela:
                    #if an adverbial modifier or adjectival modifier
                    if iKeyRel == self.AMOD or iKeyRel == self.ADVMOD:
                        #"Exist an adverbial or adjectival modifier")
                        iKeyPossibleAdver = depsDict[iKeyRel][0]
                        #print(iKeyPossibleAdver)
                        possibleAdverbialRel = self.nodes.nodes[iKeyPossibleAdver]
                        #print(possibleAdverbialRel)
                        # If Adjectival or modifier
                        wordIs = possibleAdverbialRel[self.WORD_LEMMA]
                        #print("Search Adverbial or Adjectival " + wordIs + " in Senticnet")
                        try:
                            wordInSenctinet = self.sentinect.concept(wordIs)
                            if wordInSenctinet != None:
                                #Set 1 as Aspect identifier in word position in a sentence
                                aspectInSentResult[self.nodes.nodes[subjectKey][self.WORD_ADDRESS]] = 1
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
        aspectInSentResult = list(
            repeat(0, len(self.nodes.nodes.keys())))  # list result word positions in a sentence
        #  that identifies which are aspects
        rootTagList = self.nodes.root[self.WORD_DEPS].keys()
        # First get root t word to Rule 1
        # "Verb or principal word in the sentence")
        listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        mainWordInSent = listOfWord[self.SENT_ROOT][0]
        rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]

        # Except when the sentence has an auxiliary verb, such as is,
        # was, would, should, could, etc., we apply
        for iKey in rootTagList:
            # Let a noun h be a subject of a word t
            if iKey == self.WORD_AUX:
                return None

        subjectKey = None
        relationKeyObj = None
        subjectKeyObject = None
        rule21 = rule22 = False
        rule22Object = None
        #If the verb t is modified by an adjective or adverb or is in adverbial clause
        #modifier relation with another token, then mark h as an aspect.
        for iKey in rootTagList:
            # Let a noun h be a subject of a word t
            if iKey == self.SENT_SUBJECT:
                # "Nominal subject relation")
                subjectKey = self.nodes.root[self.WORD_DEPS][iKey][0]
                # "Subject relation HEAD")
                #print(self.nodes.nodes[subjectKey])
                # "Morphological tag")
                subjectKeyObject = self.nodes.nodes[subjectKey]
                nodeTag = subjectKeyObject[self.WORD_TAG]

            elif iKey == self.ADVMOD or iKey == self.AMOD or iKey == self.OBJ:
                relationKey = self.nodes.root[self.WORD_DEPS][iKey][0]
                relationKeyObj = self.nodes.nodes[relationKey]
                relationKeyTag = relationKeyObj[self.WORD_TAG]
                if iKey == self.OBJ:
                  wordInRelation = relationKeyObj[self.WORD_LEMMA]
                  if isaNoun(relationKeyTag) and isInSentinet(self.sentinect, wordInRelation) :
                        rule22 = True
                        rule22Object = relationKeyObj
                if isAdjective(relationKeyTag):
                    rule21 = True
        nodeTag = ""
        dependencyObject = None

        if subjectKeyObject == None:
            return subjectKey
        elif relationKeyObj != None and (rule21 == True or rule22 == True):
            if rule22 == True:#Verify if obj relation a noun
                nodeTag = rule22Object[self.WORD_TAG]
                dependencyObject = rule22Object

            elif rule21 == True:
            # Verify it is a noun
              nodeTag = subjectKeyObject[self.WORD_TAG]
              dependencyObject = subjectKeyObject

            if not (isaNoun(nodeTag)):
                # Relation for main subject word
                return None
            else:
                aspectInSentResult[dependencyObject[self.WORD_ADDRESS]] = 1
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
        aspectInSentResult = list(
            repeat(0, len(self.nodes.nodes.keys())))  # list result word positions in a sentence
        #  that identifies which are aspects
        rootTagList = self.nodes.root[self.WORD_DEPS].keys()
        # First get root t word to Rule 1
        # "Verb or principal word in the sentence")
        listOfWord = self.nodes.nodes[0][self.WORD_DEPS]
        mainWordInSent = listOfWord[self.SENT_ROOT][0]
        rootWord = self.nodes.nodes[mainWordInSent][self.WORD_LEMMA]

        subjectKey = None
        relationKeyObj = None
        subjectKeyObject = None
        rule3 = False

        # If the verb t is modified by an adjective or adverb or is in adverbial clause
        # modifier relation with another token, then mark h as an aspect.
        for iKey in rootTagList:
            # Let a noun h be a subject of a word t
            if iKey == self.SENT_SUBJECT:
                # "Nominal subject relation")
                subjectKey = self.nodes.root[self.WORD_DEPS][iKey][0]
                # "Subject relation HEAD")
                # print(self.nodes.nodes[subjectKey])
                # "Morphological tag")
                subjectKeyObject = self.nodes.nodes[subjectKey]


            elif iKey == self.COP:
                relationKey = self.nodes.root[self.WORD_DEPS][iKey][0]
                relationKeyObj = self.nodes.nodes[relationKey]
                rule3 = True
        nodeTag = ""
        dependencyObject = None

        if subjectKeyObject == None:
            return subjectKey
        elif relationKeyObj != None and rule3 == True:
            nodeTag = subjectKeyObject[self.WORD_TAG]
            dependencyObject = subjectKeyObject
            if not (isaNoun(nodeTag)):
                # Relation for main subject word
                return None
            else:
                aspectInSentResult[dependencyObject[self.WORD_ADDRESS]] = 1
        else:
            return None
        return aspectInSentResult

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

        aspectInSentResult = list(
            repeat(0, len(self.nodes.nodes.keys())))  # list result word positions in a sentence
        #Analized Deep Learning  Aspect selection
        rule4 = False
        sizeDeepLearningSelection = 0
        sizeLingRuleAspectSelect = 0
        if self.algDLearningList != None and len(self.algDLearningList) > 0:
            sizeDeepLearningSelection = len(self.algDLearningList)

        if _liguistiRuleSelctAspect != None and len(_liguistiRuleSelctAspect) > 0:
            sizeLingRuleAspectSelect = len(_liguistiRuleSelctAspect)

        listForElement = sizeDeepLearningSelection if sizeDeepLearningSelection > 0 else sizeLingRuleAspectSelect

        def evaluateCompundElement ( iElement):
            #Inner function for search compund element in a sentence
            objectInDepTree = self.nodes.nodes[iElement]
            nodeTag = objectInDepTree[self.WORD_TAG]
            rule4 = False
            if isaNoun(nodeTag):
                rootTagList = objectInDepTree[self.WORD_DEPS].keys()
                for iKey in rootTagList:
                    if iKey == self.COMPOUND:
                        compoundKey = objectInDepTree[self.WORD_DEPS][iKey][0]
                        relatedCompundObject = self.nodes.nodes[compoundKey]
                        relatedCompundTag = relatedCompundObject[self.WORD_TAG]
                        if isaNoun(relatedCompundTag):  # Then marked as an Aspect
                            iPos = relatedCompundObject[self.WORD_ADDRESS]
                            aspectInSentResult[iPos] = 1
                            rule4 = True
            return rule4

        for iElement in range( listForElement):
                if (self.algDLearningList != None and self.algDLearningList[iElement] == 1) or (_liguistiRuleSelctAspect != None and _liguistiRuleSelctAspect[iElement] == 1): #It is selected as an aspect by Deep Learnig algorithm
                    aspectInSentResult[iElement] = 1
                    rule4 =evaluateCompundElement(iElement)


        if rule4 == False:
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
        if len(self.nodes.nodes.keys()) != len(_listAspectInSentence):
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

        resultAspectDependcyAux = zip(self.nodes.nodes.values(),aspectInSentence)  #Merge in a tuple Depency Parser info and
                                                                                         #aspect identification
        resultAspectDependcy = []

        for iElement in resultAspectDependcyAux:
            #Delete all stopword element. Use compresion list when we know how accecing to tuples elements
            depencyObject, isAspect = iElement
            wordLemma = depencyObject[self.WORD_LEMMA]
            if wordLemma != None and not wordLemma in self.stopwordAnalizer:
                resultAspectDependcy.append((depencyObject,isAspect))

        return resultAspectDependcy

    def applyRules(self):
        """
         Apply all rules over nodes and root for getting 
         aspect in a sentence.
         When apply each rule an aspect list is updated
        :return: A dictionary of word positions that identifies which are aspects
        """

        aspectInSentResult = self.initialiceAspectList();
        if aspectInSentResult == None:
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
            depNode, aspect = iValue
            if aspect == 1:
                isAspect = True

        if not isAspect:
            return None
        #Delete end punctuction elements if appear in last element
        lastElement = aspectInSentTuples[-1]
        if lastElement[0][self.WORD_RELATION] == self.SENT_PUNCT:
            aspectInSentTuples = aspectInSentTuples[:-1]

        #Create dictionary with word and aspect identificador 1/0
        resultDictionary = {}
        for elemt in aspectInSentTuples:
            word = elemt[0][self.WORD_TOKEN]
            #if word is not a stopword then put in dictionary
            punctuation = [c for c in word if c in string.punctuation]
            # Remove numbers
            numbers = [c for c in word if c in '0123456789']

            if  not word in self.stopwordAnalizer and len(punctuation) == 0 and  len(numbers) == 0:
             resultDictionary[word] = elemt[1]


        #endAspectInSentTpl = [ elem for elem in endAspectInSentTpl if elem[0][self.WORD_TAG] != self.PUNTUA]

        return resultDictionary

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


