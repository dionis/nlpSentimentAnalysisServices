import os
import re


class Term:
    def __init__(self):
        self.source = 0;
        self.target = 0;
        self.wordpolarity = 0;
        self.categorypolarity = 0;
        self.category = "";
        self.word = "";
        self.text = "";

        # Word Embedding asociated to category word
        self.categoryembeddings = [];

        # Word Embedding asociated to term word
        self.wordembedding = [];

        # def __setattr__(self, name: str, value: None) -> None:
        #     super().__setattr__(name, value)
        #     if name == "source":
        #         self.source = value;
        #     elif name == "target":
        #         self.target = value;
        #     elif name == "wordpolarity":
        #         self.target = value;
        #     elif name == "categorypolarity":
        #         self.target = value;
        #     elif name == "category":
        #         self.target = value;
        #     elif name == "word":
        #         self.target = value;
        #     return None


class Sentence:
    def __init__(self):
        self.source = 0;
        self.target = 0;
        self.wordpolarity = 0;
        self.categorypolarity = 0;
        self.category = "";
        self.word = "";
        self.text = "";

        # Word Embedding asociated to category word
        self.categoryembeddings = [];

        # Word Embedding asociated to term word
        self.wordembedding = [];
        self.id = 0;
        self.text = "";
        self.aspectTerms = [];
        self.aspectCategory = "";
        self.wordsembedding = [];
        self.wordslist = list();
        self.entity = "";
        self.category = "";
        self.tagList = [];
        self.preprocessText = [];
        self.wordembeddignslist = [];
        self.wordembeddignslisttag = [];
        self.tokenSentence = []

        # ________________-------------------------------------____________________________

    def addAspectTerm(self, term):
        self.aspectTerms.append(term)

    def getAspectTerm(self):
        return self.aspectTerms

    # ________________-------------------------------------____________________________

    def addAspectCategory(self, newcategory):
        self.aspectCategory.append(newcategory)

    # ________________-------------------------------------____________________________
    def insertAspectData(self, source, target, word, polarity, entity, wordembeddings):
        term = Term();
        term.source = source;
        term.target = target;
        term.word = word;
        term.wordpolarity = polarity;
        term.wordembeddings = wordembeddings;
        term.entity = entity
        self.addAspectTerm(term);

    # ________________-------------------------------------____________________________
    def insertAspectDataCategory(self, source, target, word, polarity, entity, category, wordembeddings):
        term = Term();
        term.source = source;
        term.target = target;
        term.word = word;
        term.wordpolarity = polarity;
        term.wordembeddings = wordembeddings;
        term.entity = entity
        term.category = category
        self.addAspectTerm(term);

    # ________________-------------------------------------____________________________
    def inserCategory(self, wordcategory, polarity, wordembeddings):
        term = Term();
        term.category = wordcategory;
        term.categorypolarity = polarity;
        term.categoryembeddings = wordembeddings;

    # ________________-------------------------------------____________________________
    # Note: For make that we need process all train or testvariant examples give id vocabulary and load some trained or pre-trained word embedding

    def processingText(self):
        if self.text == "":
            return list()
        else:  # Tokenize text and find wordembedding
            raise Exception("Not implemeted yet")
        # ________________-------------------------------------____________________________
        # Note: Search in word embeddig global or persistent asoaciate vector for each word

    def getWordEmbedding(self, wordlist):
        resultEmbedding = []
        if wordlist == "" or wordlist == None:
            return None
        elif type(wordlist) == "vector" and wordlist.len > 0:
            for iword in wordlist:  # Search asociated vector
                resultEmbedding.append(iword);
        return resultEmbedding

    # ________________-------------------------------------____________________________
    def returnWordEmbeding(self):
        if self.wordsembedding == None:
            wordlist = self.processingText()
            return self.getWordEmbedding(wordlist)


class Review:
    def __init__(self):
        self.annotadorid = "";
        self.author = "";
        self.date = "";
        self.hotelid = "";
        self.reviewid = "";
        self.ratingBusiness = "";
        self.ratingChekin = "";
        self.ratingClealiness = "";
        self.ratingLocation = "";
        self.ratingOverall = ""
        self.ratingRoom = "";
        self.ratingServices = "";
        self.ratingValue = ""
        self.sentences = [];
        self.document = [];


class Datasets:
    def __init__(self):
        self.type = "train";
        self.sentences = [];
        self.document = [];
        self.vocab_processor = None;
        self.sentencesSize = [];


# Extract xml information tags
def searchAllFiles(listDirectories, pathadress, listFiles):
    royalXmlFiles = []
    for fileAdress in listDirectories:
        if os.path.isdir(pathadress + os.sep + fileAdress):
            searchAllFiles(os.listdir(pathadress + os.sep + fileAdress), pathadress + os.sep + fileAdress, listFiles)
        else:
            if re.match("^\.", os.path.basename(fileAdress)) == None and re.match("^README", fileAdress) == None:
                listFiles.append(pathadress + os.sep + fileAdress)

    return listFiles
