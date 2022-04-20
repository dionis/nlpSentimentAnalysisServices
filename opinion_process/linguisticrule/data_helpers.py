import numpy as np
import re
import string
import itertools
from collections import Counter
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
#import corpusprocessing
from tensorflow.contrib import learn
MAX_SIZE = 5
TAG_VECTOR_SIZE = 6 #Size for POS tagger vector
import nltk
from nltk.corpus import stopwords
vocabulary_size = 10000
vocab_processor = None
stops = stopwords.words('english')

#POS tags set
#noun, verb, adjective,adverb, preposition, conjunction
NOUN_TAGS = ["NN", "NNS"]      #Noun set tags
VERB_TAGS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ" ] #Verb set tags
ADJ_TAGS = ["AFX", "JJ", "JJR", "JJS"] #Adjective set tags
ADV_TAGS = ["RB", "RBR", "RBS", "WRB"] #Adverb set tags
PREP_TAGS = ["IN"]                     #Preoposition set tags
CONJ_TAGS = ["CC"]                     #Conjunction set tags
####-----------------------------------------------------------------------
#####
#
# We must also determine the maximum sentence size. To do this, we look at a
#histogram of text lengths in the data set. We see that a good cut-off might be around
#25 words. Use the following code:
#
#####

def maximunSentenceSize(texts,sentenceSizes):
    # Plot histogram of text lengths
    text_lengths = []
    for x in texts:
        if len(x)>0:
            text_lengths.append(x)

    texts = text_lengths
    new_text = []
    for x in texts:
        for c in x:
            new_text.append(c)

    if type(texts[0]) == str:
        text_lengths =  [len(x.split()) for x in texts]

    texts = [' '.join(c for c in x if c not in string.punctuation) for x in new_text]
    sentence_size = max(sentenceSizes)
    #sentence_size = 65
    min_word_freq = 3

    #TensorFlow has a built-in processing tool for determining vocabulary embedding,
    #called  VocabularyProcessor() , under the  learn.preprocessing library:
    #la oracion de mayor tamano
    vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
    return vocab_processor
#-----------------------------------------------------------------------------------

def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = sentences
    if type(sentences[0]) == str:
      split_sentences = [s.split() for s in sentences]

    words = [x for sublist in split_sentences for x in sublist]

    # Initialize list of [word, word_count] for each word,
    #starting with unknown
    count = [['RARE', -1]]
    # Now add most frequent words, limited to the N-most frequent
    #(N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then
    #make it the value of the prior dictionary length
    for word, word_count in count:
      word_dict[word] = len(word_dict)
    return(word_dict)
####------------------------------------------------------------------------------------------------
#####
#  OJOO USO de NLTK
# In order to use scikt-learn's TF-IDF processing functions, we have to tell it how to
# tokenize our sentences. By this, we just mean how to break up a sentence into the
# corresponding words. A great tokenizer is already built for us in the  nltk package
# that does a great job of breaking up sentences into the corresponding words:
#
####
def tokenizer(text):
    words = []
    try:
      words = nltk.word_tokenize(text)
    except TypeError:
      print (TypeError)
    return words
####------------------------------------------------------------------------------------------------

def removelast_pointcharacter(text):
    """Delete in a sentences a last character if it is  a point or other sysmbol"""
    if text != None and type(text) == str:
        lastChar = text[-1]
        if lastChar in string.punctuation:
            return text[:-1]
    return text
#-------------------------------------------------------------------------------------
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation)
    for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x
    in texts]
    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in
    (stops)]) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    # Remove empty item
    texts = [x for x in texts if x != '']
    return(texts)
#-------------------------------------------------------------------------------------
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# def load_data_and_labels(all_data_file):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     # What to do if the file is currently being edited by someone. A lock has been set on the file. So keep A backup that is swapped in and out.
#     intent = []
#     all_examples = list(open(all_data_file, "r").readlines())
#     x_text = []
#     for example in all_examples:
#         #splitIndex = example.index("-")
#         splitIndex = example.index("#")
#         current_intent = example[0:splitIndex]
#         current_intent_index = intent.index(current_intent) if current_intent in intent else -1
#         if current_intent_index == -1:
#             intent.append(current_intent)
#         x_text.append(example[splitIndex+1:].strip())
#
#
#     number_of_intents = len(intent)
#     one_hot_vector = np.ndarray(shape=(number_of_intents,number_of_intents))
#     for encode_column in range(0,number_of_intents):
#         #one_hot_vector_row = []
#         for encode_row in range(0,number_of_intents):
#             if encode_column == encode_row:
#                 one_hot_vector[encode_row][encode_column] = 1
#             else:
#                 one_hot_vector[encode_row][encode_column] = 0
#         #one_hot_vector.append(one_hot_vector_row)
#     print(one_hot_vector)
#     x_text = [clean_str(sent) for sent in x_text]
#     y = np.ndarray(shape=(len(x_text),number_of_intents))
#     index = 0
#     for example in all_examples:
#         splitIndex = example.index("-")
#         current_intent = example[0:splitIndex]
#         y[index] = one_hot_vector[intent.index(current_intent)]
#         index = index + 1
#     print(type(y))
#     print(y)
#     print(y[0])
#     print(type(y[0]))
#     #quit()
#     '''positive_examples = list(open(positive_data_file, "r").readlines())
#     positive_examples = [s.strip() for s in positive_examples]
#     negative_examples = list(open(negative_data_file, "r").readlines())
#     negative_examples = [s.strip() for s in negative_examples]
#     # Split by words
#     x_text = positive_examples + negative_examples
#     x_text = [clean_str(sent) for sent in x_text]
#     # Generate labels
#     positive_labels = [[0, 1] for _ in positive_examples]
#     negative_labels = [[1, 0] for _ in negative_examples]
#     y = np.concatenate([positive_labels, negative_labels], 0)'''
#
#
#     if len(x_text) == len(y):
#         return [x_text, y]
#     else:
#         print("The length of the training labels and the labels assigned to them do not match")
#         quit();

def load_data_and_labels(all_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # What to do if the file is currently being edited by someone. A lock has been set on the file. So keep A backup that is swapped in and out.
    intent = []
    all_examples = list(open(all_data_file, "r").readlines())
    x_text = []
    y_text = []
    for example in all_examples:
        #splitIndex = example.index("-")
        print("Example: " + example)
        splitIndex = example.find("#")
        current_intent = example[0:splitIndex]
        x_text.append(example[splitIndex+1:].strip())
        y_text.append(current_intent)

    #x_text = [isentence.split() for isentence in x_text]
    #y_text = [isentence.split() for isentence in y_text]
    # y = np.ndarray(shape=(len(x_text)))
    # for index in enumerate(x_text):
    #     y[index] = y_text[index]

    # number_of_intents = len(intent)
    # one_hot_vector = np.ndarray(shape=(number_of_intents,number_of_intents))
    # for encode_column in range(0,number_of_intents):
    #     #one_hot_vector_row = []
    #     for encode_row in range(0,number_of_intents):
    #         if encode_column == encode_row:
    #             one_hot_vector[encode_row][encode_column] = 1
    #         else:
    #             one_hot_vector[encode_row][encode_column] = 0
    #     #one_hot_vector.append(one_hot_vector_row)
    # print(one_hot_vector)
    # x_text = [clean_str(sent) for sent in x_text]
    # y = np.ndarray(shape=(len(x_text),number_of_intents))
    # index = 0
    # for example in all_examples:
    #     splitIndex = example.index("-")
    #     current_intent = example[0:splitIndex]
    #     y[index] = one_hot_vector[intent.index(current_intent)]
    #     index = index + 1
    # print(type(y))
    # print(y)
    # print(y[0])
    # print(type(y[0]))
    #quit()
    '''positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)'''


    if len(x_text) == len(y_text):
        return [x_text, y_text]
    else:
        print("The length of the training labels and the labels assigned to them do not match")
        quit();

def load_data_and_labelsEx(all_data_file, type, address):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # What to do if the file is currently being edited by someone. A lock has been set on the file. So keep A backup that is swapped in and out.
    type = "semeval2014"
    dataset, vocab_processor = corpusprocessing.processCorpus(type, address)
    x_text = []
    y = []
    if dataset != None and vocab_processor.vocabulary_ != None:
        for idataset in dataset:
            for isentence in idataset.sentences:
                if isentence.tagList != None and len(isentence.tagList) > 0:
                    #Only sentences useful to train
                    x_text.append(isentence)
                    y.append(isentence.tagList)

    # intent = []
    # all_examples = list(open(all_data_file, "r").readlines())
    # x_text = []

    # for example in all_examples:
    #     splitIndex = example.index("-")
    #     current_intent = example[0:splitIndex]
    #     current_intent_index = intent.index(current_intent) if current_intent in intent else -1
    #     if current_intent_index == -1:
    #         intent.append(current_intent)
    #     x_text.append(example[splitIndex+1:].strip())
    # number_of_intents = len(intent)
    # one_hot_vector = np.ndarray(shape=(number_of_intents,number_of_intents))
    # for encode_column in range(0,number_of_intents):
    #     #one_hot_vector_row = []
    #     for encode_row in range(0,number_of_intents):
    #         if encode_column == encode_row:
    #             one_hot_vector[encode_row][encode_column] = 1
    #         else:
    #             one_hot_vector[encode_row][encode_column] = 0
    #     #one_hot_vector.append(one_hot_vector_row)
    # print(one_hot_vector)
    # x_text = [clean_str(sent) for sent in x_text]
    # y = np.ndarray(shape=(len(x_text),number_of_intents))
    # index = 0
    # for example in all_examples:
    #     splitIndex = example.index("-")
    #     current_intent = example[0:splitIndex]
    #     y[index] = one_hot_vector[intent.index(current_intent)]
    #     index = index + 1
    # print(type(y))
    # print(y)
    # print(y[0])
    # print(type(y[0]))
    # #quit()
    # '''positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)'''

    list_sentence = []
    for isentence in x_text:
        list_sentence.append(''+isentence.text )

    #Index sentence by position in dictonary
    listsentence_vocab = list(vocab_processor.fit_transform(list_sentence))
    # if len(listsentence_vocab) != len(x_text):
    #     print("The index sentences and sentences do not match")
    #     quit();
    # else:
    for ipos, isentence in  enumerate(x_text):
            isentence.idenxSentece = list_sentence[ipos]

    if len(x_text) == len(y):
        return [x_text, y, vocab_processor]
    else:
        print("The length of the training labels and the labels assigned to them do not match")
        quit();


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_iter_simple(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    in each sentence for each word update context windowS
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    # Get words windows for each word in a sentences
    # Remenber special traitmen for first and las word
    # and sentences for size iqual to windows (3)
    for ix_train in x_dev:
        if len(ix_train) <= WORD_WINDOWS_SIZE:
            x_train_aux.append(ix_train)
        else:
            # Modify first word in a sentences obtein windows context
            # and paddin 0 index in first and last word in a sentences
            # in left/right direction

            windows_words = [ix_train[i:i + WORD_WINDOWS_SIZE] for i in range(len(ix_train) - 1)]

            # First word
            first_word = ix_train[0:int(WORD_WINDOWS_SIZE / 2) + 1]
            first_word_padding = [0 for i in range(int(WORD_WINDOWS_SIZE / 2))]
            if len(first_word) < WORD_WINDOWS_SIZE:
                first_word = list(first_word_padding) + list(first_word)

            # Last word
            last_word_window = windows_words[-1]
            if len(last_word_window) < WORD_WINDOWS_SIZE:
                last_word_window = list(last_word_window) + list(first_word_padding)
                last_post = len(windows_words)
                windows_words[last_post - 1] = np.array(last_word_window)

            # Add firts word window and al sentences
            new_windows_words = []
            new_windows_words.append(np.array(first_word))
            for iwin_word in windows_words:
                new_windows_words.append(iwin_word)
            windows_words = new_windows_words

            x_train_aux.append(windows_words)

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def batch_iterEx(datasets, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data = np.array(datasets)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield  shuffled_data[start_index:end_index]


def preprocessingText(text):
    return normalize_text(tokenizer(text), stops)


def findwordembeddings(databaseTool, each_sencente, wordembeddings_dimention, type="MongoDb"):
    """
      Find word embeddings in file or in data base
    :param databaseTool: Database client
    :param each_sencente: Sentences object with word information 
    :param type: Database type MongoDb, MySQL or File
    :return: A Word Embedding matrix in sentences words order
    """
    wordembeddingList = None
    if each_sencente != None and databaseTool != None:
        wordembeddingList = []
        if each_sencente.tokenSentence != None and len(each_sencente.tokenSentence) > 0:
            for iword in each_sencente.tokenSentence:
                wordembeddings = databaseTool.find(iword)       #Search in database or in File
                                                                #databaseTool must be an abstract object
                                                                #with designed pattron for Data Base or File
                                                                #word embeddings controls
                if wordembeddings != None:
                     #Not inserte in a result list if not exit's word embeddings
                     wordembeddingList.append(wordembeddings)
                else:
                    emptyWordEmbeddings = np.zeros(wordembeddings_dimention)
                    wordembeddingList.append(emptyWordEmbeddings.tolist())    #Not exist information in Data Base
        return wordembeddingList
    return wordembeddingList


def createOneHotVector(word, wordPosTagger):
    postTagVector = one_hot_vector = np.zeros(shape=(TAG_VECTOR_SIZE))
    ipos = 0
    #noun, verb, adjective,adverb, preposition, conjunction
    if wordPosTagger in NOUN_TAGS:
            postTagVector[0] = 1.0
    elif wordPosTagger in VERB_TAGS:
            postTagVector[1] = 1.0
    elif wordPosTagger in ADJ_TAGS:
            postTagVector[2] = 1.0
    elif wordPosTagger in ADV_TAGS:
            postTagVector[3] = 1.0
    elif wordPosTagger in PREP_TAGS:
            postTagVector[4] = 1.0
    elif wordPosTagger in CONJ_TAGS:
            postTagVector[5] = 1.0


    return postTagVector


def appedPostTagToWordEmbddins(each_sencente):
    """
    Concat to each vector in word embeddings matrix a vector
    with POS information: noun, adjective, verb, in others
    :param each_sencente: 
    :return: 
    """
    wordembeddignslisttag = None
    if each_sencente.wordembeddignslist != None and each_sencente.depencyNodes != None:
        #If exist enough information
        wordembeddignslisttag = []
        for ipos, woremddngs in enumerate(each_sencente.wordembeddignslist):
            iworembeddingslisttag = []
            word = each_sencente.tokenSentence[ipos]     #Be careful because echa position in sentence word
                                                          #must be the same in word embeddings list and
                                                          #in Dependency parser
            wordDParser = each_sencente.depencyNodes[ipos + 1]['tag']    #Get Depency Parser word information
            wordPosTagger = ""
            one_hot_vector = createOneHotVector (word, wordPosTagger)
            iworembeddingslisttag = woremddngs
            try:
              iworembeddingslisttag.extend(one_hot_vector)
            except(AttributeError):
                print('Error')
            wordembeddignslisttag.append(iworembeddingslisttag)
        return wordembeddignslisttag
    return wordembeddignslisttag