#! /usr/bin/env python

import numpy as np
import os
import time
import datetime
#import tensorflow as tf
#from text_cnn import TextCNN




# Parameters
# ==================================================

# Model Hyperparameters
# tf.flags.DEFINE_string("glove", None, "Glove file with pre-trained embeddings (default: glove.6B.300d)")
#
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("word2vec", "/util/", "Word2vec file with pre-trained embeddings (default: None)")
# # tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
# # tf.flags.DEFINE_string("all_data_file", "./data/rt-polaritydata/CombinedDataset.train", "Combined dataset")
# tf.flags.DEFINE_string("all_data_file", "./data/ner_example/nerdataset.txt", "Combined dataset")
# # Training parameters
# tf.flags.DEFINE_integer("batch_size", 6, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 5, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many steps (default: 100)")
# # Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# FLAGS = tf.flags.FLAGS
EMBEDDING_DIMENSION=50 # Available dimensions for 6B data is 50, 100, 200, 300

#FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# Randomly shuffle data
#np.random.seed(10)
#shuffle_indices = np.random.permutation(np.arange(len(y)))
#x_shuffled = x[shuffle_indices]
#y_shuffled = y[shuffle_indices]

# Split train/testvariant set
# TODO: This is very crude, should use cross-validation
#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]



def init_preprocessig_informationExx(vocab_processor, dataset):
    #print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    #print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    dataset_size = len( [y for x in dataset for y in x.sentences] )
    cnn = TextCNN(
        sequence_length=dataset_size,
        num_classes=dataset_size,
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    if FLAGS.glove:
        # initial matrix with random uniform
        # data_directory = FLAGS.glove
        # file_name = os.path.basename(data_directory)
        # dir_file_name = os.path.dirname(data_directory)
        # glove_weights_file_path = os.path.join(data_directory, 'glove.6B.{EMBEDDING_DIMENSION}d.txt')
        #
        # if not os.path.isfile(glove_weights_file_path):
        #     # Glove embedding weights can be downloaded from https://nlp.stanford.edu/projects/glove/
        #     print("Not exists glovefile")
        # else:
        initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # load any vectors from the word2vec
        PAD_TOKEN = 0

        word2idx = {
            'PAD': PAD_TOKEN}  # dict so we can lookup indices for tokenising our text later from string to sequence of integers
        weights = []
        print("Load Glove file {}\n".format(FLAGS.glove))
        with open(FLAGS.glove, "rb") as file:

            # header = f.readline()
            # vocab_size, layer1_size = map(int, header.split())
            # binary_len = np.dtype('float32').itemsize * layer1_size
            # print(vocab_size)
            # quit()
            # for line in range(vocab_size):
            #     print(line)
            #     word = []
            #     while True:
            #         ch = f.read(1).decode('latin-1')
            #         if ch == ' ':
            #             word = ''.join(word)
            #             break
            #         if ch != '\n':
            #             word.append(ch)
            #     print(word)
            #     idx = vocab_processor.vocabulary_.get(word)
            #     print("value of idx is" + str(idx));
            #     if idx != 0:
            #         print("came to if");
            #         initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            #     else:
            #         print("came to else");
            #         f.read(binary_len)

            binary_len = np.dtype('float32').itemsize * int(FLAGS.embedding_dim)
            for index, line in enumerate(file):
                values = line.split()  # Word and weights separated by space
                word = []
                while True:
                    ch = file.read(1).decode('latin-1')
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                # print(word)
                idx = vocab_processor.vocabulary_.get(word)
                # print("value of idx is" + str(idx));
                if idx != 0:
                    print("came to if");
                    initW[idx] = np.fromstring(file.read(binary_len), dtype='float32')
                    print(word)
                else:
                    print("came to else");
                    file.read(binary_len)
            cnn.W.assign(initW)
            print("Ended")
    if FLAGS.word2vec:
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # load any vectors from the word2vec
        print("Load word2vec file {}\n".format(FLAGS.word2vec))

        with open(FLAGS.word2vec, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            print(vocab_size)
            # quit()
            for line in range(vocab_size):
                print(line)
                word = []
                while True:
                    ch = f.read(1).decode('latin-1')
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                print(word)
                idx = vocab_processor.vocabulary_.get(word)
                print("value of idx is" + str(idx));
                if idx != 0:
                    print("came to if");
                    initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    print("came to else");
                    f.read(binary_len)
            cnn.W.assign(initW)
            print("Ended")



def init_preprocessig_information(vocab_processor, dataset, word_embedding, word_embedding_dimension,
                                              word_embedding_address):
        if os.path.exists(word_embedding_address) == False:
            raise Exception("Not word embedding or glove correct address")
        else:
            if word_embedding == "word2vec":
                FLAGS.word2vec = str(word_embedding_address);
            elif word_embedding == "glove":
                FLAGS.glove = word_embedding_address;
            if type(word_embedding_dimension) == int and word_embedding_dimension >= 100:
                FLAGS.embedding_dim = word_embedding_dimension



           # #Prepare the CNN Algorithm assing datas
           # return init_preprocessig_informationEx(vocab_processor, dataset)