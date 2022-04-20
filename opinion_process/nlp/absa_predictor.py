import logging
import argparse
import math
import os
import sys
import time
import string
from time import strftime, localtime
from datetime import datetime
import random
import numpy as np
from .utils import *

#from pytorch_pretrained_bert import BertModel

from pytorch_transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset,ABSADatasetText

from .models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LifelongABSA
from .models.aen import CrossEntropyLoss_LSR, AEN_BERT
from .models.bert_spc import BERT_SPC
from .models.lcf_bert import LCF_BERT
from .models.lcf_bert_hat import LCF_BERT_HAT
from .models.aen_hat import AEN_BERT_HAT

from ..linguisticrule.poria2016.linguisticrulespacy import LinguisticRuleSpaCy

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



class Instructor:
    def __init__(self, opt, model_classes):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        else:
            self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['tests']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))

        # self.trainset = ABSADataset(opt.dataset_file['train'], self.tokenizer)
        # self.testset = ABSADataset(opt.dataset_file['tests'], self.tokenizer)
        # assert 0 <= opt.valset_ratio < 1
        # if opt.valset_ratio > 0:
        #     valset_len = int(len(self.trainset) * opt.valset_ratio)
        #     self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        # else:
        #     self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))


        if 'bert' in opt.model_name:
            # ,cache_dir="pretrained/bert/"

            print("--------load module BERT --------")
            #To from pytorch_transformers import BertModel
            self.bert = BertModel.from_pretrained(opt.pretrained_bert_name, output_attentions=True,
                                             cache_dir="pretrained/bert/")

            # Bert pretrained (Old version)
            #bert = BertModel.from_pretrained(opt.pretrained_bert_name, cache_dir="pretrained/bert/")
            print("--------DDDD-----")
            print("OUTPUT")
            print("------   Module LOADED -------")
            #self.model = model_classes[opt.model_name](bert, opt).to(opt.device)
            self.model = opt.model_class(self.bert, opt).to(opt.device)
            #self.model = AEN_BERT(self.bert, opt).to(opt.device)
            print("MODULE LOADED SPECIFIC")
        else:
            self.model = model_classes[opt.model_name](embedding_matrix, opt).to(opt.device)

        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def backwardTransfer(accNew, ncla):
        backwardTrasfer = 0.0
        if ncla <= 2:
            return (backwardTrasfer,backwardTrasfer,backwardTrasfer)
        else:
         denominator = (ncla * (ncla - 1))/2
         i = 1
         while i < ncla:
             j = 0
             while j <= (i-1):
                 backwardTrasfer += (accNew[i][j] - accNew[j][j])
                 j +=1
             i += 1

         backwardTrasfer = backwardTrasfer / denominator
         rEm = 1- np.abs(np.min(backwardTrasfer,0))
         positiveBack = np.max(backwardTrasfer,0)
         return (backwardTrasfer, rEm, positiveBack )

    def forwardTransfer(accNew, ncla):
        forwardTrasfer = 0.0
        if ncla <= 2:
            return forwardTrasfer
        else:
            denominator = (ncla * (ncla - 1)) / 2
            for i in range(ncla):
                for j in range( ncla):
                    if i < j:
                      forwardTrasfer += accNew[i][j]

            return (forwardTrasfer/denominator)

    def globallMeasure(accNew, ncla):
        forwardTrasfer = 0.0
        if ncla <= 2:
            return forwardTrasfer
        else:
            denominator = (ncla * (ncla + 1)) / 2
            for i in range(ncla):
                for j in range(ncla):
                    if i >= j:
                        forwardTrasfer += accNew[i][j]

            return (forwardTrasfer / denominator)


    def run(self):

        if self.opt.approach == 'ar1':
            from approaches import ar1 as approach
        if self.opt.approach == 'hat-tests' or self.opt.approach == 'ar1' or self.opt.approach == 'ewc' \
                    or self.opt.approach == 'si' or self.opt.approach == 'lwf':
                  from networks import bert as network

                  # from networks import alexnet_hat_test as network
            # else:
            #     from networks import alexnet as network
    ##### End Source Code Lifelong Learning ########################

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        #It is a way to obtain variables for using in optimizer and not finned tuning Bert model
        # modelVariables = [(name,var) for i, (name, var) in enumerate(self.model.named_parameters())if name.find("bert") == -1]
        #
        # for name, var in modelVariables:
        #  print ("Variable ==> " + name)

        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)


         ##### Start Source Code Lifelong Learning ########################    # Inits

        # if self.trainset.multidomain == None or self.trainset.multidomain != True:
        #     print('Load data...')
        #     train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        #     test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        #     val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        #
        #     self._reset_params()
        #     best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        #     self.model.load_state_dict(torch.load(best_model_path))
        #     self.model.eval()
        #     test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        #     logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        # else:

        print('Inits...')
        sizesentence = 0
        ncla = 0


        inputsize = (1,0,0)



        appr = None
        net = network.Net(inputsize, None, self.opt).cuda() if torch.cuda.is_available()\
                                                                     else network.Net(inputsize, None, self.opt)
        net.set_Model(self.model)
        net.set_ModelOptimizer(optimizer)

        if torch.cuda.is_available():
                dev = "cuda:0"
                self.model.to(dev)
                net.to(dev)
                print("Update GPU(Cuda support ):" + dev )
                # utils.print_model_report(net)
        self.appr = approach.Appr(net, nepochs=self.opt.nepochs, lr=self.opt.lr, args=self.opt)

        if os.path.exists(self.opt.output_algorithm):
                self.appr.loadModel(self.opt.output_algorithm)
                print("Load Module values from: " + self.opt.output_algorithm )

        print('-' * 100)
    ##### End  Source Code Lifelong Learning ########################
        print("!!!!New optmization!!!!!")
        self.appr._set_optimizer(optimizer)
        print("-------New optmization-------")
    ##### Start Source Code Lifelong Learning ########################    # Inits
        task = 0
        # if self.opt.approach == 'lifelong':
        #     appr.setAllAspect(self.trainset.all_aspects)
        #     appr.setAllWord(self.tokenizer.word2idx)
        startDateTime = datetime.now()
        test_data_list = []


        #test_data_loader = DataLoader(dataset=self.testset[u][2], batch_size=self.opt.batch_size, shuffle=False)

        #Call classifier module, and RL detector
        #test_loss, test_acc, test_recall, test_f1 = appr.eval(0, test_data_loader)

        #################################################################
        ##Analysis the algorithm answer and input text to send return
        #################################################################


    def joinTextRestul(self,text, result):

        if text == None or result == None:
            return None
        else:
            algorithmResult = result[0][0]
            print(algorithmResult)
            posAlgortiResult = 0
            for textInstance in text:
                polarity = '1' #Neutral
                if algorithmResult[posAlgortiResult] == 0: #Negative
                    polarity = '0'
                elif algorithmResult[posAlgortiResult] == 2: #Positive
                    polarity = '2'
                    posAlgortiResult +=1
                textInstance.update({"predictpolarity": polarity})
            return text

    #
    #
    #

    def process_text(self, text):
       self.trainset = ABSADatasetText(text, self.tokenizer)
       test_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=False)
       result = self.appr.classify(0, test_data_loader)
       output_result = self.joinTextRestul(text, result)


       return output_result


def main():
    tstart = time.time()



    # Hyper Parameters
    #default='bert_spc'
    #default='bert_spc'
    #--model_name lcf_bert --approach ar1
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aen_bert', type=str)
    parser.add_argument('--dataset', default='original_algt', type=str, help='twitter, restaurant, laptop, multidomain, all_multidomain, alldevice_multidomain')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=2, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')

    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')

    # Arguments LifeLearning Code

    # parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--measure', default='recall', type=str, required=False,
                        choices=['accuracy', 'recall', 'f1'], help='(default=%(default)s)')

    parser.add_argument('--experiment', default='ABSA', type=str, required=False,
                        choices=['mnist2', 'pmnist', 'cifar', 'mixture','ABSA'], help='(default=%(default)s)')
    parser.add_argument('--approach', default='ar1', type=str, required=False,
                        choices=['random', 'sgd', 'sgd-frozen', 'lwf', 'lfl', 'ewc', 'imm-mean', 'progressive',
                                 'pathnet',
                                 'imm-mode', 'sgd-restart',
                                 'joint', 'hat', 'hat-tests','si','ar1','lifelong'], help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--multi_output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=1, type=int, required=False, help='(default=%(default)d) try larger number for non-BERT models')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')

    opt = parser.parse_args()
##### Start Source Code Lifelong Learning ########################
    if opt.output == '':
        opt.output = 'res/' +  opt.model_name + '_' + opt.approach  + '_' +  str(opt.batch_size) + '_' + str(opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) +'.txt'
        opt.multi_output = 'res/multi_' + opt.model_name  + '_' + opt.approach + '_' +  str(opt.batch_size) + '_' + str(opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) + '.txt'
        opt.recall_output = 'res/recall_' + opt.model_name  + '_' + opt.approach  + '_' +  str(opt.batch_size) + '_' + str(opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) +'.txt'
        opt.f1_output = 'res/f1_' + opt.model_name  + '_' + opt.approach  + '_' +  str(opt.batch_size) + '_' + str(opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) +'.txt'

         #Algorithm path

        opt.output_algorithm = 'algorithms' + os.path.sep + 'algorithm_' +  opt.experiment + '_' + opt.approach + '_'  + opt.model_name + '.pt'
    print('=' * 100)
    print('Arguments =')
    for arg in vars(opt):
        print('\t' + arg + ':', getattr(opt, arg))
    print('=' * 100)
##### End Source Code Lifelong Learning ##########################
     ### TEST DATASET
    ###Tripadvisor
    tripAdvisorDataset = [{ 'sentence':"Shopping was a bit pricey compared to shops a half mile away but boy was the stuff nice!!",
                            'aspect':"value",
                            'polarity':'0'
                           },
                           {'sentence': "No comment on the rooms as we did not see them.",
                            'aspect': "notrelated",
                            'polarity': '1'
                           },
                           {'sentence': "Fine dining Restaurants like Benihanna, Golden Dragon (chinese), Sergios (Italian), as well as Regular eating places like pizzerias, delis, hot dog stands.",
                             'aspect': "food",
                             'polarity': '1'
                            }
                          ]

    restaurantDataset = [
        {'sentence': "I have to say they have one of the fastest $T$ in the city .",
         'aspect': "delivery times",
         'polarity': '1'
         },
        {'sentence': "Try the rose roll -LRB- not on $T$ -RRB- .",
         'aspect': "menu",
         'polarity': '0'
         },
        {'sentence': "In fact , this was not a $T$ and was barely eatable .",
         'aspect': "Nicoise salad",
         'polarity': '-1'
         },
        {'sentence': "Once we sailed , the top-notch $T$ and live entertainment sold us on a unforgettable evening .",
         'aspect': "food",
         'polarity': '1'
         },
        ]

    ###Computer
    houseApplianceDataset = [
        {'sentence': "its definately an outstanding $T$",
         'aspect': "laptop",
         'polarity': '1'
         },
        {'sentence': "Amazon delivery is very efficient and on time , as always ",
         'aspect': "<END>",
         'polarity': '-1'
         },
        {'sentence': "The contrast , crispness , and readability from even the most extreme $T$ angles ca n't be beat",
         'aspect': "viewing angles",
         'polarity': '1'
         }
        ]

    ###MovilePhone
    houseApplianceDataset2 = [
        {'sentence': "$T$ screen is good .",
         'aspect': "color screen",
         'polarity': '1'
         },
        {'sentence': "the phone comes with okay ringtones , some decent backgrounds / screensavers , but the phone has very little $T$ ( mine had 230kb as it arrived from amazon , so you do n't have too many options on what you can put on there ) .",
         'aspect': "memory",
         'polarity': '0'
         },
        {'sentence': "$T$ life is very good , i use it every day and i have to charge it every 5 or 6 days or so ." ,
         'aspect': "battery life",
         'polarity': '1'
         },
        {'sentence': "the day finally arrived when i was sure i 'd leave $T$ .",
         'aspect': "sprint",
         'polarity': '0'
         },
        ]

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'aen_bert_hat': AEN_BERT_HAT,
        'lcf_bert': LCF_BERT,
        'lcf_bert_hat': LCF_BERT_HAT,
        'lifeABSA': LifelongABSA
    }

    #Delete en all_multidomain  (Computer, Mp3 player, dvd player)
    #in x_all_multidomain there are all
    #ch_all_multidomain: Chage a restaurant dataset order at last
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'tests': './datasets/acl-14-short-data/tests.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'tests': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'tests': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'multidomain': {
            'train': {'twitter':'./datasets/tests/train_few.raw',
                      'laptop':'./datasets/tests/Laptops_Train_few.xml.seg',
                      'restaurant': './datasets/tests/Restaurants_Train.xml.seg',},
            'tests': {'twitter':'./datasets/tests/test_few.raw',
                     'laptop':'./datasets/tests/Laptops_Test_Gold_few.xml.seg',
                     'restaurant': './datasets/tests/Restaurants_Test_Gold.xml.seg'}
        },
        'original_algt_test': {
            'train': {'restaurant': './datasets/tests/Restaurants_Train.xml.seg', },
            'tests': {'restaurant': './datasets/tests/Restaurants_Test_Gold.xml.seg'}
        },
        'original_algt': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg', },
            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'}
        },
        'x_all_multidomain': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTrain.raw',
                      'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                      'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTrain.raw',
                      'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                      'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                      'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTrain.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTrain.raw',
                      },

            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                     'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                     'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTest.raw',
                     'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                     'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTest.raw',
                     'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                     'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                     'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTest.raw',
                     'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                     'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                     'hotels': './datasets/tripadvisor/tripadvisorfiles/process/tests.unique.json/tests.unique.jsonTest.raw',
                     }
        },
        'all_multidomain': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'canon g3':'./datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                      'nikon coolpix 4300':'./datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                      'nokia 6610':'./datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTrain.raw',
                      },

            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                      'canon g3':'./datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                      'nikon coolpix 4300':'./datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                      'nokia 6610':'./datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/tests.unique.json/tests.unique.jsonTest.raw',
                      }
        },
        'ch_all_multidomain': {
            'train': {
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTrain.raw',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                      'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                      'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                       'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      },

            'tests': {
                     'hotels': './datasets/tripadvisor/tripadvisorfiles/process/tests.unique.json/tests.unique.jsonTest.raw',
                     'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                     'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                     'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                     'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                     'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                     'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                      'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                     }
        },
        'device_multidomain': {
            'train': {
                      'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTrain.raw',
                      'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                      'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTrain.raw',
                      'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                      'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                      'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTrain.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                     },
            'tests': {
                     'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTest.raw',
                     'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                     'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTest.raw',
                     'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                     'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                     'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTest.raw',
                     'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                     'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                     }
        },
        'alldevice_multidomain': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'all_device': './datasets/binliu2004/process/globalTrain.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTest',
                      },
            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                     'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                     'all_device': './datasets/binliu2004/process/globalTest.raw',
                     'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTest',
                     }
        },
    }


    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'aen_bert_hat': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert':   ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert_hat': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        'lifeABSA': ['text_raw_without_aspect_indices', 'aspect_indices']

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }


    opt.model_class = model_classes[opt.model_name]
    print ("Objeto " + str(opt.model_class))
    #opt.model_class = opt.model_name
    opt.dataset_file = dataset_files[opt.dataset]

    #Define multidomain task size
    if opt.dataset == 'multidomain' or opt.dataset == 'all_multidomain' or opt.dataset == 'alldevice_multidomain' or opt.dataset == 'ch_all_multidomain' \
            or opt.dataset == 'original_algt' or opt.dataset == 'restaurant':
        opt.taskcla = len(dataset_files[opt.dataset]['train'])

    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt,model_classes)
    ins.run()
    textResult = ins.process_text(tripAdvisorDataset)
    print (textResult)

def init_predictor():
    tstart = time.time()

    # Hyper Parameters
    # default='bert_spc'
    # default='bert_spc'
    # --model_name lcf_bert --approach ar1
    # parser = argparse.ArgumentParser()

    class Dict2Class(object):

        def __init__(self, my_dict):
            for key in my_dict:
                setattr(self, key, my_dict[key])

    parser = { 'model_name':'bert_spc',
                'dataset':'original_algt',
                'optimizer':'adam',
               'initializer':'xavier_uniform_',
               'learning_rate':2e-5,
               'dropout':0.1,
               'l2reg':0.01,
               'num_epoch':10,
               'batch_size':2,
               'log_step':5,
               'embed_dim':300,
               'hidden_dim':300,
               'bert_dim':768,
               'pretrained_bert_name':'bert-base-uncased',
               'max_seq_len':80,
               'polarities_dim':3,
               'hops':3,
               'device':None,
               'seed':None,
               'valset_ratio':0,
               'local_context_focus':'cdm',
               'SRD':3,
               'measure':'recall',
               'experiment':'ABSA',
               'approach':'ar1',
               'output':'',
               'multi_output':'',
               'nepochs':1,
               'lr':0.05,
               'parameter':''
               }
    # parser.add_argument('--model_name', default='aen_bert', type=str)
    # parser.add_argument('--dataset', default='original_algt', type=str,
    #                     help='twitter, restaurant, laptop, multidomain, all_multidomain, alldevice_multidomain')
    # parser.add_argument('--optimizer', default='adam', type=str)
    # parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    # parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    # parser.add_argument('--dropout', default=0.1, type=float)
    # parser.add_argument('--l2reg', default=0.01, type=float)
    # parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    # parser.add_argument('--batch_size', default=2, type=int, help='try 16, 32, 64 for BERT models')
    # parser.add_argument('--log_step', default=5, type=int)
    # parser.add_argument('--embed_dim', default=300, type=int)
    # parser.add_argument('--hidden_dim', default=300, type=int)
    # parser.add_argument('--bert_dim', default=768, type=int)
    # parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    # parser.add_argument('--max_seq_len', default=80, type=int)
    # parser.add_argument('--polarities_dim', default=3, type=int)
    # parser.add_argument('--hops', default=3, type=int)
    # parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    # parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    # parser.add_argument('--valset_ratio', default=0, type=float,
    #                     help='set ratio between 0 and 1 for validation support')
    #
    # parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    # parser.add_argument('--SRD', default=3, type=int,
    #                     help='semantic-relative-distance, see the paper of LCF-BERT model')
    #
    # # Arguments LifeLearning Code
    #
    # # parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    # parser.add_argument('--measure', default='recall', type=str, required=False,
    #                     choices=['accuracy', 'recall', 'f1'], help='(default=%(default)s)')
    #
    # parser.add_argument('--experiment', default='ABSA', type=str, required=False,
    #                     choices=['mnist2', 'pmnist', 'cifar', 'mixture', 'ABSA'], help='(default=%(default)s)')
    # parser.add_argument('--approach', default='ar1', type=str, required=False,
    #                     choices=['random', 'sgd', 'sgd-frozen', 'lwf', 'lfl', 'ewc', 'imm-mean', 'progressive',
    #                              'pathnet',
    #                              'imm-mode', 'sgd-restart',
    #                              'joint', 'hat', 'hat-tests', 'si', 'ar1', 'lifelong'], help='(default=%(default)s)')
    # parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    # parser.add_argument('--multi_output', default='', type=str, required=False, help='(default=%(default)s)')
    # parser.add_argument('--nepochs', default=1, type=int, required=False,
    #                     help='(default=%(default)d) try larger number for non-BERT models')
    # parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    # parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
    #
    # parser.add_argument('run',  type=str, default='',   help='(default=%(default)s)')
    #
    #opt = parser.parse_args()
    opt = Dict2Class(parser)
    print ("Dict ****")
    print (opt)
    ##### Start Source Code Lifelong Learning ########################
    if opt.output == '':
        opt.output = 'res/' + opt.model_name + '_' + opt.approach + '_' + str(opt.batch_size) + '_' + str(
            opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) + '.txt'
        opt.multi_output = 'res/multi_' + opt.model_name + '_' + opt.approach + '_' + str(opt.batch_size) + '_' + str(
            opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) + '.txt'
        opt.recall_output = 'res/recall_' + opt.model_name + '_' + opt.approach + '_' + str(opt.batch_size) + '_' + str(
            opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) + '.txt'
        opt.f1_output = 'res/f1_' + opt.model_name + '_' + opt.approach + '_' + str(opt.batch_size) + '_' + str(
            opt.nepochs) + '_' + str(opt.dataset) + '_' + str(opt.measure) + '.txt'

        # Algorithm path

        opt.output_algorithm = 'algorithms' + os.path.sep + 'algorithm_' + opt.experiment + '_' + opt.approach + '_' + opt.model_name + '.pt'
    print('=' * 100)
    print('Arguments =')
    for arg in vars(opt):
        print('\t' + arg + ':', getattr(opt, arg))
    print('=' * 100)
    ##### End Source Code Lifelong Learning ##########################
    ### TEST DATASET
    ###Tripadvisor
    tripAdvisorDataset = [
        {'sentence': "Shopping was a bit pricey compared to shops a half mile away but boy was the stuff nice!!",
         'aspect': "value",
         'polarity': '0'
         },
        {'sentence': "No comment on the rooms as we did not see them.",
         'aspect': "notrelated",
         'polarity': '1'
         },
        {
            'sentence': "Fine dining Restaurants like Benihanna, Golden Dragon (chinese), Sergios (Italian), as well as Regular eating places like pizzerias, delis, hot dog stands.",
            'aspect': "food",
            'polarity': '1'
            }
        ]

    restaurantDataset = [
        {'sentence': "I have to say they have one of the fastest $T$ in the city .",
         'aspect': "delivery times",
         'polarity': '1'
         },
        {'sentence': "Try the rose roll -LRB- not on $T$ -RRB- .",
         'aspect': "menu",
         'polarity': '0'
         },
        {'sentence': "In fact , this was not a $T$ and was barely eatable .",
         'aspect': "Nicoise salad",
         'polarity': '-1'
         },
        {'sentence': "Once we sailed , the top-notch $T$ and live entertainment sold us on a unforgettable evening .",
         'aspect': "food",
         'polarity': '1'
         },
    ]

    ###Computer
    houseApplianceDataset = [
        {'sentence': "its definately an outstanding $T$",
         'aspect': "laptop",
         'polarity': '1'
         },
        {'sentence': "Amazon delivery is very efficient and on time , as always ",
         'aspect': "<END>",
         'polarity': '-1'
         },
        {'sentence': "The contrast , crispness , and readability from even the most extreme $T$ angles ca n't be beat",
         'aspect': "viewing angles",
         'polarity': '1'
         }
    ]

    ###MovilePhone
    houseApplianceDataset2 = [
        {'sentence': "$T$ screen is good .",
         'aspect': "color screen",
         'polarity': '1'
         },
        {
            'sentence': "the phone comes with okay ringtones , some decent backgrounds / screensavers , but the phone has very little $T$ ( mine had 230kb as it arrived from amazon , so you do n't have too many options on what you can put on there ) .",
            'aspect': "memory",
            'polarity': '0'
            },
        {'sentence': "$T$ life is very good , i use it every day and i have to charge it every 5 or 6 days or so .",
         'aspect': "battery life",
         'polarity': '1'
         },
        {'sentence': "the day finally arrived when i was sure i 'd leave $T$ .",
         'aspect': "sprint",
         'polarity': '0'
         },
    ]

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'aen_bert_hat': AEN_BERT_HAT,
        'lcf_bert': LCF_BERT,
        'lcf_bert_hat': LCF_BERT_HAT,
        'lifeABSA': LifelongABSA
    }

    # Delete en all_multidomain  (Computer, Mp3 player, dvd player)
    # in x_all_multidomain there are all
    # ch_all_multidomain: Chage a restaurant dataset order at last
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'tests': './datasets/acl-14-short-data/tests.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'tests': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'tests': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'multidomain': {
            'train': {'twitter': './datasets/tests/train_few.raw',
                      'laptop': './datasets/tests/Laptops_Train_few.xml.seg',
                      'restaurant': './datasets/tests/Restaurants_Train.xml.seg', },
            'tests': {'twitter': './datasets/tests/test_few.raw',
                     'laptop': './datasets/tests/Laptops_Test_Gold_few.xml.seg',
                     'restaurant': './datasets/tests/Restaurants_Test_Gold.xml.seg'}
        },
        'original_algt_test': {
            'train': {'restaurant': './datasets/tests/Restaurants_Train.xml.seg', },
            'tests': {'restaurant': './datasets/tests/Restaurants_Test_Gold.xml.seg'}
        },
        'original_algt': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg', },
            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'}
        },
        'x_all_multidomain': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTrain.raw',
                      'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                      'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTrain.raw',
                      'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                      'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                      'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTrain.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTrain.raw',
                      },

            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                     'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                     'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTest.raw',
                     'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                     'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTest.raw',
                     'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                     'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                     'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTest.raw',
                     'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                     'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                     'hotels': './datasets/tripadvisor/tripadvisorfiles/process/tests.unique.json/tests.unique.jsonTest.raw',
                     }
        },
        'all_multidomain': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                      'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                      'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                      'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                      'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTrain.raw',
                      },

            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                     'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                     'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                     'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                     'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                     'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                     'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                     'hotels': './datasets/tripadvisor/tripadvisorfiles/process/tests.unique.json/tests.unique.jsonTest.raw',
                     }
        },
        'ch_all_multidomain': {
            'train': {
                'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTrain.raw',
                'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
                'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
            },

            'tests': {
                'hotels': './datasets/tripadvisor/tripadvisorfiles/process/tests.unique.json/tests.unique.jsonTest.raw',
                'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
                'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
            }
        },
        'device_multidomain': {
            'train': {
                'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTrain.raw',
                'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Train.raw',
                'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTrain.raw',
                'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Train.raw',
                'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Train.raw',
                'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTrain.raw',
                'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTrain.raw',
                'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTrain.raw',
            },
            'tests': {
                'dvd player': './datasets/binliu2004/HuAndLiu2004/royal/process/apex ad2600 progressive-scan dvd player/apex ad2600 progressive-scan dvd playerTest.raw',
                'canon g3': './datasets/binliu2004/HuAndLiu2004/royal/process/canon g3/canon g3Test.raw',
                'creative labs nomad jukebox zen xtra 40gb': './datasets/binliu2004/HuAndLiu2004/royal/process/creative labs nomad jukebox zen xtra 40gb/creative labs nomad jukebox zen xtra 40gbTest.raw',
                'nikon coolpix 4300': './datasets/binliu2004/HuAndLiu2004/royal/process/nikon coolpix 4300/nikon coolpix 4300Test.raw',
                'nokia 6610': './datasets/binliu2004/HuAndLiu2004/royal/process/nokia 6610/nokia 6610Test.raw',
                'computer': './datasets/binliu2004/LiuEtAll2016/production/process/computer/computerTest.raw',
                'router': './datasets/binliu2004/LiuEtAll2016/production/process/router/routerTest.raw',
                'speaker': './datasets/binliu2004/LiuEtAll2016/production/process/speaker/speakerTest.raw',
            }
        },
        'alldevice_multidomain': {
            'train': {'restaurant': './datasets/semeval14/Restaurants_Train.xml.seg',
                      'laptop': './datasets/semeval14/Laptops_Train.xml.seg',
                      'all_device': './datasets/binliu2004/process/globalTrain.raw',
                      'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTest',
                      },
            'tests': {'restaurant': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
                     'laptop': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
                     'all_device': './datasets/binliu2004/process/globalTest.raw',
                     'hotels': './datasets/tripadvisor/tripadvisorfiles/process/train.unique.json/train.unique.jsonTest',
                     }
        },
    }

    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
                   'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'aen_bert_hat': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert_hat': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        'lifeABSA': ['text_raw_without_aspect_indices', 'aspect_indices']

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    print("Objeto " + str(opt.model_class))
    # opt.model_class = opt.model_name
    opt.dataset_file = dataset_files[opt.dataset]

    # Define multidomain task size
    if opt.dataset == 'multidomain' or opt.dataset == 'all_multidomain' or opt.dataset == 'alldevice_multidomain' or opt.dataset == 'ch_all_multidomain' \
            or opt.dataset == 'original_algt' or opt.dataset == 'restaurant':
        opt.taskcla = len(dataset_files[opt.dataset]['train'])

    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    #Inicialice Spacy Modules
    spacyModule = LinguisticRuleSpaCy(None)
    print ("<<------- Data to Print ----->>>")
    print(__file__)

    # Displaying the parent directory of the script
    print(os.path.dirname(__file__))
    fileAndAddress = os.path.dirname(__file__) + os.sep;
    bert = BertModel.from_pretrained(opt.pretrained_bert_name, output_attentions=True, cache_dir=fileAndAddress + "pretrained"+ os.path.sep + "bert" + os.path.sep)
    tokenizer = None
    module = None
    if 'bert' in opt.model_name:
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        module = opt.model_class(bert, opt).to(opt.device)
        #module = model_classes[opt.model_name]
    if opt.approach == 'random':
        from .approaches import random as approach
    elif opt.approach == 'sgd':
        from .approaches import sgd as approach
    elif opt.approach == 'sgd-restart':
        from approaches import sgd_restart as approach
    elif opt.approach == 'sgd-frozen':
        from .approaches import sgd_frozen as approach
    elif opt.approach == 'lwf':
        from .approaches import lwfNLP as approach
    elif opt.approach == 'lfl':
        from .approaches import lfl as approach
    elif opt.approach == 'ewc':
        from .approaches import ewcNLP as approach
    elif opt.approach == 'imm-mean':
        from .approaches import imm_mean as approach
    elif opt.approach == 'imm-mode':
        from .approaches import imm_mode as approach
    elif opt.approach == 'progressive':
        from .approaches import progressive as approach
    elif opt.approach == 'pathnet':
        from .approaches import pathnet as approach
    elif opt.approach == 'hat-tests':
        from .approaches import hat_test as approach

    elif opt.approach == 'ar1':
        from .approaches import ar1 as approach
    elif opt.approach == 'si':
        from approaches import si as approach
        # from approaches import hat as approach
    elif opt.approach == 'joint':
        from .approaches import joint as approach
    elif opt.approach == 'lifelong':
        from .approaches import lifelongBing as approach
    elif opt.approach == 'nostrategy':
        from .approaches import nostrategy as approach

        # Args -- Network
    if opt.experiment == 'mnist2' or opt.experiment == 'pmnist':
        if opt.approach == 'hat' or opt.approach == 'hat-tests':
            from .networks import mlp_hat as network
        else:
            from .networks import mlp as network
    else:
        if opt.approach == 'lfl':
            from .networks import alexnet_lfl as network
        elif opt.approach == 'hat':  # Select the BERT model for training datasets
            from .networks import bert as network
        elif opt.approach == 'progressive':
            from .networks import alexnet_progressive as network
        elif opt.approach == 'pathnet':
            from .networks import alexnet_pathnet as network
        elif opt.approach == 'lifelong' or opt.model_name.find(
                "bert") == -1:  # Only for BinLiu's method (Lifelong Learning Memory Networks for Aspect
            # Sentiment Classification)
            from .networks import NotBert  as network
        elif opt.approach == 'hat-tests' or opt.approach == 'ar1' or opt.approach == 'ewc' \
                or opt.approach == 'si' or opt.approach == 'lwf' or opt.approach == 'nostrategy':
            from .networks import bert as network

            # from networks import alexnet_hat_test as network
        else:
            from .networks import alexnet as network
        ##### End Source Code Lifelong Learning ########################

        # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #Check Letter BECAUSE I not sure that "correct module.parameters(module)"
    _params = filter(lambda p: p.requires_grad, module.parameters())
    optimizer = opt.optimizer(_params, lr=opt.learning_rate, weight_decay=opt.l2reg)
    inputsize = (3, 1, 0)
    net = network.Net(inputsize, None, opt).cuda() if torch.cuda.is_available() \
        else network.Net(inputsize, None, opt)
    net.set_Model(module)
    net.set_ModelOptimizer(optimizer)

    appr = approach.Appr(net, nepochs=opt.nepochs, lr=opt.lr, args=opt)

    if os.path.exists(opt.output_algorithm):
        appr.loadModel(opt.output_algorithm)
        print("Load Module values from: " + opt.output_algorithm)

    return (opt,model_classes,tokenizer, spacyModule, appr)

##### OJOOOOO WAY TO CALL THE MODELL #####

def getABSAModelClassify( sentence, aspect, tokenizer, appr, opt):
        """
         Apply Continual and Deep Learning model in senteces with aspect token
         with our trained model

        :param sentence:
        :param aspect:  Aspect text (It is a token in sentence)
        :return: A polarity value (positive, negative or neutral)

        """
        if tokenizer != None and  type(sentence) is str and sentence != "" and type(aspect) is str and aspect != "":
           testset = ABSADataset((sentence,aspect), tokenizer)

           test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

           return appr.eval_classify(0, test_data_loader)
           #test_loss, test_acc, test_recall, test_f1, test_kappa = appr.eval(0, test_data_loader)
        else:
          return None


##
##  Bibliografy:
##
##    Heuristyc of Entities extractor
##
##      https://spacy.io/usage/rule-based-matching#entityruler
##
##    View all Entity tags
##      import spacy
##
##      nlp = spacy.load("en_core_web_sm")
##      nlp.get_pipe("ner").labels
##

def entitiesSentencesExtrac(sentence):
    return None
if __name__ == '__main__':
    tripAdvisorDataset = [
        {'sentence': "Shopping was a bit pricey compared to shops a half mile away but boy was the stuff nice!!",
         'aspect': "value",
         'polarity': '0'
         },
        {'sentence': "No comment on the rooms as we did not see them.",
         'aspect': "notrelated",
         'polarity': '1'
         },
        {
            'sentence': "Fine dining Restaurants like Benihanna, Golden Dragon (chinese), Sergios (Italian), as well as Regular eating places like pizzerias, delis, hot dog stands.",
            'aspect': "food",
            'polarity': '1'
        }
    ]

    restaurantDataset = [
        {'sentence': "I have to say they have one of the fastest $T$ in the city .",
         'aspect': "delivery times",
         'polarity': '1'
         },
        {'sentence': "Try the rose roll -LRB- not on $T$ -RRB- .",
         'aspect': "menu",
         'polarity': '0'
         },
        {'sentence': "In fact , this was not a $T$ and was barely eatable .",
         'aspect': "Nicoise salad",
         'polarity': '-1'
         },
        {'sentence': "Once we sailed , the top-notch $T$ and live entertainment sold us on a unforgettable evening .",
         'aspect': "food",
         'polarity': '1'
         },
    ]
    opt, model_classes = init_predictor()
    ins = Instructor(opt, model_classes)
    ins.run()
    #textResult = ins.process_text(tripAdvisorDataset)
    #print(textResult)
    #main()



