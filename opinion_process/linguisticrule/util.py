import os,sys
import numpy as np
import tensorflow as tf
from copy import deepcopy
from copy import copy

# import torch
# from tqdm import tqdm

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

# In PyTorch, the learnable parameters (i.e. weights and biases) of an torch.nn.Module model are contained
# in the model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary
# object that maps each layer to its parameter tensor. Note that only layers with learnable
# parameters (convolutional layers, linear layers, etc.) have entries in the model’s state_dict. Optimizer objects (torch.optim) also have a state_dict, which contains information about the optimizer’s state, as well as the hyperparameters used.
#
# Because state_dict objects are Python dictionaries, they can be easily saved, updated, altered,
# and restored, adding a great deal of modularity to PyTorch models and optimizers.


# Deepcopy
#
# A deep copy creates a new object and recursively adds the
# copies of nested objects present in the original elements.
#
# Let’s continue with example 2. However, we are going to create
# deep copy using deepcopy() function present in copy module.
# The deep copy creates independent copy of original object
# and all its nested objects.

def get_model(model):
    # return deepcopy(model.state_dict())

    return copy(model)

def set_model_(model,state_dict):
    # model.load_state_dict(deepcopy(state_dict))
    model = copy(state_dict)
    return

def freeze_model(model):
    # for param in model.parameters():
    #     param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    # mean=0
    # std=0
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # for image, _ in loader:
    #     mean+=image.mean(3).mean(2)
    # mean /= len(dataset)
    #
    # mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    # for image, _ in loader:
    #     std+=(image-mean_expanded).pow(2).sum(3).sum(2)
    #
    # std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()
    #
    # return mean, std
    pass

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    # fisher={}
    # for n,p in model.named_parameters():
    #     fisher[n]=0*p.data
    # # Compute
    # model.train()
    # for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
    #     b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
    #     images=torch.autograd.Variable(x[b],volatile=False)
    #     target=torch.autograd.Variable(y[b],volatile=False)
    #     # Forward and backward
    #     model.zero_grad()
    #     outputs=model.forward(images)
    #     loss=criterion(t,outputs[t],target)
    #     loss.backward()
    #     # Get gradients
    #     for n,p in model.named_parameters():
    #         if p.grad is not None:
    #             fisher[n]+=sbatch*p.grad.data.pow(2)
    # # Mean
    # for n,_ in model.named_parameters():
    #     fisher[n]=fisher[n]/x.size(0)
    #     fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    # return fisher
    pass

########################################################################################################################
def cross_entropy(outputs,targets, exp=1,size_average=False,eps=1e-5):
    pivot = tf.convert_to_tensor(exp)
    output = tf.pow(tf.cast(outputs, dtype=pivot.dtype), pivot)
    y_output = tf.pow(tf.cast(targets, dtype=pivot.dtype), pivot)

    sum_output = tf.reduce_sum(output)
    sum_youtput = tf.reduce_sum(y_output)

    # print(sess.run(sum_output))
    # print(sess.run(sum_youtput))
    lossOld = tf.convert_to_tensor(np.zeros(1))

    for ivalue in range(output.shape[0]):
        zero_array = np.zeros(output.shape[0])
        zero_array[ivalue] = 1
        # print("To print ")
        # print(zero_array)
        aux_array = tf.convert_to_tensor(zero_array)
        aux_pow_element = tf.multiply(tf.cast(tf.transpose(output), dtype=tf.float64), aux_array)
        # print(sess.run(aux_pow_element))
        # print("------ Y Output ------")

        yaux_pow_element = tf.multiply(tf.cast(tf.transpose(y_output), dtype=tf.float64), aux_array)
        # print(sess.run(yaux_pow_element))
        # print("Para " + str(ivalue))
        y_o = tf.div(tf.reduce_sum(aux_pow_element), tf.cast(sum_output, dtype=tf.float64))

        y_oRoyal = tf.div(tf.reduce_sum(yaux_pow_element), tf.cast(sum_youtput, dtype=tf.float64))
        lossOld += tf.multiply(y_o, tf.log(y_oRoyal))

    ce= tf.multiply( tf.cast(-1, dtype=tf.float64),lossOld)
    if size_average:  #In Batch process
        ce=ce.mean()
    return ce

# def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
#     out=torch.nn.functional.softmax(outputs)
#     tar=torch.nn.functional.softmax(targets)
#     if exp!=1:
#         out=out.pow(exp)
#         out=out/out.sum(1).view(-1,1).expand_as(out)
#         tar=tar.pow(exp)
#         tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
#     out=out+eps/out.size(1)
#     out=out/out.sum(1).view(-1,1).expand_as(out)
#     ce=-(tar*out.log()).sum(1)
#     if size_average:
#         ce=ce.mean()
#     return ce
#     pass

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################
