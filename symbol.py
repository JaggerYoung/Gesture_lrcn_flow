import find_mxnet
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import sys
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias", "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol", "init_states", "last_states", "seq_data", "seq_labels", "seq_outputs", "param_bloacks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    i2h = mx.symbol.FullyConnected(data=indata, 
                                   weight=param.i2h_weight,
				   bias=param.i2h_bias,
				   num_hidden=num_hidden * 4,
				   name="t%d_l%d_i2h"%(seqidx,layeridx))
    h2h = mx.symbol.FullyConnected(data=prev_state.h,
                                   weight=param.h2h_weight,
				   bias=param.h2h_bias,
				   num_hidden=num_hidden * 4,
				   name="t%d_l%d_h2h"%(seqidx, layeridx))

    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, 
                                      name="t%d_l%d_slice"%(seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)

def get_cnn(input_data):
    # stage 1
    conv1 = mx.symbol.Convolution(data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    return fc1

def c3d_bilstm(num_lstm_layer, seq_len, num_hidden, num_label):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
	                             i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
				     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    input_data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')

    wordvec = mx.symbol.SliceChannel(data=input_data, num_outputs=seq_len, axis=2)   
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = get_cnn(wordvec[seqidx])
	for i in range(num_lstm_layer):
	    next_state = lstm(num_hidden, indata=hidden, 
	                      prev_state=last_states[i],
			      param=param_cells[i],
			      seqidx=seqidx, layeridx=i)
	    hidden = next_state.h
	    last_states[i] = next_state
	hidden_all.append(hidden)

    hidden_concat = mx.symbol.Concat(*hidden_all, dim=0)
    pred = mx.symbol.FullyConnected(data=hidden_concat, num_hidden=num_label, name='fc')
    
    label_slice = mx.symbol.SliceChannel(data=label, num_outputs=seq_len, axis=1, squeeze_axis=True)
    label_all = [label_slice[t] for t in range(seq_len)]
    label_1 = mx.symbol.Concat(*label_all, dim=0)
    label = mx.symbol.Reshape(data=label_1, target_shape=(0,))

    sm = mx.symbol.SoftmaxOutput(data=pred, label=label, name='softmax')
    return sm

