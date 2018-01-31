# -*- coding:  UTF-8 -*-
'''
Created on 2018.01.30
@author: AI-World
'''
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.layers.core import Dense
from compiler.ast import flatten
import matplotlib.pyplot as plt

import math
#paramters
batch_size=64 #you can choose 16,or ...
iter_num=20
n_input=250  #embedding size
n_hidden=300 #vae embeddings
c_hidden=512 #classifer embedding
bata=0.5
keep_prob = tf.placeholder("float")
alpha = tf.placeholder("float")
it_learning_rate=tf.placeholder("float")
z_size=50

#data set
label_size=112

#tensor definition
input_x = tf.placeholder(dtype=tf.int32)
l_y=tf.placeholder(dtype=tf.float32,shape=[batch_size,label_size])
vae_y=tf.placeholder("float",[batch_size,None,label_size])  #vae_y
vae_y_u=tf.placeholder("float",[label_size,batch_size,None,label_size])

target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
un_target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
un_max_target_sequence_length = tf.reduce_max(un_target_sequence_length, name='max_target_len')
l_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

u_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
u_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
latentscale_iter=tf.placeholder(dtype=tf.float32)

#global list
table_X={} #trajectory
new_table_X={}
new_table_X = {}
voc_tra=list()
#define the weight and bias dictionary
with tf.name_scope("weight_inital"):
    weights_de={
        'w_':tf.Variable(tf.random_normal([z_size,n_hidden],mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([2*c_hidden, label_size]))
    }
    biases_de = {
    'b_': tf.Variable(tf.random_normal([n_hidden], mean=0.0, stddev=0.01)),
    'out': tf.Variable(tf.random_normal([label_size]))
    }

def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x
def extract_character_vocab(total_T):
    special_words = ['<PAD>', '<GO>', '<EOS>']
    set_words = list(set(flatten(total_T)))
    set_words = sorted(set_words)
    set_words = [str(item) for item in set_words]
    print len(set_words)
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def extract_words_vocab():
    print 'dictionary length',len(voc_tra)
    int_to_vocab={idx: word for idx, word in enumerate(voc_tra)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def getPvector(i):  # Embedding tensor
    return new_table_X[i]
def get_index(userT):
    userT = list(set(userT))
    User_List = sorted(userT)
    # print userT
    return User_List
def get_mask_index(value, User_List):
    #     print User_List #weikong
    return User_List.index(value)
def get_true_index(index, User_List):
    return User_List[index]
def getXs():  # =
    fpointvec = open('data/gowalla_user_vector250d_.dat', 'r')  # it has used word2vec
    #     table_X={}  #=
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()
        if (len(lineArr) < 250): #delete fist row
            continue
        item += 1  # 统计条目数
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  # 读取向量数据
        if lineArr[0] == '</s>':
            table_X['<PAD>']=X  #dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] =X
    print "point number item=", item
    return table_X
def readtraindata():
    test_T = list()
    test_UserT = list()
    test_lens = list()
    ftraindata = open('data/gowalla_scopus_1104.dat',
                      'r')
    tempT = list()
    pointT = list()
    userT = list()
    seqlens = list()
    item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        tempT.append(X)
        userT.append(int(X[0]))
        pointT.append(X[1:])
        seqlens.append(len(X) - 1)
        item += 1
    Train_Size =10000
    pointT = pointT[:Train_Size]  # all tra
    userT = userT[:Train_Size]  # all user
    seqlens = seqlens[:Train_Size]  # all length
    User_List = get_index(userT)
    flag = 0
    count = 0;
    temp_pointT = list()
    temp_userY = list()
    temp_seqlens = list()
    User = 0  #
    rate = 0.5 #split rate
    for index in range(len(pointT)):
        if (userT[index] != flag or index == (len(pointT) - 1)):
            User += 1
            #split data
            if (count > 1):  #
                test_T += (pointT[int((index - math.ceil(count * rate))):index])
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])
                temp_pointT += (pointT[int((index - count)):int((index - count * rate))])
                temp_userY += (userT[int((index - count)):int((index - count * rate))])
                temp_seqlens += (seqlens[int((index - count)):int((index - count * rate))])
            else:
                temp_pointT += (pointT[int((index - count)):int((index))])
                temp_userY += (userT[int((index - count)):int((index))])
                temp_seqlens += (seqlens[int((index - count)):int((index))])
            count = 1;
            flag = userT[index]
        else:
            count += 1
    pointT = temp_pointT
    userT = temp_userY
    total_T = pointT + test_T
    print 'Total Numbers=', item - 1
    print 'train trajectories number=', len(total_T)
    print 'Train Size=', len(pointT), ' Test Size=', len(test_T), "User numbers=", len(User_List)
    return pointT, userT,test_T, test_UserT,User_List#
#input
getXs()
pointT, userT,test_T, test_UserT,User_List=readtraindata()
total_Ts=pointT+test_T
for i_ in range(len(total_Ts)):
    for j_ in range(len(total_Ts[i_])):
        new_table_X[total_Ts[i_][j_]]=table_X[total_Ts[i_][j_]]
#
new_table_X['<GO>']=table_X['<GO>']
new_table_X['<EOS>']=table_X['<EOS>']
new_table_X['<PAD>']=table_X['<PAD>']
for keys in new_table_X:
    voc_tra.append(keys)
print 'train trajectory size',len(pointT)
print 'test trajectory size',len(test_T)

int_to_vocab, vocab_to_int=extract_words_vocab()
print 'POIs number is ',len(vocab_to_int)
TOTAL_SIZE = len(vocab_to_int)

#convert to int type
#Train Dataset
new_pointT = list()
for i in range(len(pointT)):
    temp = list()
    for j in range(len(pointT[i])):

        temp.append(vocab_to_int[pointT[i][j]])
    new_pointT.append(temp)

#Test Dataset
new_testT = list()
for i in range(len(test_T)):
    temp = list()
    for j in range(len(test_T[i])):
        temp.append(vocab_to_int[test_T[i][j]])
    new_testT.append(temp)
#Get dictionary
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
dic_embeddings=tf.constant(dic_em())
print 'Dictionary Size',len(dic_em())
#------------------------------------------------------------------------------
#classifer
def classifer(encoder_embed_input,keep_prob=0.5,reuse=False):
    with tf.variable_scope("classifier",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # , state_is_tuple=True
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # , state_is_tuple=True
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
        #
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_, dtype=tf.float32, time_major=True,
                                                            )
        new_outputs = tf.concat(outputs, 2)
        pred = (tf.matmul(new_outputs[-1], weights_de["out"]) + biases_de["out"])
        return pred

# ENCODER PART of VAE
def encoder(encoder_embed_input,y,keep_prob=0.5,reuse=False):
    with tf.variable_scope("encoder",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        encode_lstm = tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0, state_is_tuple=True)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_lstm, output_keep_prob=keep_prob)
        (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)
        new_states=tf.concat([states[-1],y],1)
        o_mean = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                   scope="z_mean")
        o_stddev = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                     scope="z_std")
        return outputs, states, o_mean, o_stddev, new_states
#DECODER PART of VAE
def decoder(decoder_embed_input,decoder_y,target_length,max_target_length,encode_state,keep_prob,reuse=False):
    with tf.variable_scope("decoder",reuse=reuse):
        decode_lstm = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        decode_cell = tf.contrib.rnn.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        decoder_initial_state = encode_state
        output_layer = Dense(TOTAL_SIZE) #TOTAL_SIZE
        decoder_input_ = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), decoder_embed_input],
                                   1)  # add GO to the end
        decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_input_)
        decoder_input=tf.concat([decoder_input,decoder_y],2)
        # # input_=tf.transpose(decoder_input,[1,0,2])
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                            sequence_length=target_length)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, training_helper, decoder_initial_state,
                                                           output_layer)
        output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                         impute_finished=True,
                                                         maximum_iterations=max_target_length)
        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_length, max_target_length, dtype=tf.float32, name='masks')
        #target = tf.concat([target_input, tf.fill([batch_size, 1], vocab_to_int['<EOS>'])], 1)  #
        target = decoder_embed_input
        return output,predicting_logits,training_logits,masks,target

def get_cost_c(pred): #compute classifier cost
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=l_y))
    return cost
def get_cost_l(encoder_embed_input,decoder_embed_input,l_y,decoder_y,target_sequence_length,max_target_sequence_length,reuse=False):
    encode_outputs, encode_states, z_mean, z_stddev, new_states = encoder(encoder_embed_input,l_y, keep_prob,reuse)
    samples = tf.random_normal(tf.shape(z_stddev))
    z = z_mean + tf.exp(z_stddev * 0.5) * samples
    h_state = tf.nn.softplus(tf.matmul(z, weights_de['w_']) + biases_de['b_'])
    #c_state = tf.nn.softplus(tf.matmul(z, weights_de['w_2']) + biases_de['b_2'])
    decoder_initial_state = LSTMStateTuple(h_state, encode_states[1])
    decoder_output, predicting_logits, training_logits, masks, target = decoder(decoder_embed_input,decoder_y,target_sequence_length,max_target_sequence_length,decoder_initial_state,keep_prob,reuse)
    #KL term-------------
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)
    latent_cost = tf.reduce_mean(latent_loss)

    encropy_loss = tf.contrib.seq2seq.sequence_loss(training_logits, target, masks)#/batch_size

    cost = tf.reduce_mean(encropy_loss + latentscale_iter * (latent_loss))

    return cost,encropy_loss,latent_cost,training_logits
def get_cost_u(u_encoder_embed_input,u_decoder_embed_input):
    prob_y=classifer(u_encoder_embed_input,keep_prob=keep_prob,reuse=True)
    prob_y=tf.nn.softmax(prob_y) #
    for label in range(label_size):
        y_i=get_onehot(label)
        cost_l, en_cost, kl_cost,training_logits=get_cost_l(u_encoder_embed_input,u_decoder_embed_input,[y_i]*batch_size,vae_y_u[label],un_target_sequence_length,un_max_target_sequence_length,reuse=True)
        u_cost = tf.expand_dims([cost_l], 1)  #
        if label == 0: L_ulab = tf.identity( u_cost )
        else: L_ulab = tf.concat([L_ulab, u_cost],1)

    U = (1./label_size)*tf.reduce_sum(tf.multiply(L_ulab, prob_y) - tf.multiply(prob_y, tf.log(prob_y)))  #

    return U#,L_ulab

def creat_y_scopus(label_y,seq_length): # copy
    lcon_y= [label_y for j in range(seq_length)]
    return lcon_y
def creat_u_y_scopus(seq_length): #
    ucon_y=[]
    for i in range(label_size):
        label_y=get_onehot(i)
        temp=[]
        for j in range(batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        ucon_y.append(temp)
    return ucon_y

pred=classifer(l_encoder_embed_input)
cost_c=get_cost_c(pred)

cost_l,encropy_loss,latent_cost,training_logits=get_cost_l(l_encoder_embed_input,l_decoder_embed_input,l_y,vae_y,target_sequence_length,max_target_sequence_length)

cost_u=get_cost_u(u_encoder_embed_input,u_decoder_embed_input) #unlabel data

cost=cost_c+cost_l+bata*cost_u #alpha*

#optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)

#evaluate model
correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(l_y,1)) #
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch] #
def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
initial = tf.global_variables_initializer()
def   train_model():
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(initial)
        #saver.restore(sess, './model/gw_tulvae_112.pkt')
        print'Read train & test data'
        initial_learning_rate = 0.00095
        learning_rate_len = 0.000008
        min_kl=0.0
        min_kl_epoch=min_kl
        kl_lens = 0.0008
        #sort
        index_T={}
        new_trainT=[]
        new_trainU=[]
        for i in range(len(new_pointT)):
            index_T[i]=len(new_pointT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id=temp_size[i][0]
            new_trainT.append(new_pointT[id])
            new_trainU.append(userT[id])

        #sort for test dataset
        #sort
        index_T = {}
        testT = []
        testU = []
        for i in range(len(new_testT)):
            index_T[i] = len(new_testT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id = temp_size[i][0]
            testT.append(new_testT[id])
            testU.append(test_UserT[id])
        #-----------------------------------
        TRAIN_ACC=[]
        COST=[]
        tempU = list(set(User_List))
        TRAIN_DIC = {}
        for i in range(len(tempU)):
            TRAIN_DIC[i] = [0, 0, 0]  # use mask
        TRAIN_P = []
        TRAIN_R = []
        TRAIN_F1 = []
        TRAIN_ACC1 = []
        TRAIN_ACC5 = []

        TEST_P = []
        TEST_R = []
        TEST_F1 = []
        TEST_ACC1 = []
        TEST_ACC5 = []
        Learning_rate = []
        count = 0
        alpha_epoch=1
        alpha_value=(2.0-1.0)/iter_num
        for epoch in range(iter_num):

            #initial_learning_rate -= learning_rate_len
            if(initial_learning_rate<=0):
                initial_learning_rate=0.000001
            step=0
            acc=0
            acc5 = 0
            train_cost=0
            label_cost=0
            unlabel_cost=0
            classifier_cost=0
            while step < len(new_trainT) // batch_size:
                start_i = step * batch_size
                input_x = new_trainT[start_i:start_i + batch_size]
                input_ux=testT[start_i:start_i + batch_size]
                # 补全序列
                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
                encode_batch=eos_sentence_batch(input_x,vocab_to_int['<EOS>'])
                input_batch=pad_sentence_batch(encode_batch,vocab_to_int['<PAD>'])

                # 无标签补全序列
                un_sources_batch = pad_sentence_batch(input_ux, vocab_to_int['<PAD>'])
                un_encode_batch=eos_sentence_batch(input_ux,vocab_to_int['<EOS>'])
                un_input_batch=pad_sentence_batch(un_encode_batch,vocab_to_int['<PAD>'])

                # 记录长度 unlabel
                un_pad_source_lengths = []
                for source in input_ux:
                    un_pad_source_lengths.append(len(source)+1)
                # 记录长度
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source)+1)
                #print len(input_batch[0])
                target_maxlength=len(input_batch[0])+1 #get max length

                un_target_maxlength = len(un_input_batch[0]) + 1  # get max length
                if min_kl_epoch<1.0:
                    min_kl_epoch = min_kl + count* kl_lens
                else:
                    min_kl_epoch=1.0
                batch_y = []
                decode_y=[]
                user_mask_id = []
                for y_i in range(start_i, start_i + batch_size):
                    xsy_step = get_onehot(get_mask_index(new_trainU[y_i], User_List))
                    #print xsy_step
                    user_mask_id.append(get_mask_index(new_trainU[y_i], User_List))
                    TRAIN_DIC.get(get_mask_index(new_trainU[y_i], User_List))[2]+=1 #Groud value Groud Truth a+c
                    decode_y.append(creat_y_scopus(xsy_step,target_maxlength)) #copy
                    batch_y.append(xsy_step)
                decode_uy=creat_u_y_scopus(un_target_maxlength)
                #print batch_y
                #print decode_uy[0]
                #print decode_y

                pred_batch,c_pred,op,batch_cost,l_cost,u_cost,c_cost=sess.run([pred,correct_pred,optimizer,cost,cost_l,cost_u,cost_c],feed_dict={vae_y:decode_y,vae_y_u:decode_uy,
                    l_encoder_embed_input:sources_batch, l_y: batch_y,u_encoder_embed_input: un_sources_batch,u_decoder_embed_input: un_input_batch,
                                                                           it_learning_rate: initial_learning_rate,latentscale_iter:min_kl_epoch,
                                                                           keep_prob: 0.5,l_decoder_embed_input: input_batch,
                                                                           target_sequence_length: pad_source_lengths,un_target_sequence_length:un_pad_source_lengths,alpha:alpha_epoch})
                #computing

                for i in range(len(pred_batch)):
                    value=pred_batch[i]

                    top1=np.argpartition(a=-value,kth=1)[:1]
                    TRAIN_DIC.get(top1[0])[1] += 1  # recommend value a+b
                    top5=np.argpartition(a=-value,kth=5)[:5]
                    if user_mask_id[i] in top5:
                        acc5+=1
                    if c_pred[i]==True:
                        acc+=1
                        TRAIN_DIC.get(user_mask_id[i])[0]+=1 #REAL value a
                #print logit.shape
                if(step%10==0 and step is not 0):
                    print 'min_kl_epoch',min_kl_epoch
                    print 'TRAIN LOSS', train_cost, 'LABEL COST', label_cost, 'Unlabel Cost', unlabel_cost, 'Classifier Cost', classifier_cost
                loss=np.mean(batch_cost)
                lbatch_cost=np.mean(l_cost)
                ubatch_cost=np.mean(u_cost)
                cbatch_cost=np.mean(c_cost*alpha_epoch)
                classifier_cost+=cbatch_cost
                unlabel_cost+=ubatch_cost
                label_cost+=lbatch_cost
                train_cost+=loss
                step+=1 # while
                count+=1
            alpha_epoch+=alpha_value

            # Precision Recall, F1
            P = []
            R = []
            for i in TRAIN_DIC.keys():
                # print TRAIN_DIC.get(i)[0],TRAIN_DIC.get(i)[1]
                if TRAIN_DIC.get(i)[1] == 0:
                    TRAIN_DIC.get(i)[1] = 1
                if TRAIN_DIC.get(i)[2] == 0:
                    TRAIN_DIC.get(i)[2] = 1
                Pi = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[1]
                Ri = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[2]
                P.append(Pi)
                R.append(Ri)
            macro_R = np.mean(R)
            macro_P = np.mean(P)
            macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
            TRAIN_P.append(macro_P)
            TRAIN_R.append(macro_R)
            TRAIN_F1.append(macro_F1)
            TRAIN_ACC1.append(acc / (step * batch_size))
            TRAIN_ACC5.append(acc5 / (step * batch_size))
            print '\nTRAIN RESULT'
            print 'macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1
            print 'total train number', step * batch_size, 'learning rate', initial_learning_rate
            print 'iter', epoch, 'Accuracy', acc / (step * batch_size), 'Accuracy5', acc5 / (
            step * batch_size), 'TRAIN LOSS', train_cost
            print '\nepoch TEST'
            TEST_p, TEST_r, TEST_f1, TEST_acc1, TEST_acc5 = test_model(sess, testT, testU, epoch)
            TEST_P.append(TEST_p)
            TEST_R.append(TEST_r)
            TEST_F1.append(TEST_f1)
            TEST_ACC1.append(TEST_acc1)
            TEST_ACC5.append(TEST_acc5)
            Learning_rate.append(initial_learning_rate)
            saver.save(sess, './model/gw_tulvae_112.pkt')
        save_metrics(Learning_rate, TEST_P, TEST_R, TEST_F1, TEST_ACC1, TEST_ACC5, root='./out/gw_tulvae_test_112.txt')
        save_metrics(Learning_rate, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, TRAIN_ACC5,
                     root='./out/gw_tulvae_train_112.txt')
        draw_pic_metric(TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, TRAIN_ACC5, name='train')
        draw_pic_metric(TEST_P, TEST_R, TEST_F1, TEST_ACC1, TEST_ACC5, name='test')

                #metric_compute(correct_pred)
def test_model(sess,testT,testU,epoch):
    step = 0
    count = 0
    acc = 0
    acc5 = 0
    tempU = list(set(User_List))
    TEST_DIC = {}
    for i in range(len(tempU)):
        TEST_DIC[i] = [0, 0, 0]  # use mask
    while step < len(testT) // batch_size:  #
        start_i = step * batch_size
        input_x = testT[start_i:start_i + batch_size]
        # 补全序列
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
        # 记录长度
        pad_source_lengths = []
        user_mask_id = []
        for source in input_x:
            pad_source_lengths.append(len(source) + 1)
        batch_y = []
        for y_i in range(start_i, start_i + batch_size):
            xsy_step = get_onehot(get_mask_index(testU[y_i], User_List))
            user_mask_id.append(get_mask_index(testU[y_i], User_List))
            TEST_DIC.get(get_mask_index(testU[y_i], User_List))[2] += 1  # Groud value Groud Truth a+c
            batch_y.append(xsy_step)
        c_pred,pred_batch=sess.run([correct_pred,pred],feed_dict={l_encoder_embed_input:sources_batch, l_y: batch_y,

                                                                           keep_prob: 1.0,l_decoder_embed_input: input_batch,
                                                                           target_sequence_length: pad_source_lengths})
        for i in range(len(pred_batch)):
            value = pred_batch[i]
            top1 = np.argpartition(a=-value, kth=1)[:1]
            TEST_DIC.get(top1[0])[1] += 1  # recommend value a+b
            top5 = np.argpartition(a=-value, kth=5)[:5]
            if user_mask_id[i] in top5:
                acc5 += 1
            if c_pred[i] == True:
                acc += 1
                TEST_DIC.get(user_mask_id[i])[0] += 1  # REAL value a
        step+=1 # while
    # Precision Recall, F1
    P = []
    R = []
    for i in TEST_DIC.keys():
        if TEST_DIC.get(i)[1] == 0:
            TEST_DIC.get(i)[1] = 1
        if TEST_DIC.get(i)[2] == 0:
            TEST_DIC.get(i)[2] = 1
        Pi = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[1]
        Ri = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[2]
        P.append(Pi)
        R.append(Ri)
    macro_R = np.mean(R)
    macro_P = np.mean(P)
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
    print 'macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1
    print 'iter', epoch, 'Accuracy For TEST', acc / (step * batch_size), 'Accuracy5 For TEST', acc5 / (
    step * batch_size), 'total test number', step * batch_size
    return macro_P, macro_R, macro_F1, acc / (step * batch_size), acc5 / (step * batch_size)
def save_metrics(LEARN_RATE,TRAIN_P,TRAIN_R,TRAIN_F1,TRAIN_ACC1,TRAIN_ACC5,root='out/gw_metric_tulvae.txt'):
    files=open(root,'w')
    files.write('epoch \t learning_rate \t Precision \t Recall \t F1 \t ACC1 \t ACC5\n')
    for i in range(len(TRAIN_P)):
        files.write(str(i)+'\t')
        files.write(str(LEARN_RATE[i])+'\t')
        files.write(str(TRAIN_P[i]) + '\t'+str(TRAIN_R[i])+'\t'+str(TRAIN_F1[i])+'\t'+str(TRAIN_ACC1[i])+str(TRAIN_ACC5[i])+'\n')
    files.close()
def draw_pic_metric(TEST_P,Test_R,Test_F1,Test_ACC1,TEST_ACC5,name='train'):
    font = {'family': name,
            'weight': 'bold',
            'size': 18
            }
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    train_axis = np.array(range(1, len(TEST_P) + 1, 1))
    plt.plot(train_axis, np.array(TEST_P), "b--", label="Test P")
    train_axis = np.array(range(1, len(Test_R) + 1, 1))
    plt.plot(train_axis, np.array(Test_R), "r--", label="Test R")
    train_axis = np.array(range(1, len(Test_F1) + 1, 1))
    plt.plot(train_axis, np.array(Test_F1), "g--", label="Test F1-score")
    train_axis = np.array(range(1, len(Test_ACC1) + 1, 1))
    plt.plot(train_axis, np.array(Test_ACC1), "y--", label="Test ACC1")
    train_axis = np.array(range(1, len(TEST_ACC5) + 1, 1))
    plt.plot(train_axis, np.array(TEST_ACC5), "c--", label="Test ACC5")
    plt.title(name)
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('value')
    plt.xlabel('Training iteration')
    plt.show()

if __name__ == "__main__":
    train_model()
    print 'Model END'