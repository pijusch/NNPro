from __future__ import print_function
import collections
import os
import pandas as pd
from keras.initializers import RandomNormal, RandomUniform
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import tensorflow as tf
from keras.models import Sequential, Model
from keras.utils import multi_gpu_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, Multiply, dot, subtract, add, multiply
from keras.optimizers import Adam, Adagrad
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling1D
import numpy as np
import argparse
import gensim
import json
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pickle
from keras.layers import Input
from keras import backend as K
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from keras.backend import clear_session
from keras.constraints import MinMaxNorm
class NN:

  def nn_function(self,filename,epochs,type_,embedding_type,first,weights,margin):
    #filename = input()
    #epochs = int(input())
    #typ_ = int(input())

    class Metrics(Callback):
      def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
      
      def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(parallel_model.predict([val1, val2]))).round()
        val_targ = valy
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (' — val_f1: %f — val_precision: %f — val_recall %f' %( _val_f1, _val_precision, _val_recall))
        return


    metrics = Metrics()

    with open(filename,'rb') as f:
        pic = pickle.load(f)
    pic[0] = np.array(pic[0][:])
    pic[1] = np.array(pic[1][:])

    #pic=  np.array(pic)

    mini = min(100,pic[0].shape[0])

    num = len(pic[0])


    valy = np.array([0]*100+[1]*100)

    x1 = pic[0][int(num*0.0):,:]
    x2 = pic[1][int(num*0.0):,:]
    ones = np.array([1]*len(x1))
    y = [margin]*len(x1)

    val1 = pic[0][:int(num*0.01),:]
    val2 = pic[1][:int(num*0.01),:]
    valones = np.array([1]*len(val1))
    valy = [1]*len(val1)

    embed = x1.shape[1]
    #first = 15
    #new = int(x1.shape[1]/2)
    dim = embed

    clear_session()

    from keras.layers import Input
    from keras import backend as K
    one = Input(shape = (1,))
    w1 = Input(shape = (1,))
    w2 = Input(shape = (1,))
    w3 = Input(shape = (1,))
    #constants = [1] * 16
    #k_constants = K.variable(constants, name = "ones_variable")
    #ones_tensor = Input(tensor=k_constants, name = "ones_tensor")

    # Fixed
    model1 = Sequential()
    #model1.add(Dense(new, activation='sigmoid', name='h13',input_dim=embed,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01 , seed=0)))
    model1.add(Dense(first, activation='sigmoid', name='h11',input_dim=embed))#, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)))#,kernel_initializer=RandomNormal(mean=0.0, stddev=0.001 , seed=0)))
    #model1.add(Dense(new, activation='sigmoid', name='h14',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=0)))
    model1.add(Dense(dim, activation='sigmoid', name='h12'))#, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)))#,kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=1)))
    #model1.add(Dense(first, activation='sigmoid', name='h11',input_dim=embed))
    #model1.add(Dense(dim, activation='sigmoid', name='h12'))

    model2 = Sequential()
    #model2.add(Dense(new, activation='sigmoid', name='h23',input_dim=embed,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=0)))
    model2.add(Dense(first, activation='sigmoid', name='h21',input_dim = embed))#, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)))#,kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=0)))
    #model2.add(Dense(new, activation='sigmoid', name='h24',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=0)))
    model2.add(Dense(dim, activation='sigmoid', name='h22'))#, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)))#,kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=1)))
    #model2.add(Dense(first, activation='sigmoid', name='h21',input_dim=embed))
    #model2.add(Dense(dim, activation='sigmoid', name='h22'))


    # Loss Generation
    
    # A-B, A'-B'
    Sub1 = subtract([model1.output,model2.output])
    Sub2 = subtract([model1.input,model2.input])
    
    # (|A-B| - |A'-B'|)^2
    Dot4 = dot([Sub1,Sub1], axes=-1, normalize=True)
    Dot5 = dot([Sub2,Sub2],axes = -1,normalize=True)
    Sub3 = subtract([Dot4,Dot5])
    #Mul6 = multiply([w3,multiply([Sub3,Sub3])])
    Mul6 = multiply([Sub3,Sub3])

    # (A-A').(B-B')
    dDot = dot([Sub1,Sub2], axes=-1, normalize=True)

    # A.A'
    dDot2 = dot([model1.input,model1.output], axes=-1, normalize=True)
    
    # B.B'
    dDot3 = dot([model2.input,model2.output], axes=-1, normalize=True)
    
    # A.A' + B.B' + (A-A').(B-B')
    aAdd=  add([dDot,dDot2])
    aAdd2 = add([aAdd,dDot3])

    # (1-(A-A').(B-B'))
    Dot = subtract([one,dot([Sub1,Sub2], axes=-1, normalize=True)])
    # (1 - A.A')
    Dot2 = subtract([one,dot([model1.input,model1.output], axes=-1, normalize=True)])
    # (1- B.B')
    Dot3 = subtract([one,dot([model2.input,model2.output], axes=-1, normalize=True)])
    #Mul1= multiply([w2,multiply([Dot,Dot])])
    #Mul2 = multiply([w1,multiply([Dot2,Dot2])])
    #Mul3 = multiply([w1,multiply([Dot3,Dot3])])
    Mul1= multiply([Dot,Dot])
    Mul2 = multiply([Dot2,Dot2])
    Mul3 = multiply([Dot3,Dot3])
    Add=  add([Mul1,Mul2])
    Add2 = add([Add,Mul3])
    Add3 = add([Add2,Mul6])
    
    
    # (A-A')
    Sub3 = subtract([model1.input,model1.output])
    # (B-B')
    Sub4 = subtract([model2.input,model2.output])
    
    Mul4 = dot([Sub3,Sub3], axes=-1, normalize=True)
    Mul5 = dot([Sub4,Sub4], axes=-1, normalize=True)
    #Add=  add([Mul1,Mul4])
    #Add2 = add([Add,Mul5])
    
    model = Model(inputs=[model1.input,model2.input,one,w1,w2,w3], outputs= Add3)
    #model.add(Dense(1, activation = 'sigmoid'))
    # model = Multiply()([model1.get_layer('out1').output,model2.get_layer('out2').output])
    
    # model.add(TimeDistributed(Dense(vocabulary)))
    # model.add(Activation('softmax'))
    
    #optimizer = Adam()
    # model1.compile(loss='mean_squared_error', optimizer='adam')
    # parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model = model
    #parallel_model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001) , metrics=['acc'])
    parallel_model.compile(loss='mean_squared_error', optimizer='adam' )
    
    print(model.summary())
    print(model1.summary())
    print(model2.summary())
    # checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    num_epochs = epochs
    w1s = np.array([weights[0]]*len(ones))
    w2s = np.array([weights[1]]*len(ones))
    w3s = np.array([weights[2]]*len(ones))
  
    # previous stable 256
    parallel_model.compile(loss='mean_squared_error', optimizer='adam' )
    history = parallel_model.fit(x=[x1, x2,ones,w1s,w2s,w3s], y=y, batch_size=64, epochs=int(num_epochs),
                    )


    embd1 = Model(inputs=parallel_model.input,
                                    outputs=parallel_model.get_layer('h1'+embedding_type).output)

    embd2 = Model(inputs=parallel_model.input,outputs=parallel_model.get_layer('h2'+embedding_type).output)

    emb1v = embd1.predict([x1,x2,ones,w1s,w2s,w3s])
    emb2v = embd2.predict([x1,x2,ones,w1s,w2s,w3s])

    history = pd.DataFrame.from_dict({'iter':range(len(history.history['loss'])),'loss':history.history['loss']})
    history.to_csv('./static/loss.csv',index=False)

    x = [emb1v,emb2v]
    tname = np.concatenate([pic[3],pic[4]])
    col = [0]*len(pic[2])+[1]*len(pic[2])
    #with open('reduced_set.pkl','wb') as f:
      #pickle.dump(x,f)



    #print(parallel_model.predict([val_sent_e[:100], val_claim_e[:100]]), val_y[:100])
    #print(parallel_model.evaluate([val_sent_e,val_claim_e],val_y))
    #parallel_model.save("final_model.hdf5")
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #with open('reduced_set.pkl','rb') as f:
    #    x = pickle.load(f)


    #fig,ax = plt.subplots()

    #for i in range(len(a1)):
    #    plt.plot([a1[i],b1[i]],[a2[i],b2[i]],linewidth=0.4)
    '''
    ax.scatter(x[1][:,0],x[1][:,1],c='blue',label='Objects')
    ax.scatter(x[0][:,0],x[0][:,1],c='red',label='Subjects')
    if typ_==1:
      for i in range(len(x[0])):
        ax.annotate(pic[2][i], (x[0][i,0], x[0][i,1]))

      for i in range(len(x[0])):
        ax.annotate(pic[3][i], (x[1][i,0], x[1][i,1]))

    plt.legend()

    plt.show()
    exit(0)
    '''
    x = np.concatenate((x[0],x[1]),axis=0)

    pca = PCA()
    lll = pca.fit(np.transpose(np.array(x)))

    variance = lll.explained_variance_ratio_
    rest = 0
    for i in range(4,len(variance)):
      rest+=variance[i]
    bar = pd.DataFrame.from_dict({'year':["2019"],'dim1':[variance[0]*100],'dim2':[variance[1]*100],'dim3':[variance[2]*100],'dim4':[variance[3]*100],'rest':[rest*100]})
    bar.to_csv('static/pca.csv',index=False)
    lll = lll.components_


    n = int(len(lll[0])/2)
    a1 = lll[0][:n]
    a2 = lll[1][:n]
    b1 = lll[0][n:]
    b2 = lll[1][n:]
    
    cs = pd.DataFrame.from_dict({"x":np.concatenate([a1,b1]),"y":np.concatenate([a2,b2]),"name":tname,"color":col})

  
    #cs.to_csv('~/Desktop/vis paper/d3/2d.csv',index=False)

    '''plt.scatter(a1,a2,c='red',label='Objects')
    plt.scatter(b1,b2,c='blue',label='Subjects')
    plt.legend()
    #for i in range(len(a1)):
    #    plt.plot([a1[i],b1[i]],[a2[i],b2[i]],linewidth=0.4)
    plt.show()'''

    #print(pca.explained_variance_ratio_)  

    #print(pca.singular_values_) 
    return cs
  
if __name__ == '__main__':
  n = NN()
  n.nn_function('plural.pkl',80,0,"2",15,[1,1,1],1).to_csv('static/2d.csv',index=False)
