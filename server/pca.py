import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time

def pca_function(fil,rel,lines):
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    if fil!='plural.pkl':
        tot = pd.read_table('./data/total.csv',sep='\t',header=None)
        relations = list(set(list(tot[2])))
        with open('./embeddings/'+fil,'rb') as f:
            embed = pickle.load(f)
        tot = tot[tot[2] == rel]
        #rint(len(tot))
        #tot = tot.iloc[:100000]
        #print(relations[int(rel)])
        #l = list(tot[0])
        #l+= list(tot[1])
        #l = list(set(l))
        ll = []
        nm = []

        for i in range(len(tot)):
            if tot.iloc[i][0] in embed:
                ll.append(embed[tot.iloc[i][0]])
                nm.append(tot.iloc[i][0])
        for i in range(len(tot)):
            if tot.iloc[i][1] in embed:
                ll.append(embed[tot.iloc[i][1]])
                nm.append(tot.iloc[i][1])
        nnm = []
        ll = np.array(ll)
        n = (len(ll)/2)
        rm = list(set(np.argwhere(np.isnan(ll))[:,0]))
        lll = []
        for i in range(len(ll)):
            if i not in rm and ((i<n and (n+i) not in rm) or (i>=n and (i-n) not in rm)):
                lll.append(ll[i])
                nnm.append(nm[i])
    else:
        with open(fil,'rb') as f:
            temp = pickle.load(f)
            lll = np.array(temp[0]+temp[1])
            nnm = temp[3]+temp[4]
    print(len(lll))
    pca = PCA(n_components=2)
    start_time = time.time()
    lll = pca.fit(np.transpose(np.array(lll)))
    print(lll.explained_variance_ratio_)
    lll = lll.components_

    n = int(len(lll[0])/2)
    a1 = lll[0][:n]
    a2 = lll[1][:n]
    b1 = lll[0][n:]
    b2 = lll[1][n:]

    linevar = 0
    if lines == True:
        linevar = 1
    cs = pd.DataFrame.from_dict({'x':np.concatenate([a1,b1]),'y':np.concatenate([a2,b2]), 'name':nnm,'color':[0]*n+[1]*n,'xx':np.concatenate([b1,a1]),'yy':np.concatenate([b2,a2]),"line":linevar})
    #cs.to_csv('~/Desktop/vis paper/d3/pca.csv',index=False)
    #plt.scatter(a1,a2,c='red',label='Objects')
    #plt.scatter(b1,b2,c='blue',label='Subjects')
    #plt.legend()
    #for i in range(len(a1)):
    #    plt.plot([a1[i],b1[i]],[a2[i],b2[i]],linewidth=0.4)
    #plt.show()

    #print(pca.explained_variance_ratio_)

    #print(pca.singular_values_)
    print(time.time()-start_time)

    return cs


if __name__ == '__main__':
    fil = 'kg_ent.pkl'
    rel = 'language'
    pca_function(fil,rel,True)
