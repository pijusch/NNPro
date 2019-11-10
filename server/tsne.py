import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time

def tsne_function(fil,rel,lines):
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    tot = pd.read_table('./data/total.csv',sep='\t',header=None)
    relations = list(set(list(tot[2])))
    with open('./embeddings/'+fil,'rb') as f:
        embed = pickle.load(f)
    tot = tot[tot[2] == rel]
    print(len(tot))
    #tot = tot.iloc[:100]
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
    print(len(lll))
    pca = TSNE(n_components=2)
    starttime = time.time()
    lll = pca.fit_transform(np.array(lll))
    #print(lll.explained_variance_ratio_)
    lll = np.array(lll)
    n = int(len(lll)/2)
    a1 = lll[:n,0]
    a2 = lll[:n,1]
    b1 = lll[n:,0]
    b2 = lll[n:,1]
    print(time.time()-starttime)
    linevar = 0
    if lines == True:
        linevar = 1
    x = np.concatenate([a1,b1])
    y =np.concatenate([a2,b2])
    color = [0]*n+[1]*n
    x_new = []
    y_new = []
    color_new = []
    dic = dict()
    nnm_new = []
    for i in range(len(x)):
        if nnm[i] in dic:
            continue
        dic[nnm[i]] = 0
        nnm_new.append(nnm[i])
        x_new.append(x[i])
        y_new.append(y[i])
        color_new.append( color[i])


    cs = pd.DataFrame.from_dict({'x':np.array(x_new),'y':np.array(y_new), 'name':nnm_new,'color':color_new,"line":linevar})
    #cs.to_csv('~/Desktop/vis paper/d3/pca.csv',index=False)
    #plt.scatter(a1,a2,c='red',label='Objects')
    #plt.scatter(b1,b2,c='blue',label='Subjects')
    #plt.legend()
    #for i in range(len(a1)):
    #    plt.plot([a1[i],b1[i]],[a2[i],b2[i]],linewidth=0.4)
    #plt.show()

    #print(pca.explained_variance_ratio_)

    #print(pca.singular_values_)
    #return cs.to_json()
    return cs

if __name__ == '__main__':
    fil = 'kg_ent.pkl'
    rel = 'language'
    tsne_function(fil,rel,True)
