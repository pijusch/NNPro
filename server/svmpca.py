import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import pickle
import time
def svmpca_function(fil,rel,lines):
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    if (fil!='plural.pkl'):
        tot = pd.read_table('./data/total.csv',sep='\t')
        with open('./embeddings/'+fil,'rb') as f:
            embed = pickle.load(f)
        #tot = tot[tot['2'] == rel]
        tot = tot.iloc[:100000]
        #tot = tot.iloc[:100]
        #tot = tot.iloc[:100]
        #print(tot.iloc[0])
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
    lll = np.array(lll)
    pca = PCA(n_components=1)
    start_time =time.time()
    llll = pca.fit(np.transpose(lll)).components_


    n = int(len(llll[0])/2)
    a1 = llll[0][:n]
    b1 = llll[0][n:]



    Y = [0]*n + [1]*n

    clf2 = svm.LinearSVC(C=1).fit(lll, Y)
    w =  clf2.coef_[0]
    ww = np.linalg.norm(w)
    xxx = []
    for i in range(len(lll)):
        xxx.append((np.dot(lll[i],w)+clf2.intercept_[0])/ww)

    a2 = xxx[:n]
    b2 = xxx[n:]

    linevar = 0
    if lines == True:
        linevar = 1
    cs = pd.DataFrame.from_dict({'x':np.concatenate([a1,b1]),'y':np.concatenate([a2,b2]), 'name':nnm,'color':[0]*n+[1]*n,'xx':np.concatenate([b1,a1]),'yy':np.concatenate([b2,a2]),"line":linevar})
    #plt.scatter(a2,a1,c='red',label = 'Objects')
    #plt.scatter(b2,b1,c='blue',label='Subjects')
    #for i in range(len(a1)):
    #    plt.plot([a2[i],b2[i]],[a1[i],b1[i]],linewidth=0.4)
    #plt.legend()
    #plt.show()

    #print(pca.explained_variance_ratio_)  

    #print(pca.singular_values_)  

    print(time.time()-start_time)
    return cs
if __name__ == '__main__':
    svmpca_function('kg_ent.pkl','language',True)