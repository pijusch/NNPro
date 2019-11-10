import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pickle
def gen_set_function(fil,rel):
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    tot = pd.read_table('./data/total.csv',sep='\t',header=None)
    relations = list(set(list(tot[2])))
    with open('./embeddings/'+fil,'rb') as f:
        embed = pickle.load(f)
    #tot = tot[tot[2] == rel]
    tot = tot.iloc[:100000]
    #print(relations[int(rel)])
    #l = list(tot[0])
    #l+= list(tot[1])
    #l = list(set(l))
    ll = []
    names = []
    nnames = []
    for i in range(len(tot)):
        if tot.iloc[i][0] in embed:
            ll.append(embed[tot.iloc[i][0]])
            names.append(tot.iloc[i][0])
    for i in range(len(tot)):
        if tot.iloc[i][1] in embed:
            ll.append(embed[tot.iloc[i][1]])
            names.append(tot.iloc[i][1])

    ll = np.array(ll)
    n = (len(ll)/2)
    rm = list(set(np.argwhere(np.isnan(ll))[:,0]))
    lll = []
    for i in range(len(ll)):
        if i not in rm and ((i<n and (n+i) not in rm) or (i>=n and (i-n) not in rm)):
            lll.append(ll[i])
            nnames.append(names[i])
    lll = np.array(lll)
    x1 = []
    x2 = []
    y = []
    sub = []
    obj = []
    for i in range(int(len(lll)/2)):
        x1.append(lll[i])
        sub.append(nnames[i])
        x2.append(lll[i+int(len(lll)/2)])
        obj.append(nnames[i+int(len(lll)/2)])
        y.append(np.dot(x1[i],x2[i]))

    pic = [np.array(x1),np.array(x2),np.array(y),sub,obj]

    with open('gen_set.pkl','wb') as f:
        pickle.dump(pic,f)

if __name__ == '__main__':
    gen_set_function('kg_ent.pkl','language')
