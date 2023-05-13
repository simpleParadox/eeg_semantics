from jwlab.constants_null import word_list
import numpy as np

def label_remapping_sliding_window(wordlist_label0, df_list):
    assert type(wordlist_label0) == list
    
    index_label0 = []
    for i in wordlist_label0:
        index_label0 += [word_list.index(i)]
    
    ret = []
    for i in range(len(df_list)):
        temp = []
        for j in range(len(df_list[i])):
            label = list(df_list[i][j]['label'].values)
            for k in range(len(label)):
                label[k] = 0 if label[k] in index_label0 else 1
                
            temp += [np.asarray(label)]
        ret += [temp]
    
    return ret