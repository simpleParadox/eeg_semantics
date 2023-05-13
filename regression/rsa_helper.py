'''
    File name: rsa_helper.py
    Author: Wenxuan Guo
    Date created: 7/27/2020
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# def make_rdm(embeds):
#     coef = np.corrcoef(embeds)
#     return coef


def make_rdm(embeds, type='cosine'):
    """
    Makes the general RDM for the input array.
    Args:
        embeds = The list of word embeddings. (Should be of size 16 for now).
        type = Distance metric to be used.
    """
    # number of stimuli
    num_stimuli = embeds.shape[0]
    # initialize RDM
    RDM = np.ones((num_stimuli, num_stimuli))
    # compute correlation distance between vectors
    # idxs = np.triu_indices(RDM.shape[0],1)
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            if type == 'pearson':
                RDM[i][j] = 1 - pearsonr(embeds[i], embeds[j])[0]
            elif type == 'cosine':
                RDM[i][j] = distance.cosine(embeds[i], embeds[j])

    return RDM

def corr_between_rdms(rdm1, rdm2, type='spearman'):
    # We will only calculate the correlation between the upper triangle of the matrices.
    num_stimuli = rdm1.shape[0]
    # initialize RDM
    RDM_corr = np.ones((num_stimuli, num_stimuli))
    # compute correlation distance between vectors
    # idxs = np.triu_indices(RDM_corr.shape[0], 1)
    rdm_corr = p_val = None
    if type == 'spearman':
        rdm_corr, p_val = spearmanr(rdm1[np.triu_indices(rdm1.shape[0], k=1)], rdm2[np.triu_indices(rdm2.shape[0], k=1)])
    elif type == 'pearson':
        rdm_corr, p_val = pearsonr(rdm1[np.triu_indices(rdm1.shape[0], k=1)], rdm2[np.triu_indices(rdm2.shape[0], k=1)])

    return rdm_corr, p_val

def eeg_filter_by_group(data, group, word):
    """
    Filter out data for different months - for single word. For example. 9 month olds - word 0, 12 month olds word 1, etc.
    Args:
        data: the eeg data
        group: the age group
        word: the word label
    Return:
        the dataset that contains the subset of eeg for an age group for a specific word.
    """
    if group == 9:
        data = data.iloc[1008:]
        data = data[data['label']==float(word)]#.iloc[:, :18000].values
    elif group == 12:
        data = data[:1008]
        data = data[data['label'] == float(word)]#.iloc[:,:18000].values
    else:
        data = data[data['label'] == float(word)]#.iloc[:, 18000].values

    # Now average the readings for each subject for that word.
    participants = [i for i in range(min(data['participant']), max(data['participant']))]
    eeg_means = []
    for part in participants:
        eeg_data = data[data['participant']==part].iloc[:, :18000].values
        eeg_means.append(np.mean(eeg_data, axis=0))
    eeg_means = np.array(eeg_means)
    return eeg_means, participants, 'all' + str(group)+ ' months', word


def eeg_filter_by_group_all_words(data, group):
    """
    Filter out data for different months - for all words For example. 9 month olds - all, 12 month all words, etc.
    Args:
        data: the eeg data
        group: the age group
    Return:
        the dataset that contains the subset of eeg for an age group for a specific word.
    """
    eeg_means = []
    if group == 9:
        data = data.iloc[1008:]
        # for word in range(16):
        #     eeg_means.append(np.nanmean(data[data['label']==float(word)].iloc[:, :18000].values, axis=0))  #.iloc[:, :18000].values
    elif group == 12:
        data = data[:1008]
        # for word in range(16):
        #     eeg_means.append(np.nanmean(data[data['label']==float(word)].iloc[:, :18000].values, axis=0))#.iloc[:,:18000].values

    for word in range(16):
        eeg_means.append(np.mean(data[data['label'] == float(word)].iloc[:, :18000].values, axis=0))


    # Now average the readings for each subject for that word.
    # participants = [i for i in range(min(data['participant']), max(data['participant']))]
    # eeg_means = []
    # for part in participants:
    #     eeg_data = data[data['participant']==part].iloc[:, :18000].values
    #     eeg_means.append(np.mean(eeg_data, axis=0))
    eeg_means = np.array(eeg_means)
    return eeg_means, group, 'all' + str(group) + ' months', 'all words'



def eeg_filter_subject(data, subject, word):
    """
    Returns the data for one subject only for a specific word
    Args:
        data: the whole dataset
        subject: the participant number
        word: the word label
    Returns:
        The data for a single participant for a specific word.
    """
    data = data[data['participant']==subject]
    print(data.shape)
    data = data[data['label']==float(word)].iloc[:,:18000]#.values
    print(data.shape)
    return data, subject, word


def eeg_filter_subject_all_words(data, ps, labels_array=None):
    """
    Filter out all the words averaged together for a single participants
    """
    data = data[data['participant'] == ps]
    if labels_array == None:
        labels = [i for i in range(16)]
    else:
        labels = labels_array
    ps_data = []
    for word in labels:
        words = data[data['label'] == float(word)].iloc[:, :18000].values
        ps_data.append(np.mean(words, axis=0))

    ps_data = np.array(ps_data)
    return ps_data, ps, 'all words'

def PCA_selection(X, percent_variance = .9):
    pca = PCA(svd_solver='auto')
    pca.fit(X)
    csum_variance = np.cumsum(pca.explained_variance_ratio_)
    # find the first q vectors that inform 90% of the variance
    q = len(csum_variance[csum_variance < percent_variance]) + 1

    pca = PCA(n_components=200, svd_solver='auto')
    pca.fit(X)
    score = pca.transform(X)

    return score

def PFA_selection(X, n_features, percent_variance = .9):
    """
    Args:
        X: input matrix of shape (nsamples, nfeatures)
        n_features: the number of features used for PFA
        percent_variance: lowest percentage of the variance explained by the first q components of PCA; default = 90%
    """
    q_evectors = PCA_selection(X, percent_variance=percent_variance)

    # Choose the subspace dimension q and construct the matrix
    # the transformed matrix A_q
    A_q = q_evectors.T
    # n_features should be higher than q
    kmeans = KMeans(n_clusters=n_features).fit(A_q)
    # For each cluster, only the vector closest to the center of cluster is retained,
    clusters = kmeans.predict(A_q)
    # Coordinates of cluster centers
    cluster_centers = kmeans.cluster_centers_

    dists = dict(list)
    for i, c in enumerate(clusters):
        dist = euclidean_distances(
            [A_q[i, :]], [cluster_centers[c, :]])[0][0]
        dists[c].append((i, dist))

    # the column indices of the kept features
    indices = [sorted(f, key=lambda x: x[1])[0][0]
                                for f in dists.values()]
    # the transformed matrix
    features = X[:, indices]

    return indices, features

def eeg_rdm_dist_corr(score, type='pearson'):
    """
    calculates an RDM from the fmri voxels using correlation distance
    Args:
        X: the original data
        evectors: the first q principal components
    """
    # number of stimuli
    num_stimuli = score.shape[0]
    # initialize RDM
    RDM = np.ones((num_stimuli, num_stimuli))
    # compute correlation distance between vectors
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            if type == 'pearson':
                RDM[i][j] = 1 - pearsonr(score[i], score[j])[0]
            else:
                RDM[i][j] = distance.correlation(score[i], score[j])
    
    return RDM



def word_RDM(model, label_list):
    # number of stimuli
    num_stimuli = len(label_list)
    # initialize RDM
    RDM = np.ones((num_stimuli, num_stimuli))
    # compute cosine similarity between words
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            try:
                similarity = model.wv.similarity(label_list[i], label_list[j])
                RDM[i][j] = 1 - similarity # dissimilarity
            except:
                pass
    
    return RDM


def alternate_ps_eeg_corr(score1, score2, type='pearson'):
    """
    calculates an RDM from the fmri voxels using correlation distance for two different subjects.
    Args:
        X: the original data
        evectors: the first q principal components
    """
    # number of stimuli
    num_stimuli = score1.shape[0]
    # initialize RDM
    RDM = np.ones((num_stimuli, num_stimuli))
    # compute correlation distance between vectors
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            if type == 'pearson':
                RDM[i][j] = 1 - pearsonr(score1[i], score2[j])[0]
            else:
                RDM[i][j] = distance.correlation(score1[i], score2[j])

    return RDM


def RDM_vis(RDM, subject, word, ps1=None, ps2=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    # plt.text(RDM)
    cmap = mpl.cm.GnBu
    # norm = mpl.colors.Normalize(vmin=5, vmax=10)

    # cb1 = mpl.colorbar.ColorbarBase(plt, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='vertical')
    # plt.matshow(RDM, cmap=cmap, vmin=0, vmax=1.6268522741479599)#np.max(RDM))
    plt.matshow(RDM, cmap=cmap)
    plt.colorbar()
    # plt.clim(vmin=1)
    if ps1 == None or ps2 == None:
        plt.title('RDM PS: '+ str(subject) + ' / word: ' + str(word))
        plt.xlabel('Axes denote words present for all subjects')
    else:
        print("yes")
        plt.title('RDM PS: ' + str(ps1) + ' vs ' + str(ps2) + '/ word: ' + str(word))
        plt.xlabel('PS ' + str(ps1))
        plt.ylabel('PS ' + str(ps2))
    plt.show()