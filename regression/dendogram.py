from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from numpy import savez_compressed
from sklearn.preprocessing import StandardScaler

# ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
#                    400., 754., 564., 138., 219., 869., 669.])


labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}

# Load the pretrained word vectors.
model = KeyedVectors.load_word2vec_format("G:\jw_lab\jwlab_eeg\\regression\GoogleNews-vectors-negative300.bin", binary=True) # Pretrained Word2Vec vectors.
# model1 = KeyedVectors.load_word2vec_format("G:\\jw_lab\\jwlab_eeg\\tuned_vectors_pre_w2v_cbt_childes.bin", binary=True) # 200 iterations
# model2 = KeyedVectors.load_word2vec_format("G:\\jw_lab\\jwlab_eeg\\tuned_vectors_pre_w2v_cbt_childes_1.bin", binary=True) # 1000 iterations

def get_glove_6b_embeddings():
    embeddings_dict = {}
    with open("G:\jw_lab\jwlab_eeg\glove6B\glove.6B.50d.txt", 'r',encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    glove_pre_trained_embeds = []
    for stim in labels_mapping.values():
        glove_pre_trained_embeds.append(embeddings_dict[stim])
    glove_pre_trained_embeds = np.array(glove_pre_trained_embeds)
    # savez_compressed("G:\jw_lab\jwlab_eeg\\regression\glove_embeds\glove_pre_wiki_giga_200d.npz", glove_pre_trained_embeds)
    return glove_pre_trained_embeds


ytdist1 = get_glove_6b_embeddings()

residual_pretrained_w2v_path = "G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\pretrained_w2v_residuals.npz"
residual_tuned_w2v_path = "G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\\tuned_w2v_residuals.npz"

w2v_pretrained_residual = np.load(residual_pretrained_w2v_path, allow_pickle=True)['arr_0']


def get_embedding_from_model():
    # The model is initialized above.
    child_only_w2v = []
    for stim in labels_mapping.keys():
        # if stim == 'mom':
        child_only_w2v.append(w2v_pretrained_residual[stim])
        # else:
        #     child_only_w2v.append(model.wv[stim])

    child_only_w2v = np.array(child_only_w2v)

    # The following is to save the child_only_w2v vectors.
    # savez_compressed("G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\\tuned_w2v_cbt_childes_300d.npz", child_only_w2v)
    return child_only_w2v


# ytdist = load('G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\gen_w2v_embeds_avg_trial_and_ps.npz')
# ytdist = ytdist['arr_0']

ytdist1 = get_embedding_from_model()


# scaler = StandardScaler()
# ytdist = scaler.fit_transform(ytdist)

# Get the actual dendrogram values. change the parameter values.
Z = hierarchy.linkage(ytdist1, 'complete', metric='cosine')
plt.clf()
dn = hierarchy.dendrogram(Z)

custom_ticks = []
ticks_pos = np.arange(0,16)
for i in ticks_pos:
    custom_ticks.append(dn['ivl'][i] + " " + labels_mapping[int(dn['ivl'][i])])

plt.xlabel("Word")
locs, labels = plt.xticks()
plt.xticks(ticks=locs, labels=custom_ticks, rotation=270)
plt.ylabel('1 - cosine_similarity')
plt.title("Glove 50 dims (Complete linkage-Cosine Similarity)")

plt.show()

# cosine_similarity([ytdist[10]], [ytdist[11]])



#----------------------------------------------------------------------------------------------


# The following section shows analyzes the differences between the pretrained w2v embeddings
# and fine-tuned vectors.

# Idea is to rank them in ascending order.
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# First get both groups of vectors.
pre_vectors_data = load('G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\embeds_with_label_dict.npz', allow_pickle=True)['arr_0'][0]
tuned_vectors = load('G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\\tuned_w2v_cbt_childes_300d.npz', allow_pickle=True)['arr_0'].tolist()

pre_vectors = [vect.tolist() for vect in pre_vectors_data.values()]

# Calculate cosine distances.
distances = []
similarities = []
for i in range(16):
    distances.append(cosine(pre_vectors[i], tuned_vectors[i]))
    similarities.append(cosine_similarity([pre_vectors[i]], [tuned_vectors[i]]))
