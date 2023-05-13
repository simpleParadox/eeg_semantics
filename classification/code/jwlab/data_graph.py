import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_good_trial_participant(participants, data):
    x = np.arange(len(participants))
    fig, ax = plt.subplots()
    rects = ax.bar(x, data, 0.6, label='good trials')
    ax.set_xticks(x)
    ax.set_xticklabels(participants)
    ax.set_xlabel('Participants')
    ax.set_ylabel('Good trials (%)')
    ax.set_title('Good trials (%) left for each participant')
    autolabel(rects, ax)
    plt.show()


def plot_good_trial_word(participant, data):
    new_names = {}
    for i in range(len(participant)):
        new_names[i] = int(participant[i])
    # pandas would automatically convert int to float if nan exists
    df = pd.DataFrame(data).T
    df = df.replace(np.nan, -1)
    df = df.astype(int)
    df = df.replace(-1, "-")
    df = df.rename(columns=new_names)
    display(df)
    
def generate_window_label(prediction_error):
    window_len_list = []
    for i in prediction_error:
        window_len_list.append(1100-100*len(i))
    window_label = []
    for i in window_len_list:
        count = 0
        while count <= 1000-i:
            window_label.append(str(count)+"-"+str(count+i))
            count += 100
    return window_label

def plot_error_rate_window_single(prediction_error, ps):
    count = len(prediction_error)
    data = [round(item,2) for item in prediction_error]
    x = np.arange(count)
    fig, ax = plt.subplots()
    rects = ax.bar(x, data, label='error rate')
    ax.set_xticks(x)
    labels = ["0-100", "0-200", "100-200"]
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sliding windows (ms)')
    ax.set_ylabel('Error rate (%)')
    ax.set_title('Error rate for each sliding window')
    autolabel(rects, ax)
    plt.savefig('{0}_avg_3.png'.format(ps))


def plot_error_rate_window(prediction_error, old):
    count = sum([len(item) for item in prediction_error])
    data = [round(error_rate,2) for item in prediction_error for error_rate in item]
    x = np.arange(count)
    fig, ax = plt.subplots(figsize=(30,10))
    rects = ax.bar(x, data, label='error rate')
    ax.set_xticks(x)
    ax.set_xticklabels(generate_window_label(prediction_error))
    ax.set_xlabel('Sliding windows (ms)')
    ax.set_ylabel('Error rate (%)')
    ax.set_title('Error rate for each sliding window')
    autolabel(rects, ax)
    plt.savefig('first_20_{0}.png'.format(old))
    
def run_plot(participants, good_trial_count):
    plot_good_trial_participant(participants, good_trial_count[0])
    plot_good_trial_word(participants, good_trial_count[1])
