# import sys
# sys.path.append('G:\jw_lab\jwlab_eeg\classification\code')
from jwlab.constants import old_participants

def map_participants(ys, participants):
    for i in range(len(participants)):
        if participants[i] in old_participants:  # if old participants
            continue
        else:  # if new participants
            ys[i] = squish_new_participants(ys[i])
    return ys


def squish_new_participants(y):
    for idx, c_y in enumerate(y):
        if c_y > 16 and c_y <= 32:
            y[idx] = c_y - 16
    return y
