import pandas as pd
import numpy as np
import random
from scipy import stats
import more_itertools as mit
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import sys
import time
import matplotlib as mpl
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from scipy.cluster import hierarchy

# pattern = tx.compile(['baby', 'bat', 'bad'])
# hits = pattern.findall('The baby was scared by the bad bat.')
# hits = ['baby', 'bat', 'bad']
sys.path.insert(1, 'classification/code')  ## For loading the following files.

from jwlab.ml_prep_perm import prep_ml, prep_matrices_avg, remove_samples
from matplotlib import pyplot as plt

sys.path.insert(1, '../eeg_semantics')
from regression.functions import get_w2v_embeds_from_dict, two_vs_two, extended_2v2_phonemes, extended_2v2_perm, \
    get_phoneme_onehots, get_phoneme_classes, get_sim_agg_first_embeds, get_sim_agg_second_embeds, extended_2v2, w2v_across_animacy_2v2, w2v_within_animacy_2v2, \
    ph_within_animacy_2v2, ph_across_animacy_2v2, get_audio_amplitude, get_stft_of_amp, get_tuned_cbt_childes_w2v_embeds, get_all_ph_concat_embeds, \
    get_glove_embeds, get_cbt_childes_50d_embeds, get_reduced_w2v_embeds, sep_by_prev_anim, prep_filtered_X, get_residual_pretrained_w2v, get_residual_tuned_w2v, \
    extended_2v2_mod, corr_score


# from regression.rsa_helper import make_rdm, corr_between_rdms

from sklearn.linear_model import Ridge

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}


def process_temp_results(data):
    # This function processes the temp_results and return the average for all the iterations of each window.
    temp_results_dict = [d[0] for d in data]
    d_list = []
    for i in range(len(temp_results_dict)):
        int_dict = {}
        for key, val in temp_results_dict[i].items():
            int_dict[key] = val[0]
        d_list.append(int_dict)
    res_df = pd.DataFrame(d_list)
    answer = dict(res_df.mean())
    answer_dict = {}
    for key, val in answer.items():
        answer_dict[key] = [val]
    return [answer_dict]


def process_permuted_tgm_res(tgm_perm_list):
    # Just average the tgms across those '50' xval iterations to get one TGM.
    avg_tgm = np.mean(tgm_perm_list, axis=0)
    return avg_tgm
# Use this when training only on minimal mouth information words. These are the words which will be excluded.
minimal_mouth_labels_exclude = {4: 'cat', 5: 'dog', 7: 'mom',
                  8: 'banana', 9: 'bottle', 11: 'cracker'}

# Use this when training only on non-minimal mouth information words. These are the words which will be excluded.
minimal_mouth_labels_include = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                    6: 'duck', 10: 'cookie', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}


first_sound_visible_on_face = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny', 7: 'mom', 8: 'banana', 9: 'bottle', 14: 'milk'}
not_fist_sound_visible_on_face = {4: ' cat', 5: 'dog', 6: 'duck', 10: 'cookie', 11: 'cracker', 12: 'cup', 13: 'juice', 15: 'spoon'}


def minimal_mouth_X(X):
    """
    This function returns the dataframe with the minimal mouth information words only.

    """

    # First get the row indexes to be deleted.
    i = j = 0
    idxs = []
    for key, word in first_sound_visible_on_face.items():
        idxs.extend(X[i][j][X[i][j]['label'] == float(key)].index) # Indexes to remove.

    X_mod = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            # X[i][j] is a dataframe.
            temp = X[i][j].drop(idxs)
            X_mod.append(temp)
    
    return [X_mod]


def cluster_analysis_procedure(age_group, useRandomizedLabel, averaging, sliding_window_config, cross_val_config, type='simple', residual=False, child_residual=False, seed=-1, corr=False, target_pca=False, animacy=False, no_animacy_avg=False, do_eeg_pca=False, do_sliding_window=False,
                               embeddings='w2v', age_group_1=9, age_group_2=12):
    print("Cluster analysis procedure")
    num_folds, cross_val_iterations, sampling_iterations = cross_val_config[0], cross_val_config[1], cross_val_config[2]


    sampl_iter_word_pairs_2v2 = []
    results = {}
    successful_iterations = 0
    animacy_results = {}
    preds_results = {}
    # null_dist_results = []
    tgm_results = []
    tgm_diff_results = []
    tgm_both_diff_results = []
    tgm_neg_diff_results = []
    equal_count_results = []
    roe_matrix_results = []
    flag = 0
    w2v_res_list = []
    cbt_res_list = []
    r2_train_values = []
    r2_test_values = []

    if seed > 0:
        seed_range = np.arange((seed - 1) * sampling_iterations, seed * sampling_iterations, 1)
    # else:
    #     pos_seed = seed * -1
    #     seed_range = np.arange((pos_seed - 1) * sampling_iterations, pos_seed * sampling_iterations)


    if residual == True:
        # Calculate the residual phoneme vectors here by calling the function.
        print('Calculating residuals')
        w2v_residuals, r2 = cv_all_ph_concat_padded_residual_mod(child=child_residual)


    for i in range(sampling_iterations):
        print("Sampling iteration: ", i, flush=True)
        if averaging == "permutation":
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config,
                                                      downsample_num=1000)

            # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel)  # This is a new statement.

            temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)

            # temp_results = cross_validaton_averaging(X_train, X_test, y_train, y_test, useRandomizedLabel)

        elif averaging == "average_trials_and_participants":
            flag = 0

            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, "no_average_labels", sliding_window_config, downsample_num=1000, animacy=animacy)
            # print("Y after prep_ml is: ", y)
            # print("Raveled after prep_ml is: ", np.ravel(y))
            # print("Length of raveled Y after prep_ml is: ", np.ravel(y).shape)
            # print("Good trial count: ", good_trial_count)
            # X = remove_samples(X) # Use only for the 9 month olds.
            # X = minimal_mouth_X(X)


            # For residual stuff.
            # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=True, test_size=0)

            # The following two lines are for the experiments where we divide the dataset based on the animacy of the previous trial.
            # filtered_X = sep_by_prev_anim(X,y, current_type='inanimate', prev_type='animate')
            # X_train, X_test, y_train, y_test = prep_filtered_X(filtered_X)

            if type == 'permutation':
                # The split of the dataset into train and test set happens more than once here for each permuted label assignment.
                temp_results_list = []
                print('Permutation')
                raise NotImplementedError
                # for i in range(50):
                #     X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.2)
                #     cv_results, temp_animacy_results, temp_preds, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train, y_train, X_test, y_test)
                #
                #     # temp_results, temp_diag_tgm, word_pairs_2v2_sampl = cv_residual_w2v_ph_eeg(X_train, X_test, y_train, y_test)
                #
                #     temp_results_list.append(temp_results)
                #
                # # Process temp_results_list to obtain a single "temp_results" list.
                # temp_results = process_temp_results(temp_results_list)
                # temp_results

            else:
                # Non-permuted test.
                print('Simple')
                # Get the indices of the stimuli words which will be later used to get the embeddings.
                try:
                    if animacy:
                        X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, False, train_only=False, test_size=0.20, animacy=animacy, no_animacy_avg=no_animacy_avg)
                    else:
                        # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.20, animacy=animacy, no_animacy_avg=no_animacy_avg, current_seed=i)
                        X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.20, animacy=animacy, no_animacy_avg=no_animacy_avg, current_seed=i)
                    successful_iterations += 1
                    # print(f"Shape of X_test and y_test {X_test[0][0].shape} {y_test[0][0].shape}")
                except Exception as e:
                    print("Error in prep_matrices_avg, continuing to the next sampling iteration.")
                    print(e)
                    continue

                # if residual == True:
                #     print('Residual is true')
                # temp_results, temp_diag_tgm, word_pairs_2v2_sampl = cv_residual_w2v_ph_eeg_mod(X_train, X_test, y_train, y_test, w2v_residuals)
                # else:
                # if not animacy:
                temp_results, temp_animacy_results, temp_preds, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train, y_train, X_test, y_test, animacy=animacy, iteration=i, embeddings=embeddings)
                # else:
                #     try:
                #         temp_results = cv_animacy(X_train, y_train, X_test, y_test, do_eeg_pca=do_eeg_pca, do_sliding_window=do_sliding_window)
                #     except Exception as  e:
                #         print("Something wrong in cv_animacy, continuing to next iteration.")
                #         print(e)
                #         continue
                    


                # temp_results, temp_diag_tgm, word_pairs_2v2_sampl, r2_train, r2_test = cv_residual_w2v_ph_eeg(X_train, X_test, y_train, y_test, child=True)
                # r2_train_values.append(r2_train)
                # r2_test_values.append(r2_test)

                # w2v_res, cbt_res,  = cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test)

            # w2v_res_list.append(w2v_res)
            # cbt_res_list.append(cbt_res)



            # For phonemes and w2v embeddings.
            # temp_results, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train, y_train, X_test, y_test)

            # sampl_iter_word_pairs_2v2.append(word_pairs_2v2_sampl)  # For stimuli related curve.
            #
            # For concatenation of w2v and phonemes, concats etc.
            # temp_results, temp_diag_tgm = cross_validaton_nested_concat(X_train, y_train, X_test, y_test)
            
            # tgm_results.append(temp_diag_tgm)

            if sampling_iterations == 0:
                print("Warning: This does not do fold validation")
        elif averaging == "tgm":
            # TGM within group.
            print('TGM')
            if seed >= 0:
                current_seed = seed_range[i]
                print("Seed range: ", seed_range)
                print("Current seed: ", current_seed)
            else:
                current_seed = -1
            try:
                print("Inside try")
                X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, "no_average_labels",
                                                      sliding_window_config, downsample_num=1000, current_seed=current_seed)
            except Exception as e:
                print(e)

            if type == 'permutation':
                # The split of the dataset into train and test set happens more than once here for each permuted label assignment.
                temp_results_list = []
                print('Permutation')
                for i in range(50):
                    X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.2)
                    tgm_results_res = cross_validaton_tgm(X_train, y_train, X_test, y_test, child=False, res=False)
                    temp_results_list.append(tgm_results_res)  # The temp_results is expected to be a square matrix.
                # Process temp_results_list to obtain a single "temp_results" list.
                temp_tgm_results = process_permuted_tgm_res(temp_results_list)
                tgm_results.append(temp_tgm_results)
            else:
                print('Simple: tgm')
                try:
                    # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.2, current_seed=i)
                    X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.2)
                    successful_iterations += 1
                except:
                    print("Error in prep_matrices_avg, continuing to next iteration.")
                    continue
                tgm_results_res, tgm_diff_res, tgm_both_diff_res, tgm_neg_diff_res, equal_count_matrix, roe_matrix = cross_validaton_tgm(X_train, y_train, X_test, y_test, child=False, res=False, target_pca=target_pca)
                tgm_results.append(tgm_results_res)  # The temp_results is expected to be a square matrix.
                tgm_diff_results.append(tgm_diff_res)
                tgm_both_diff_results.append(tgm_both_diff_res)
                tgm_neg_diff_results.append(tgm_neg_diff_res)
                equal_count_results.append(equal_count_matrix)
                roe_matrix_results.append(roe_matrix)

        elif averaging == 'across':
            print('Across')
            # Other group was loaded before.
            age_group_1 = 9
            age_group_2 = 12
            try:
                print(f"Loading age group 1: age group 1 - {age_group_1}")
                X_1, y_1, good_trial_count_1, num_win_1 = prep_ml(age_group_1, useRandomizedLabel, "no_average_labels", sliding_window_config, downsample_num=1000)
                num_win = num_win_1

                print("Preparing age group 1")
                X_train_1, X_test_1, y_train_1, y_test_1 = prep_matrices_avg(X_1, age_group_1, useRandomizedLabel, train_only = True, test_size=0)
                print(f"Loading age group 2: age group 2 - {age_group_2}")
                X_2, y_2, good_trial_count_2, num_win_2 = prep_ml(age_group_2, useRandomizedLabel, "no_average_labels", sliding_window_config, downsample_num=1000)
                # Now the other group
                print("Preparing age group 2")
                X_train_2, X_test_2, y_train_2, y_test_2 = prep_matrices_avg(X_2, age_group_2, useRandomizedLabel, train_only = False, test_size=0.9, current_seed=i)
            except Exception as e:
                print(e)
                print("Error in loading or preparing data, moving to next iteration.")
                continue
                
            if type == 'tgm':
                print(type)
                tgm_results_res, tgm_diff_res, tgm_both_diff_res, tgm_neg_diff_res, equal_count_matrix, roe_matrix = cross_validaton_tgm(X_train_1, y_train_1, X_test_2, y_test_2, child=False, res=False, target_pca=target_pca)
                tgm_results.append(tgm_results_res)  # The temp_results is expected to be a square matrix.
                tgm_diff_results.append(tgm_diff_res)
                tgm_both_diff_results.append(tgm_both_diff_res)
                tgm_neg_diff_results.append(tgm_neg_diff_res)
                equal_count_results.append(equal_count_matrix)
                roe_matrix_results.append(roe_matrix)
            else:
                print(type)
                temp_results, temp_animacy_results, temp_preds, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train_1, y_train_1, X_test_2, y_test_2, embeddings=embeddings
                                                                                                                            )
                # temp_results, temp_diag_tgm, word_pairs_2v2_sampl = cv_residual_w2v_ph_eeg(X_train_1, X_test_2, y_train_1, y_test_2, child=False)
            
            

        else:
            print("Warning: This will only use the requested averaging matrix to perform a cross val")
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)

            temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)


        # for i in range(len(temp_preds)):
        #     if i not in preds_results.keys():
        #         preds_results[i] = {}
        #     for j in range(len(temp_preds[i])):
        #         if j in preds_results[i].keys():
        #             preds_results[i][j] += temp_preds[i][j]
        #         else:
        #             preds_results[i][j] = temp_preds[i][j]
    
    
    # if flag == 0:
    # The following code list is for predicting animacy from predicted word embeddings.
        # for i in range(len(temp_animacy_results)):
        #     if i not in animacy_results.keys():
        #         animacy_results[i] = {}
        #     for j in range(len(temp_animacy_results[i])):
        #         if j in animacy_results[i].keys():
        #             animacy_results[i][j] += temp_animacy_results[i][j]
        #         else:
        #             animacy_results[i][j] = temp_animacy_results[i][j]


        if averaging == 'average_trials_and_participants' or (averaging=='across' and type!='tgm'):
            # print("Results", results)
            for i in range(len(temp_results)):
                if i not in results.keys():
                    results[i] = {}
                for j in range(len(temp_results[i])):
                    if j in results[i].keys():
                        results[i][j] += temp_results[i][j]
                    else:
                        results[i][j] = temp_results[i][j]
    print("Successful sampling iterations: ", successful_iterations)


    # NOTE: Next three lines for storing the .npz arrays.
    # perm_results = np.array(results)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # np.savez_compressed(f'/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/12m fine tuned res/{timestr}.npz', perm_results)
    # else:
    #Averaging was of type 'tgm'.
    ## NOTE: The following are for TGMs only (non-permuted).
    ##Calculate average of all the matrices across 'n' sampling iterations. 
    if averaging == 'tgm' or (averaging == 'across' and type=='tgm'):
        if corr:
            # Use correlation instead of 2 vs 2.
            corr_file_name = f"{age_group}m_corr"
            if useRandomizedLabel:
                npz_folder_path = f"{age_group}m_corr"
                final_roe_matrix_tgm = np.mean(roe_matrix_results, axis=0)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                print("Saving the permuted correlation matrix")
                np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/permutation/{npz_folder_path}/{timestr}_corr_seed_{seed}.npz", final_roe_matrix_tgm)
            else:
                final_roe_matrix_tgm = np.mean(roe_matrix_results, axis=0)
                step_size = sliding_window_config[3]
                ind = np.arange(-150, 1050, step_size).tolist()
                cols = np.arange(-150, 1050, step_size).tolist()
                corr_df = pd.DataFrame(data=final_roe_matrix_tgm, index=ind, columns=cols)
                corr_df.to_csv(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{corr_file_name}.csv")
                plt.clf()
                truth = corr_df.values
                fig, singax= plt.subplots()
                thresh_fdr = np.max(truth) # For 9m_tgm_pre_w2v
                plax, im = plot_image(truth, [-100, 1000], mask=truth > thresh_fdr, ax=singax,
                        draw_mask=True, draw_contour=True, colorbar=True,
                        draw_diag=True, draw_zerolines=True, xlabel='Test time (ms)', ylabel='Train time (ms)',
                        cbar_unit="Correlation", cmap="RdBu_r", mask_alpha=1, mask_cmap="RdBu_r")
                
                plt.savefig(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{corr_file_name}.png")
        else:
            # file_name = f"{age_group}_detrending_low_pass_only_reref_with_baseline_filepath_mod"
            # Using the 2 vs 2 test.
            file_name = f"{age_group}m_tgm"
            if averaging == 'across':
                file_name = f"{age_group_1}m_to_{age_group_2}m_2v2_tgm"
            if useRandomizedLabel:
                npz_folder_path = f"{age_group}m_2v2_tgm_results"
                if averaging == 'across':
                    npz_folder_path = f"across_{age_group_1}_to_{age_group_2}_2v2_tgm"
                # Permutation test.
                final_tgm = np.mean(tgm_results, axis=0)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                final_equal_count_tgm = np.mean(equal_count_results, axis=0)
                # np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/permutation/{npz_folder_path}/{timestr}_{file_name}_seed_{seed}.npz", final_tgm)
                # np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/permutation/{npz_folder_path}/{timestr}_equal_count_seed_{seed}.npz", final_equal_count_tgm)
            else:
                # Non-permuted test.
                final_tgm = np.mean(tgm_results, axis=0)
                # Save the tgm in a csv file.
                step_size = sliding_window_config[3]
                ind = np.arange(-150, 1050, step_size).tolist()
                cols = np.arange(-150, 1050, step_size).tolist()
                df = pd.DataFrame(data=final_tgm, index=ind, columns=cols)
                truth = df.values
                fig, singax= plt.subplots()
                thresh_fdr = 10.5355 # For 9m_tgm_pre_w2v
                # df.to_csv(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_2v2.csv")
                plax, im = plot_image(truth, [-100, 1000], mask=truth > thresh_fdr, ax=singax, vmax=0.65, vmin=0.35,
                        draw_mask=True, draw_contour=True, colorbar=True,
                        draw_diag=True, draw_zerolines=True, xlabel='Test time (ms)', ylabel='Train time (ms)',
                        cbar_unit="2 vs. 2 accuracy", cmap="RdBu_r", mask_alpha=1, mask_cmap="RdBu_r")
                plt.savefig(f"{file_name}_2v2.png")

                # final_tgm1 = np.mean(tgm_diff_results, axis=0)
                # df1 = pd.DataFrame(data=final_tgm1, index=ind, columns=cols)
                # df1.to_csv(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_diff.csv")

                # final_tgm2 = np.mean(tgm_both_diff_results, axis=0)
                # df2 = pd.DataFrame(data=final_tgm2, index=ind, columns=cols)
                # df2.to_csv(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_both_diff.csv")


                # final_tgm3 = np.mean(tgm_neg_diff_results, axis=0)
                # df3 = pd.DataFrame(data=final_tgm3, index=ind, columns=cols)
                # df3.to_csv(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_neg_diff.csv")

                # truth = df1.values
                # plt.clf()
                # fig, singax= plt.subplots()
                # thresh_fdr = np.max(truth) # For 9m_tgm_pre_w2v
                # plax, im = plot_image(truth, [-100, 1000], mask=truth > thresh_fdr, ax=singax,
                #         draw_mask=True, draw_contour=True, colorbar=True,
                #         draw_diag=True, draw_zerolines=True, xlabel='Test time (ms)', ylabel='Train time (ms)',
                #         cbar_unit="2 vs. 2 accuracy", cmap="RdBu_r", mask_alpha=1, mask_cmap="RdBu_r")
                #
                # plt.savefig(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_diff.png")
                #
                #
                # truth = df2.values
                # plt.clf()
                # fig, singax= plt.subplots()
                # thresh_fdr = np.max(truth) # For 9m_tgm_pre_w2v
                # plax, im = plot_image(truth, [-100, 1000], mask=truth > thresh_fdr, ax=singax,
                #         draw_mask=True, draw_contour=True, colorbar=True,
                #         draw_diag=True, draw_zerolines=True, xlabel='Test time (ms)', ylabel='Train time (ms)',
                #         cbar_unit="2 vs. 2 accuracy", cmap="RdBu_r", mask_alpha=1, mask_cmap="RdBu_r")
                #
                # plt.savefig(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_both_diff.png")
                #
                #
                # truth = df3.values
                # plt.clf()
                # fig, singax= plt.subplots()
                # thresh_fdr = np.max(truth) # For 9m_tgm_pre_w2v
                # plax, im = plot_image(truth, [-100, 1000], mask=truth > thresh_fdr, ax=singax,
                #         draw_mask=True, draw_contour=True, colorbar=True,
                #         draw_diag=True, draw_zerolines=True, xlabel='Test time (ms)', ylabel='Train time (ms)',
                #         cbar_unit="2 vs. 2 accuracy", cmap="RdBu_r", mask_alpha=1, mask_cmap="RdBu_r")
                #
                # plt.savefig(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results_rohan_new/{file_name}_neg_diff.png")

            

        


    # NOTE: This following code section is only for TGMs permutated labels.
    # perm_results = np.array(tgm_results)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # np.savez_compressed(f'/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/tgms/Across/12_to_9m_pre_w2v_xval/{timestr}.npz', perm_results)

    # For predicting animacy from predictions.
    # pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg = t_test(animacy_results, num_win, num_folds)

    # adj_clusters_pos, adj_clusters_neg, clusters_pos, clusters_neg = find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg)

    # max_abs_tmass, t_mass_pos, t_mass_neg = get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg)

    # Uncomment the following three lines for the regualar (EEG->word embedding stuff stats calculation.)
    # For predicting raw w2v from EEG.
    
    
    # # dendrogram(np.mean(w2v_res_list, axis=0))
    # ## REMOVE FOR NULL FUNCTION
    if averaging == 'average_trials_and_participants' or (averaging=='across' and type!='tgm'):
        if useRandomizedLabel:
            # final_tgm = np.mean(results, axis=0)
            file_name = f'animacy_{age_group}m'
            timestr = time.strftime("%Y%m%d-%H%M%S")
            folder_path = f"animacy_{age_group}m"
            if averaging == 'across':
                folder_path = f"{age_group_1}_to_{age_group_2}_mod_2v2_cleaned2_90_test"
            np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/same_time_results/permutation/{folder_path}/{timestr}_{file_name}_all_data.npz", results)
        else:
            # pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg = t_test(results, num_win, num_folds)

            # adj_clusters_pos, adj_clusters_neg, clusters_pos, clusters_neg = find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg)

            # max_abs_tmass, t_mass_pos, t_mass_neg = get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg)
            if len(sliding_window_config[2]) == 1:
                # print('R2 train values: ', r2_train_values)
                # print('R2 test values: ', r2_test_values)
                print("Results: ", results)
                createGraph(results)#, t_mass_pos, adj_clusters_pos)
                
                # # word_pair_graph(sampl_iter_word_pairs_2v2)
                # print(animacy_results)
                # createGraph(animacy_results, tvalues_pos, adj_clusters_pos)
                
            else:
                print("Graph function is not supported for multiple window sizes")

        # rsa(np.mean(w2v_res_list, axis=0), np.mean(cbt_res_list, axis=0))

    return results


def process_pair_graph(data_dict):
    # This function is for the stimuli comparison.
    all_stim_bin_dict = {}
    window_len = len(data_dict[0])

    for wind in range(window_len):
        window_dict = {}
        all_stim_wind_avg = []
        for iter_dict in data_dict: # 'iter_dict' is for each sampling iteration.
            window_word_dict = iter_dict[wind] # {0: [], 1: [], ...}
            for key, value in window_word_dict.items():
                if key not in window_dict.keys():
                    window_dict[key] = value
                else:
                    window_dict[key].extend(value)

        # Now calculate the average.
        for k in range(0,16):
            all_stim_wind_avg.append(np.mean(window_dict[k]))

        all_stim_bin_dict[wind] = all_stim_wind_avg
        # Now for each window we have a 2d array of size (should be) 16x1. The 16x1 array is averaged across 50 iterations for each stimuli.

    # Now process some more to get them lined up in one list. So finally we will have a 2d matrix where each row will be for one word.
    avg_pair_acc_dict = {}
    for stim in range(0,16):
        avg_pair_acc_dict[stim] = []
        for window, acc in all_stim_bin_dict.items():
            avg_pair_acc_dict[stim].append(acc[stim])

    # Now we should have a 2d array where for each stimuli there will be a list containing the accuracies for each window.

    return avg_pair_acc_dict

def word_pair_graph(sampl_iter_word_pairs_2v2):
    # Structure -> [{0: {0: [], 1: [], ...}, 1: {0: [], 1: []}, ...}, {}]
    # First process the dictionary.
    # This function is for the stimuli comparison.
    avg_pair_acc_dict = process_pair_graph(sampl_iter_word_pairs_2v2)

    plt.clf()
    plt.figure()
    length_per_window_plt = 1200 / len(avg_pair_acc_dict[0])
    x_graph = np.arange(-200, 1000, length_per_window_plt)
    for stim, accs in avg_pair_acc_dict.items():
        if stim < 8:
            plt.plot(x_graph, accs, label=str(stim))
        else:
            break
    plt.legend(loc=1, bbox_to_anchor=(1.20, 1.0), title="12m Animate")
    plt.xlabel("Time (ms)")
    plt.ylabel("2v2 Accuracy")
    plt.title("Accuracy over time given word stimuli 12m (animate) 50iters")
    plt.savefig("03-04-2021 12m pre-w2v animate stim comparison")


    plt.clf()
    plt.figure()
    for stim, accs in avg_pair_acc_dict.items():
        if stim >= 8:
            plt.plot(x_graph, accs, label=str(stim))
    plt.legend(loc=1, bbox_to_anchor=(1.20, 1.0), title="12m Inanimate")
    plt.xlabel("Time (ms)")
    plt.ylabel("2v2 Accuracy")
    plt.title("Accuracy over time given word stimuli 12m (inanimate) 50iters")
    plt.savefig("03-04-2021 12m pre-w2v inanimate stim comparison")

def plt_rdm(mat, title, f_name):
    plt.clf()
    fig = plt.figure()
    corr = mat
    mask = np.zeros_like(corr)
    labels = list(labels_mapping.values())
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(corr, mask=mask,
                     xticklabels=labels, yticklabels=labels,
                     vmin=0.1, vmax=1.3,
                     cmap="YlGnBu", cbar_kws={'label': "Dissimilarity \n Cosine Distance"})
    # im = ax.matshow(mat, cmap=mpl.cm.jet)
    # fig.colorbar(im)
    # ax.set(xticks=range(n), xticklabels=labels)
    # # ax.set_xticks(np.arange(n))
    # # ax.set_xticklabels(labels)
    # ax.set_yticks(np.arange(n))
    # ax.set_yticklabels(labels)

    # ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    # Rotate and align top ticklabels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
    #          ha="left", va="center", rotation_mode="anchor")

    ax.set_title(title, pad=20)
    fig.tight_layout()
    # plt.show()

    fig.savefig(f_name)


def rsa(embeds_w2v, embeds_cbt):
    # Default metric is 'cosine'.
    rdm_w2v = make_rdm(embeds_w2v)
    rdm_cbt = make_rdm(embeds_cbt)

    plt_rdm(np.copy(rdm_w2v), "W2V res all_ph", "W2V res embeds RDM scaled 2 ")
    plt_rdm(np.copy(rdm_cbt), "CBT_cdes res all_ph", "CBT_chdes res RDM scaled 2")

    # 'spearman' is default correlation metric.
    rdm_corr, p_val = corr_between_rdms(rdm_w2v, rdm_cbt)
    print(f"Spearmans Correlation(similarity) between all_ph w2v res and cbt res is {rdm_corr} with p-value: {p_val}")



def dendrogram(ytdist):
    Z = hierarchy.linkage(ytdist, 'complete', metric='cosine')
    plt.clf()
    dn = hierarchy.dendrogram(Z)

    custom_ticks = []
    ticks_pos = np.arange(0, 16)
    for i in ticks_pos:
        custom_ticks.append(dn['ivl'][i] + " " + labels_mapping[int(dn['ivl'][i])])

    plt.xlabel("Word")
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs, labels=custom_ticks, rotation=270)
    plt.ylabel('1 - cosine_similarity')
    plt.title("cbt_full_childes_treebank non-scaled (Complete linkage-Cosine Similarity)")
    plt.savefig("cbt_full_childes residual non-scaled (Complete-cosine)")

    

def createGraph(results, age_group, embeddings):
    scoreMean = []
    stdev = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            scoreMean.append(round(np.mean(results[i][j]), 4))
            stdev.append(round(stats.sem(results[i][j]), 4))

    length_per_window_plt = 10  # 1200 / len(scoreMean)

    x_graph = np.arange(-200, 910, length_per_window_plt)
    x_graph += 100  # NOTE: Change this based on the window size.
    y_graph = scoreMean

    stdevplt = np.array(stdev)
    error = stdevplt
    # Done: Find the above chance accuracy x1 and x2 values for shading. Do you really need the tvalues_pos?
    # You need tvalues_pos if you are shading the region with max t-value. This might be a good idea.
    # It also might be a good idea to save the results somewhere to avoid rerunning the experiments everytime.
    # max_pos_tvalue_idx = t_mass_pos.index(max(t_mass_pos)) - 1
    # # # # print("max_pos_tvalue_idx: ", max_pos_tvalue_idx)
    # # # # print("clusters_pos: ", clusters_pos)
    # # # print("adj_clusters_pos: ", adj_clusters_pos)
    # x1 = adj_clusters_pos[max_pos_tvalue_idx][0] + 50
    # x2 = adj_clusters_pos[max_pos_tvalue_idx][-1] + 50

    # Running the following line will fail if h1 and h0 are not initialized.
    # h0 needs to be initialized for many permutation iterations.
    # reject, pvals_corrected, p_vals_idx_sort = significance(h1, h0)
    # dot_idxs = p_vals_idx_sort[reject]
    # x_dots = x_graph[dot_idxs]

    # Make sure to change this index to which you wanna retrieve.
    # pink_clusters = [list(i) for i in mit.consecutive_groups(sorted(x_graph[p_vals_idx_sort[:27]] / 10))]

    # y_dots = [0.35] * len(x_dots)

    plt.clf()
    # plt.rcParams["figure.figsize"] = (20, 15)

    plt.plot(x_graph, y_graph, 'k-', label=f'{age_group}_{embeddings}')

    # , markersize=10)
    ylim = [0.3, 0.8]
    plt.ylim(ylim[0], ylim[1])
    # for cluster in clusters[1:]:
    #     x1 = cluster[0] * 10
    #     x2 = cluster[-1] * 10
    #     plt.fill_betweenx(y=ylim, x1=x1, x2=x2, color="#f5e6fc")
    # plt.fill_betweenx(y=ylim, x1=x1, x2=x2, color="#f5e6fc")
    plt.axhline(0.5, linestyle='--', color='#696969')
    plt.axvline(0, linestyle='--', color='#696969')
    plt.fill_between(x_graph, y_graph - error, y_graph + error, color='#58c0fc')
    # plt.scatter(x_dots, y_dots, marker='.')
    plt.title(f"{age_group} {embeddings} decoding")
    plt.xlabel("Time (ms)")
    plt.ylabel("2v2 Accuracy")
    plt.xticks(np.arange(-200, 1001, 200), ['-200', '0', '200', '400', '600', '800', '1000'])
    plt.legend(loc=1)
    acc_at_zero = y_graph[np.where(x_graph == 0)[0][0]]
    plt.text(700, 0.35, str(f"Acc At 0ms: {acc_at_zero}"))
    plt.savefig(f"{age_group}_{embeddings}_decoding.png")
    

def shuffle_labels(y_train, y_test):
    train_len = len(y_train)
    test_len = len(y_test)

    all_labels = np.concatenate((y_train, y_test), axis=0)
    np.random.shuffle(all_labels)
    random.shuffle(all_labels)

    y_train_shuffled = all_labels[:train_len]
    y_test_shuffled = all_labels[test_len:]

    return y_train_shuffled, y_test_shuffled


def average_data_on_labels(X, y):
    """
    Note: X contains a column called labels. Will use this average those columns together.
    X and y are correctly matched with the EEG data and the labels.
    """
    x_array = []
    y_array = []
    y_set = set(y)
    y_set_list = list(y_set)
    for label in y_set_list:
        x_array.append(X[X.label == label].iloc[:, :-2].values.mean(axis=0))
        y_array.append(label)
    return np.array(x_array), y_array


def cross_validaton(num_iterations, num_win, num_folds, X, y):
    results = []
    # rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_iterations)
    rkf = ShuffleSplit(n_splits=num_folds, test_size=.20)
    scoreMean = []
    stdev = []

    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    for i in range(len(num_win)):
        temp_scoreMean = []
        temp_stdev = []
        temp_results = {}
        for j in range(num_win[i]):
            X_temp = X[i][j]
            y_temp = y[i][j]

            for train_index, test_index in rkf.split(X_temp):
                X_train, X_test = X_temp.iloc[train_index, :], X_temp.iloc[test_index, :]
                y_train, y_test = y_temp[train_index], y_temp[test_index]

                y_train_labels = get_w2v_embeds_from_dict(y_train)
                y_test_labels = get_w2v_embeds_from_dict(y_test)

                X_train = X_train.drop(labels=['label', 'participant'], axis=1)  # Reshape X_train by dropping labels.
                # Now average the test data here -> X_test and y_test.

                X_test, y_test = average_data_on_labels(X_test, y_test)

                # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
                model = Ridge()
                clf = GridSearchCV(model, ridge_params, scoring='neg_mean_squared_error', n_jobs=6, cv=5)
                clf.fit(X_train, y_train_labels)

                y_pred = clf.predict(X_test)
                points, total_points, testScore, gcf, grid, diff, both_diff = extended_2v2(y_test_labels, y_pred)

                if j in temp_results.keys():
                    temp_results[j] += [testScore]
                else:
                    temp_results[j] = [testScore]

            temp_scoreMean.append(round(np.mean(temp_results[j]), 2))
            # temp_stdev.append(round(np.std(temp_results[j]), 2))
            temp_stdev.append(round(stats.sem(temp_results[j]), 2))
        results.append(temp_results)
        scoreMean.append(temp_scoreMean)
        stdev.append(temp_stdev)

    for i in range(len(num_win)):
        for j in range(num_win[i]):
            assert len(results[i][j]) == 1 * num_folds

    return results


def cross_validaton_averaging(X_train, X_test, y_train, y_test, useRandomizedLabel):
    results = []

    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)
            model = Ridge()
            y_train_labels = get_w2v_embeds_from_dict(y_train[i][j])
            y_test_labels = get_w2v_embeds_from_dict(y_test[i][j])

            # if useRandomizedLabel:
            #     print("Randomized Labels")
            #     y_train_labels, y_test_labels = shuffle_labels(y_train_labels, y_test_labels)

            model.fit(X_train[i][j], y_train_labels)
            y_pred = model.predict(X_test[i][j])
            points, total_points, testScore, gcf, grid, diff, both_diff = extended_2v2(y_test_labels, y_pred)

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results


def calculate_residual(true_vecs, pred_vecs):
    # Note: The arugments contain many arrays.
    return true_vecs - pred_vecs

def cv_all_ph_concat_padded_residual_mod(child=False):
    scoring = 'neg_mean_squared_error'
    r2_values = []


    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    # This is because all the word embeddings are the same for each window.
    i = 0
    j = 0

    if child == False:
        # y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
        print("child is false")
        y_test_w2v = get_w2v_embeds_from_dict(labels_mapping.keys())
    else:
        # y_train_w2v = get_tuned_cbt_childes_w2v_embeds(y_train[i][j])
        print("Fine-tuned")
        y_test_w2v = get_tuned_cbt_childes_w2v_embeds(labels_mapping.keys())

    # x_train_ph = get_all_ph_concat_embeds(y_train[i][j])
    x_test_ph = get_all_ph_concat_embeds(labels_mapping.keys())

    # Implement LOOCV
    loo = LeaveOneOut()
    # for train_idx, test_idx in loo.split(x_test_ph):
    #     print(train_idx)
    #     print(test_idx)
        # X_train, X_test = x_test_ph[train_idx], x_test_ph[test_idx]
        # y_train, y_test = y_test_w2v[train_idx], y_test_w2v[test_idx]

    model = Ridge()

    clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=loo.split(x_test_ph))
    clf.fit(x_test_ph, y_test_w2v)

    y_pred_w2v_test = clf.predict(x_test_ph)  # Get the prediction w2v embeddings from the phonemes.
    r2 = clf.best_estimator_.score(x_test_ph, y_test_w2v)

    # y_pred_w2v_train = clf.predict(x_train_ph)
    # best_model = Ridge(alpha=clf.best_params_['alpha'])
    # best_model.fit(x_test_ph, y_test_w2v)
    # r2 = best_model.score(x_test_ph, y_test_w2v)
    # clf_glove = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)
    # clf_glove.fit(x_train_ph, y_train_glove)
    # y_pred_glove_test = clf_glove.predict(x_test_ph)  # Get the prediction w2v embeddings from the phonemes.
    # y_pred_glove_train = clf_glove.predict(x_train_ph)

    # clf_cbt = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)
    # clf_cbt.fit(x_train_ph, y_train_cbt)
    # y_pred_cbt_test = clf_cbt.predict(x_test_ph)
    # y_pred_cbt_train = clf_cbt.predict(x_train_ph)

    # Now we calculate residual for training and test data.
    # w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
    w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)
    return w2v_test_res, r2

def cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test, child=False):
    # First calculate the residuals. Train on ph, test on w2v, then get w2v residuals.
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'


    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    # This is because all the word embeddings are the same for each window.
    i = 0
    j = 0
    # print(y_train[i][j])
    if child==False:
        y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
        y_test_w2v = get_w2v_embeds_from_dict(y_test[i][j])
    else:
        y_train_w2v = get_tuned_cbt_childes_w2v_embeds(y_train[i][j])
        y_test_w2v = get_tuned_cbt_childes_w2v_embeds(y_test[i][j])

    # Get glove embeddings here.
    # y_train_glove = get_glove_embeds(y_train[i][j])
    # y_test_glove = get_glove_embeds(y_test[i][j])


    x_train_ph = get_all_ph_concat_embeds(y_train[i][j])
    x_test_ph = get_all_ph_concat_embeds(y_test[i][j])
    model = Ridge()

    clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=5)
    clf.fit(x_train_ph, y_train_w2v)
    y_pred_w2v_test = clf.predict(x_test_ph)  # Get the prediction w2v embeddings from the phonemes.
    y_pred_w2v_train = clf.predict(x_train_ph)

    r2_train = clf.best_estimator_.score(x_train_ph, y_train_w2v)
    r2_test = clf.best_estimator_.score(x_test_ph, y_test_w2v)

    # clf_glove = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)
    # clf_glove.fit(x_train_ph, y_train_glove)
    # y_pred_glove_test = clf_glove.predict(x_test_ph)  # Get the prediction w2v embeddings from the phonemes.
    # y_pred_glove_train = clf_glove.predict(x_train_ph)

    # clf_cbt = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)
    # clf_cbt.fit(x_train_ph, y_train_cbt)
    # y_pred_cbt_test = clf_cbt.predict(x_test_ph)
    # y_pred_cbt_train = clf_cbt.predict(x_train_ph)




    # Now we calculate residual for training and test data.
    w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
    w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)

    # glove_train_res = calculate_residual(y_train_glove, y_pred_glove_train)
    # glove_test_res = calculate_residual(y_test_glove, y_pred_glove_test)

    # cbt_train_res = calculate_residual(y_train_cbt, y_pred_cbt_train)
    # cbt_test_res = calculate_residual(y_test_cbt, y_pred_cbt_test)

    # return w2v_test_res#, cbt_test_res  # This can be changed according to what you want (for w2v or for cbt).
    return w2v_train_res, w2v_test_res, r2_train, r2_test
    # return cbt_train_res, cbt_test_res
    # return glove_train_res, glove_test_res

def cv_residual_w2v_ph_eeg_mod(X_train, X_test, y_train, y_test, residual_vecs):
    # First calculate the residuals. Train on ph, test on w2v, then get w2v residuals.
    results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    all_word_pairs_2v2 = {}

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    #
    # w2v_train_res, w2v_test_res = cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test, child=child)
    w2v_test_res = residual_vecs # Because it's only 16 words.


    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):

            # this is for predicting the second phoneme only (sim_agg.csv).
            # First remove the data for which the second phoneme is not present.
            # NOTE: The remove data function is not being used because phoneme alternatives are now being used.
            # X_train[i][j], y_train[i][j] = remove_data(X_train[i][j], y_train[i][j])
            # X_test[i][j], y_test[i][j] = remove_data(X_test[i][j], y_test[i][j])

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)

            # y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
            # y_test_w2v = get_w2v_embeds_from_dict(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            # x_train_ph = get_sim_agg_first_embeds(y_train[i][j])
            # x_test_ph = get_sim_agg_first_embeds(y_test[i][j])
            # which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2

            # w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
            # w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)

            # Now we train on EEG to predict the residuals from Word2Vec embeddings which were predicted from the phoneme embeddings.
            w2v_train_res = [residual_vecs[int(stim)] for stim in y_train[i][j]]
            model_res = Ridge()

            clf_res = GridSearchCV(model_res, ridge_params, scoring=scoring, n_jobs=-1, cv=5)

            clf_res.fit(X_train[i][j], w2v_train_res)

            y_pred_w2v_res = clf_res.predict(X_test[i][j])

            points, total_points, testScore, gcf, grid, word_pairs, diff, both_diff = extended_2v2(w2v_test_res, y_pred_w2v_res)
            # points, total_points, testScore, gcf, grid = w2v_across_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid= w2v_within_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # testScore = accuracy_score(y_test_labels, y_pred)

            tgm_matrix_temp[j, j] = testScore

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results, tgm_matrix_temp, all_word_pairs_2v2

def cv_residual_w2v_ph_eeg(X_train, X_test, y_train, y_test, child=False):

    # First calculate the residuals. Train on ph, test on w2v, then get w2v residuals.
    results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    all_word_pairs_2v2 = {}

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    w2v_train_res, w2v_test_res, r2_train, r2_test = cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test, child=child)



    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):

            # this is for predicting the second phoneme only (sim_agg.csv).
            # First remove the data for which the second phoneme is not present.
            # NOTE: The remove data function is not being used because phoneme alternatives are now being used.
            # X_train[i][j], y_train[i][j] = remove_data(X_train[i][j], y_train[i][j])
            # X_test[i][j], y_test[i][j] = remove_data(X_test[i][j], y_test[i][j])

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)

            # y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
            # y_test_w2v = get_w2v_embeds_from_dict(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            # x_train_ph = get_sim_agg_first_embeds(y_train[i][j])
            # x_test_ph = get_sim_agg_first_embeds(y_test[i][j])
            # which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2
            
            # w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
            # w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)

            # Now we train on EEG to predict the residuals from Word2Vec embeddings which were predicted from the phoneme embeddings.
            model_res = Ridge()

            clf_res = GridSearchCV(model_res, ridge_params, scoring=scoring, n_jobs=-1, cv=5)

            clf_res.fit(X_train[i][j], w2v_train_res)

            y_pred_w2v_res = clf_res.predict(X_test[i][j])



            points, total_points, testScore, gcf, grid, word_pairs, diff, both_diff = extended_2v2(w2v_test_res, y_pred_w2v_res)
            # points, total_points, testScore, gcf, grid = w2v_across_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid= w2v_within_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # testScore = accuracy_score(y_test_labels, y_pred)

            tgm_matrix_temp[j, j] = testScore

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results, tgm_matrix_temp, all_word_pairs_2v2, r2_train, r2_test


def monte_carlo_animacy_from_vectors(y_vectors):
    y_embed_labels = [i for i in range(0, 16)]
    scoring = 'neg_mean_squared_error'
    lr_params = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    x = y_vectors
    y = np.array([0 if t < 8 else 1 for t in y_embed_labels])
    # perm_accs = []
    # for i in range(100):
    #     print(i)
    #     np.random.shuffle(y)
    sf = ShuffleSplit(50, test_size=0.20)
    accs = []
    for train_idx, test_idx in sf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression()
        cv = GridSearchCV(model, param_grid=lr_params, scoring=scoring, cv=5, n_jobs=-1)
        cv.fit(x_train, y_train)
        preds = cv.predict(x_test)

        # Now compare the preds and true_values
        acc = (preds == y_test).sum() / len(y_test)
        accs.append(acc)
        # perm_accs.append(np.mean(acc))

    # print("Accuracy: ", np.mean(perm_accs))
    return np.mean(accs)


def compute_ncsnr(
        betas: np.ndarray,
        stimulus_ids: np.ndarray,
):
    """
    Computes the noise ceiling signal to noise ratio.

    :param betas: Array of betas or other neural data with shape (num_betas, num_voxels)
    :param stimulus_ids: Array that specifies the stimulus that betas correspond to, shape (num_betas)
    :return: Array of noise ceiling snr values with shape (num_voxels)
    """

    unique_ids = np.unique(stimulus_ids)  # For animacy, this corresponds to the samples that are animate and inanimate.
    # print("unique ids: ", unique_ids)


    betas_var = []
    for i in unique_ids:
        stimulus_betas = betas[stimulus_ids == i]
        betas_var.append(stimulus_betas.var(axis=0, ddof=1))  # I believe axis=0 calculates the variance for each feature.
    betas_var_mean = np.nanmean(np.stack(betas_var), axis=0)

    std_noise = np.sqrt(betas_var_mean)

    std_signal = 1. - betas_var_mean
    # print("Std signal: ", std_signal)
    std_signal[std_signal < 0.] = 0.
    std_signal = np.sqrt(std_signal)
    ncsnr = std_signal / std_noise

    return ncsnr


def compute_nc(ncsnr: np.ndarray, num_averages: int = 1):
    """
    Convert the noise ceiling snr to the actual noise ceiling estimate

    :param ncsnr: Array of noise ceiling snr values with shape (num_voxels)
    :param num_averages: Set to the number of repetitions that will be averaged together
        If there are repetitions that won't be averaged, then leave this as 1
    :return: Array of noise ceiling values with shape (num_voxels)
    """
    ncsnr_squared = ncsnr ** 2
    nc = 100. * ncsnr_squared / (ncsnr_squared + (1. / num_averages))
    return nc

def cv_animacy(X_train, y_train, X_test, y_test, do_eeg_pca=False, do_sliding_window=False):
    print("Inside cv animacy")
    results = []
    
    scoring = 'accuracy'
    

    all_y_train_labels = []
    all_y_test_labels = []

    ## Define the hyperparameters.
    logreg_params = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    xgboost_classifier_params = {'n_estimators': [500]}
    train_data_features = []
    test_data_features = []
    temp_results = {}
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            # print("Window: ", j)
            y_train_labels_w2v = y_train[i][j]
            y_test_labels_w2v = y_test[i][j]
            all_y_train_labels.append(y_train_labels_w2v)
            all_y_test_labels.append(y_test_labels_w2v)

            # model = xgb.XGBClassifier(silent=True)
            # clf = GridSearchCV(model, xgboost_classifier_params, scoring=scoring, n_jobs=-1, cv=5)
            # print("Doing XGBoost Classification")
            if do_eeg_pca:
                model = LogisticRegression()
                clf = GridSearchCV(model, logreg_params, scoring=scoring, n_jobs=-1, cv=5)
                # Do standard scaling here.
                scaler = StandardScaler()
                x_data = scaler.fit_transform(X_train[i][j].values)
                x_test_data = scaler.transform(X_test[i][j].values)
                pca = PCA(n_components=0.95)  # Using 95% of variance explained.
                x_data = pca.fit_transform(x_data)
                x_test_data = pca.transform(x_test_data)
                clf.fit(x_data, y_train_labels_w2v)
                y_pred = clf.predict(x_test_data)
            else:
                # Select the features here using noise ceiling.
                # First calculate the noise ceiling on the training data.
                betas = X_train[i][j].values
                betas = (betas - betas.mean(axis=0)) / (betas.std(axis=0) + 1e-7)
                # print("Betas: ", betas)
                # test_betas = X_test[i][j].values
                # test_betas = (test_betas - test_betas.mean(axis=0)) / (test_betas.std(axis=0) + 1e-7)

                # test_eeg = test_betas
                
                train_stims = y_train_labels_w2v
                # print("Train_ stims: ", train_stims)
                # test_stims = y_test_labels_w2v
                train_eeg = betas
                # Compute the signal to noise ratio and noise ceiling for the train data.
                train_ncsnr = compute_ncsnr(train_eeg, train_stims)
                # print("train_ncsnr: ", train_ncsnr)s
                train_nc = compute_nc(train_ncsnr, num_averages=1)
                # print("Train NC: ", train_nc)
                # print("Train NC len: ", len(train_nc))
                

                # test_ncsnr = compute_ncsnr(test_eeg, test_stims)
                # test_nc = compute_nc(test_ncsnr, num_averages=1)
                

                # Top nth percentile.
                # thresh = np.percentile(train_nc, 90)
                thresh = 0.10  # Instead of doing this just select the top 10 sensors.
                
                # Select the top 10 features.
                top_sensors = sorted(range(len(train_nc)), key=lambda i: train_nc[i])[-10:]
                # print("Calculated the top 10 sensors: ", top_sensors)

                # train_feature_mask = train_nc >= thresh
                train_feature_mask = top_sensors

                # print("test_feature_mask: ", sum(test_feature_mask))

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train[i][j].values)
                X_test_scaled = scaler.transform(X_test[i][j].values)

                X_train_scaled_features = X_train_scaled[:, train_feature_mask]
                X_test_scaled_features = X_test_scaled[:, train_feature_mask]  
                train_data_features.append(X_train_scaled_features)
                test_data_features.append(X_test_scaled_features)

    # print("Train data features: ", train_data_features)
    # print("Test data features: ", test_data_features)

    # assert len(train_data_features) == 1200
    # assert len(test_data_features) == 1200
    window_size = 100
    sliding_step_size = 10
    
    sliding_window_train_data = []
    sliding_window_test_data = []
    if do_sliding_window:
        for i in range(0, len(train_data_features)-window_size + sliding_step_size, sliding_step_size):
        
            # Do the sliding window procedure here because the noise ceiling is being calculated
            # for each time point but for a set of sixty sensors only
            sliding_window_train_data.append(np.hstack(train_data_features[i:i+window_size]))
            sliding_window_test_data.append(np.hstack(test_data_features[i:i+window_size]))
        assert len(sliding_window_train_data) == len(sliding_window_test_data)

    

    for i in range(len(sliding_window_train_data)):
        # print("Window: ", i)
        train_data = sliding_window_train_data[i]
        test_data = sliding_window_test_data[i]
        y_train_labels = all_y_train_labels[i]
        y_test_labels = all_y_test_labels[i]
        model = LogisticRegression(max_iter=4000)
        clf = GridSearchCV(model, logreg_params, scoring=scoring, n_jobs=-1, cv=5)
        clf.fit(train_data, y_train_labels)
        y_pred = clf.predict(test_data)

        testScore = accuracy_score(y_test_labels, y_pred)
        
        if i in temp_results.keys():
            temp_results[i] += [testScore]
        else:
            temp_results[i] = [testScore]

    results.append(temp_results)
    return results
        


def cross_validaton_nested(X_train, y_train, X_test, y_test, animacy=False, iteration=-1, embeddings='w2v'):
    print("Calling cross_validaton_nested")
    results = []
    preds = []
    animacy_results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    best_alphas = []
    all_word_pairs_2v2 = {}

    for i in range(len(X_train)):
        temp_results = {}
        temp_preds = {}
        temp_animacy_results = {}
        for j in range(len(X_train[i])):

            # this is for predicting the second phoneme only (sim_agg.csv).
            # First remove the data for which the second phoneme is not present.
            # NOTE: The remove data function is not being used because phoneme alternatives are now being used.
            # X_train[i][j], y_train[i][j] = remove_data(X_train[i][j], y_train[i][j])
            # X_test[i][j], y_test[i][j] = remove_data(X_test[i][j], y_test[i][j])

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)

            # Get pretrained word2vec google news embeddings.
            if animacy:
                y_train_labels_w2v = y_train[i][j]
                y_test_labels_w2v = y_test[i][j]

                # print("Y train labels w2v: ", y_train_labels_w2v)
            else:
                if embeddings == 'w2v':
                    y_train_labels_w2v = get_w2v_embeds_from_dict(y_train[i][j])
                    y_test_labels_w2v = get_w2v_embeds_from_dict(y_test[i][j])
                elif embeddings == 'phoneme':
                    y_train_labels_w2v = get_all_ph_concat_embeds(y_train[i][j])
                    y_test_labels_w2v = get_all_ph_concat_embeds(y_test[i][j])
                elif embeddings == 'fine_tuned':
                    y_train_labels_w2v = get_tuned_cbt_childes_w2v_embeds(y_train[i][j])
                    y_test_labels_w2v = get_tuned_cbt_childes_w2v_embeds(y_test[i][j])

            
            # print("y_train_labels_w2v: ", y_train_labels_w2v)
            # print("y_test_labels_w2v: ", y_test_labels_w2v)

            # one_count = sum(y_train_labels_w2v)
            # zero_count = len(y_train_labels_w2v) - one_count

            # print(f"label count: one={one_count}, zero={zero_count}")

            

            # Get padded phoneme embeddings (Predict phoneme vectors from EEG).
            # y_train_labels_w2v = get_all_ph_concat_embeds(y_train[i][j])
            # y_test_labels_w2v = get_all_ph_concat_embeds(y_test[i][j])
            # print("Phonemes shape: ", y_train_labels_w2v.shape)
            

            # Get tuned w2v vectors on CBT and Childes. 1000 epochs during the fine-tuning process.
            # y_train_labels_w2v = get_tuned_cbt_childes_w2v_embeds(y_train[i][j])
            # y_test_labels_w2v = get_tuned_cbt_childes_w2v_embeds(y_test[i][j])


            # Get residual pretrained w2v embeddings.
            # print('Getting pretrained residuals')
            # y_train_labels_w2v = get_residual_pretrained_w2v(y_train[i][j])
            # y_test_labels_w2v = get_residual_pretrained_w2v(y_test[i][j])


            # Get residual tuned w2v embeddings.
            # print('Getting tuned residuals')
            # y_train_labels_w2v = get_residual_tuned_w2v(y_train[i][j])
            # y_test_labels_w2v = get_residual_tuned_w2v(y_test[i][j])

            # The following functions retrieve the embeddings of size 16x16 (SVD/PCA).
            # y_train_labels_w2v = get_reduced_w2v_embeds(y_train[i][j], type='svd')
            # y_test_labels_w2v = get_reduced_w2v_embeds(y_test[i][j], type='svd')

            # Get glove embeddings here - 300 dimensions.
            # y_train_labels_w2v = get_glove_embeds(y_train[i][j])
            # y_test_labels_w2v = get_glove_embeds(y_test[i][j])


            # Get glove embeddings here - 200 dimensions.
            # y_train_labels_w2v = get_glove_embeds_200(y_train[i][j])
            # y_test_labels_w2v = get_glove_embeds_200(y_test[i][j])

            # # Get glove embeddings here - 100 dimensions.
            # y_train_labels_w2v = get_glove_embeds_100(y_train[i][j])
            # y_test_labels_w2v = get_glove_embeds_100(y_test[i][j])

            # # Get glove embeddings here - 50 dimensions.
            # y_train_labels_w2v = get_glove_embeds_50(y_train[i][j])
            # y_test_labels_w2v = get_glove_embeds_50(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            # y_train_labels_ph = get_sim_agg_first_embeds(y_train[i][j])
            # y_test_labels_ph = get_sim_agg_first_embeds(y_test[i][j])
            # which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels_ph = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels_ph = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2

            # Get fourier transform of audio amplitudes here.
            # y_train_labels_audio = get_audio_amplitude(y_train[i][j])
            # y_test_labels_audio = get_audio_amplitude(y_test[i][j])

            # get stft of audio applitudes here.
            # y_train_labels_audio_stft = get_stft_of_amp(y_train[i][j])
            # y_test_labels_audio_stft = get_stft_of_amp(y_test[i][j])

            # model = LogisticRegression(multi_class='multinomial')

            model = Ridge()
            if animacy:
                # model = LogisticRegression(max_iter=1000)
                ridge_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                model = LogisticRegression()
                # ridge_params = {'min_samples_split': [2,3,4]}
                scoring = "accuracy"
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[i][j].values)
            X_test_scaled = scaler.transform(X_test[i][j].values)
            

            clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=5)
            # clf.fit(X_train[i][j].values, y_train_labels_w2v)
            clf.fit(X_train_scaled, y_train_labels_w2v)
            # best_alphas.append(clf.best_params_)
            # y_pred = clf.predict(X_test[i][j].values)
            y_pred = clf.predict(X_test_scaled)
            preds.append(y_pred)
            # print("y_pred inside cross_validaton_nested: ", y_pred)

            

            # Many scoring functions.

            # points, total_points, testScore, gcf, grid, word_pairs, diff, both_diff, neg_diff = extended_2v2(y_test_labels_w2v, y_pred)
            if animacy:
                testScore = accuracy_score(y_test_labels_w2v, y_pred)
            else:
                points, total_points, testScore, gcf, grid, word_pairs, diff, both_diff, neg_diff, equal_count = extended_2v2_mod(y_test_labels_w2v, y_pred)
            # points, total_points, testScore, gcf, grid = w2v_across_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid= w2v_within_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # Across and within for phonemes
            # points, total_points, testScore, gcf, grid = ph_across_animacy_2v2(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)
            # points, total_points, testScore, gcf, grid = ph_within_animacy_2v2(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)


            # Using word embeddings to predict animacy.
            # animacy_score = monte_carlo_animacy_from_vectors(y_pred)

            # all_word_pairs_2v2[j] = word_pairs  # Here 'j' is each window across the whole timeline.

            # testScore = accuracy_score(y_test_labels, y_pred)

            # tgm_matrix_temp[j, j] = testScore

            # if j in temp_preds.keys():
            #     temp_preds[j] += [y_pred]
            # else:
            #     temp_preds[j] = [y_pred]

            # if j in temp_animacy_results.keys():
            #     temp_animacy_results[j] += [animacy_score]
            # else:
            #     temp_animacy_results[j] = [animacy_score]


            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)
        # Save the predictions to disk.
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/predictions/12m_pre_w2v_iteration_{iteration}.npz", preds)
        # preds.append(temp_preds)
        # animacy_results.append(temp_animacy_results)
    # print(best_alphas)

    return results, animacy_results, preds, tgm_matrix_temp, all_word_pairs_2v2



def cross_validaton_nested_concat(X_train, y_train, X_test, y_test):
    results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    #
    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):


            # this is for predicting the second phoneme only (sim_agg.csv).
            # First remove the data for which the second phoneme is not present.
            # NOTE: The remove data function is not being used because phoneme alternatives are now being used.
            # X_train[i][j], y_train[i][j] = remove_data(X_train[i][j], y_train[i][j])
            # X_test[i][j], y_test[i][j] = remove_data(X_test[i][j], y_test[i][j])

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)

            y_train_labels_w2v = get_w2v_embeds_from_dict(y_train[i][j])
            y_test_labels_w2v = get_w2v_embeds_from_dict(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            # y_train_labels_ph = get_sim_agg_first_embeds(y_train[i][j])
            # y_test_labels_ph = get_sim_agg_first_embeds(y_test[i][j])
            # which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels_ph = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels_ph = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2


            # model = LogisticRegression(multi_class='multinomial')

            # If concat == True -> Concat the w2v and phoneme embeddings.

            y_train_concat_w2v_ph = np.concatenate((y_train_labels_w2v, y_train_labels_ph), axis=1)
            y_test_concat_w2v_ph = np.concatenate((y_test_labels_w2v, y_test_labels_ph), axis=1)


            model = Ridge()

            clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)

            # clf.fit(X_train[i][j], y_train_concat_w2v_ph)
            # y_pred = clf.predict(X_test[i][j])


            # For predicting EEG from concatenation of w2v vectors and phoneme embeddings.
            svd = TruncatedSVD(n_components=1000)  # Using 1000 components for now.
            X_train_reduced = svd.fit_transform(X_train[i][j])
            X_test_reduced = svd.transform(X_test[i][j])
            clf.fit(y_train_concat_w2v_ph, X_train_reduced)
            x_pred = clf.predict(y_test_concat_w2v_ph)
            points, total_points, testScore, gcf, grid, diff, both_diff = extended_2v2(X_test_reduced, x_pred)




            # points, total_points, testScore, gcf, grid = extended_2v2(y_test_concat_w2v_ph, y_pred)
            # points, total_points, testScore, gcf, grid = w2v_across_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid= w2v_within_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # Across and within for phonemes
            # points, total_points, testScore, gcf, grid = ph_across_animacy_2v2(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)
            # points, total_points, testScore, gcf, grid = ph_within_animacy_2v2(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)


            # testScore = accuracy_score(y_test_labels, y_pred)

            tgm_matrix_temp[j, j] = testScore

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results, tgm_matrix_temp


def cross_validaton_tgm(X_train, y_train, X_test, y_test, child=False, res=False, target_pca=False):
    # results = []
    print("Inside cross-validation tgm")
    tgm_matrix_temp = np.zeros((120, 120))
    diff_tgm = np.zeros((120, 120))
    both_diff_tgm = np.zeros((120, 120))
    neg_diff_tgm = np.zeros((120, 120))
    equal_count_matrix = np.zeros((120, 120))
    roe_matrix = np.zeros((120, 120))

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    

    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):
            
            if res == False:
                y_train_labels = get_w2v_embeds_from_dict(y_train[i][j])
            else:
                y_train_labels, y_test_labels = cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test, child=child)
            
            if target_pca:
                print("Initiate target PCA")
                target_scaler = StandardScaler()
                y_train_labels = target_scaler.fit_transform(y_train_labels)
                pca = PCA(n_components = 0.95, random_state=42)
                y_train_labels = pca.fit_transform(y_train_labels)
            
                
            model = Ridge()
            clf = GridSearchCV(model, ridge_params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[i][j].values)
            # clf.fit(X_train[i][j].values, y_train_labels)
            clf.fit(X_train_scaled, y_train_labels)
            equal_count_list = []

            for k in range(len(X_test[i])):
                if res==False:
                    y_test_labels = get_w2v_embeds_from_dict(y_test[i][k])
                    if target_pca:
                        y_test_labels = target_scaler.transform(y_test_labels)
                        y_test_labels = pca.transform(y_test_labels)
                X_test_scaled = scaler.transform(X_test[i][k].values)
                # y_pred = clf.predict(X_test[i][k].values)
                y_pred = clf.predict(X_test_scaled)

                # roe_mean, feature_corrs = corr_score(y_test_labels, y_pred)
                # roe_matrix[j, k] = roe_mean

                # points, total_points, testScore, gcf, grid, word_pairs, diff, both_diff, neg_diff, equal_count = extended_2v2(y_test_labels, y_pred)

                points, total_points, testScore, gcf, grid, word_pairs, diff, both_diff, neg_diff, equal_count = extended_2v2_mod(y_test_labels, y_pred)

                equal_count_matrix[j, k] = equal_count

                tgm_matrix_temp[j, k] = testScore
                diff_tgm[j, k] = diff
                both_diff_tgm[j, k] = both_diff
                neg_diff_tgm[j,k] = neg_diff



            #     if k in temp_results.keys():
            #         temp_results[j] += [testScore]
            #     else:
            #         temp_results[j] = [testScore]
            # if j in temp_results.keys():
            #     temp_results[j] += [testScore]
            # else:
            #     temp_results[j] = [testScore]
            #
            # results.append(temp_results)

    return tgm_matrix_temp, diff_tgm, both_diff_tgm, neg_diff_tgm, equal_count_matrix, roe_matrix


def t_test(results, num_win, num_folds):
    pvalues_pos = []
    pvalues_neg = []
    tvalues_pos = []
    tvalues_neg = []
    for i in range(len(results)):
        for j in range(num_win[i]):
            # change the second argument below for comparison
            istat = stats.ttest_1samp(results[i][j], .5)
            pvalues_pos += [istat.pvalue] if istat.statistic > 0 else [1]
            pvalues_neg += [istat.pvalue] if istat.statistic < 0 else [1]
            # removed just so that we can get the negative value from the pre window
            tvalues_pos += [istat.statistic] if istat.statistic > 0 else [0]
            tvalues_neg += [istat.statistic] if istat.statistic < 0 else [0]
    print("Positive p-values: ", pvalues_pos)
    print("Negative p-values: ", pvalues_neg)
    return pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg


# Finding contiguous time cluster
def find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg):
    valid_window_pos = [i for i, v in enumerate(pvalues_pos) if v < 0.05]
    valid_window_neg = [i for i, v in enumerate(pvalues_neg) if v < 0.05]
    ## REMOVE FOR NULL FUNCTION
    print("Valid positive windows are: {0}\n".format(valid_window_pos))
    print("Valid negative windows are: {0}\n".format(valid_window_neg))

    # Obtain clusters (2 or more consecutive meaningful time) -> Initally it was 3 clusters. I changed it to 2 and now it's 3 again as of 09-04-2021.
    clusters_pos = [list(group) for group in mit.consecutive_groups(valid_window_pos)]
    clusters_pos = [group for group in clusters_pos if len(group) >= 3]

    clusters_neg = [list(group) for group in mit.consecutive_groups(valid_window_neg)]
    clusters_neg = [group for group in clusters_neg if len(group) >= 3]

    adj_clusters_pos = []
    for c in clusters_pos:
        new_list = [((x * 10) - 200) for x in c]
        adj_clusters_pos.append(new_list)

    adj_clusters_neg = []
    for c in clusters_neg:
        new_list = [((x * 10) - 200) for x in c]
        adj_clusters_neg.append(new_list)

    ## REMOVE FOR NULL FUNCTION
    print("Positive clusters are: {0}\n".format(adj_clusters_pos))
    print("Negative clusters are: {0}\n".format(adj_clusters_neg))

    return adj_clusters_pos, adj_clusters_neg, clusters_pos, clusters_neg


def get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg):
    t_mass_pos = [0]
    for c in clusters_pos:
        t_scores_pos = 0
        for time in c:
            t_scores_pos += tvalues_pos[time]
        t_mass_pos += [t_scores_pos]

    ## REMOVE FOR NULL FUNCTION
    print("Positive tmass values are: {0}\n".format(t_mass_pos))
    max_t_mass_pos = max(t_mass_pos)

    t_mass_neg = [0]
    for c in clusters_neg:
        t_scores_neg = 0
        for time in c:
            t_scores_neg += tvalues_neg[time]
        t_mass_neg += [t_scores_neg]

    ## REMOVE FOR NULL FUNCTION
    print("Negative tmass values are: {0}\n".format(t_mass_neg))
    max_t_mass_neg = min(t_mass_neg)

    max_abs_tmass = max(max_t_mass_pos, abs(max_t_mass_neg))

    ## REMOVE FOR NULL FUNCTION
    print("The max positive t mass is: {0}\n".format(max_t_mass_pos))
    print("The max negative t mass is: {0}\n".format(max_t_mass_neg))
    print("The max absolute t mass is: {0}\n".format(max_abs_tmass))

    return max_abs_tmass, t_mass_pos, t_mass_neg




def cluster_permutation_test(results1, results2, threshold=0.05, n_permutations=10000):
    """
    Non-parametric cluster-level paired t-test.
    """
    from mne.stats import spatio_temporal_cluster_1samp_test
    from mne.stats import permutation_cluster_1samp_test
    from scipy.stats import ttest_ind
    # results1_temp = np.array(results1).reshape(-1, 1)
    #
    # results2_temp = np.array(results2).reshape(-1, 1)
    # X = results1_temp - results2_temp
    # X = X.reshape(X.shape[0], -1)

    # NOTE: results1 and results2 are the results dictionary that has all the cv fold accuracy for each window.
    results1_temp = []
    for i in range(len(results1)):
        for k,v in results1[i].items():
            results1_temp.append(v)
    results2_temp = []
    for i in range(len(results2)):
        for k,v in results2[i].items():
            results2_temp.append(v)
    results1_temp = np.array(results1_temp)
    results2_temp = np.array(results2_temp)

    X = results1_temp - results2_temp

    # X = [results1, results2]
    threshold = 0.05
    n_permutations = 10000

    # ttest_ind(results1, results2)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X, threshold=threshold, n_permutations=n_permutations)

