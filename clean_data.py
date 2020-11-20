# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.apply(lambda col: pd.to_numeric(col, errors='coerce')).dropna().drop(
        "{}".format(extra_feature), 1)
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_run = CTG_features.apply(lambda col: pd.to_numeric(col, errors='coerce')).drop("{}".format(extra_feature), 1)
    # c_run = c_run.drop(c_run.index[0])

    def removeNanWithSamples(col):
        col_without_nan = col.dropna()
        random_col = np.random.choice(col_without_nan, col.size)
        col[col.isnull()] = random_col[col.isnull()]
        return col
    c_cdf = c_run.apply(removeNanWithSamples)

    # temp = rm_ext_and_nan(CTG_features, extra_feature)
    #
    # for c in c_run.keys():
    #     col = c_run[c]
    #     col_without_nan = col.dropna()
    #     random_col = np.random.choice(col_without_nan, col.size)
    #     col[col.isnull()] = random_col[col.isnull()]
    #     c_run[c] = col
    #     temp = c_run
    #     pd.Series(np.random.choice(temp[c].dropna()) )


    #     cdf = np.cumsum(temp[c]) / np.sum(temp[c])
    #     sort_temp = temp[c].sort_values(ascending=True)
    #     num_of_nan = c_run[c].isnull().sum()
    #     rand_vec = np.random.uniform(0, 1, num_of_nan)
    #     sampeled_vec = []
    #     j = 0
    #
    #     for i in range(num_of_nan):
    #         for k in range(cdf.shape[0] - 1):
    #             if ((rand_vec[i] <= cdf.values[k + 1]) & (rand_vec[i] >= cdf.values[k])):
    #                 sampeled_vec.append(sort_temp.values[k])
    #                 break
    #
    #     for row in range(c_run.shape[0]):
    #         if (np.isnan(c_run[c].values[row])):
    #             c_run[c].values[row] = sampeled_vec[j]
    #             j = j + 1
    #             if (j == num_of_nan):
    #                 break
    #
    #     c_cdf.update({c: c_run[c]})

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for c in c_feat.keys():
        minimum = c_feat[c].min()
        quar = c_feat[c].quantile(0.25)
        med = c_feat[c].median()
        tri_quart = c_feat[c].quantile(0.75)
        maximum = c_feat[c].max()
        new = {"min": minimum, "Q1": quar, "median": med, "Q3": tri_quart, "max": maximum}
        d_summary.update({c: new})
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for c in c_feat.keys():
        iqr = d_summary[c]['Q3'] - d_summary[c]['Q1']
        upper = d_summary[c]['Q3'] + iqr * 1.5
        lower = d_summary[c]['Q1'] - iqr * 1.5
        c_feat_coll = c_feat[c]
        new = c_feat_coll[(c_feat_coll <= upper) & (c_feat_coll >= lower)]
        c_no_outlier.update({c: new})
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[feature]
    filt_feature = np.array(filt_feature[filt_feature <= thresh])
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = CTG_features.copy()
    if (mode == 'standard'):
        for i in nsd_res.keys():
            nsd_res[i] = (nsd_res[i] - nsd_res[i].median()) / nsd_res[i].std()

    if (mode == 'MinMax'):
        for i in nsd_res.keys():
            nsd_res[i] = (nsd_res[i] - nsd_res[i].min()) / (nsd_res[i].max() - nsd_res[i].min())

    if (mode == 'mean'):
        for i in nsd_res.keys():
            nsd_res[i] = (nsd_res[i] - nsd_res[i].mean()) / (nsd_res[i].max() - nsd_res[i].min())

    if (flag == True):
        import matplotlib.pyplot as plt
        plt.hist(nsd_res[x], bins=80, label=x)
        plt.hist(nsd_res[y], bins=80, label=y)
        plt.ylabel('Frequency')
        plt.xlabel('Values')
        plt.legend(loc='upper right')
        plt.show()

        # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
