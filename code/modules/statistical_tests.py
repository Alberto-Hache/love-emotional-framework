import pandas as pd
import numpy as np
from scipy.stats import kstwobign, pearsonr
from numpy import random
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import f
import sys

from IPython.display import display, HTML

#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#   Functionality to support statistical tests
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def read_and_merge_surveys_1_and_2(path_1, path_2, verbose=True):
    '''
    Read, merge and validate two files with results of one test (same columns) run in two different orders.
    Generate a second version excluding tests with some invalid answer to the control questions.

    Arguments:
    'path_1'    (str) Full path to .csv file with test 1.
    'path_2'    (str) Full path to .csv file with test 2.

    Returns:
    df_12       (DataFrame) The merged tests, all responses.
    '''
    # Read survey 1 and 2.
    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)

    if verbose:
        print(f"Length of the tests 1 & 2 (all responses): \t\t{len(df_1)} + {len(df_2)} = {len(df_1) + len(df_2)} data points")
    assert set(df_1.columns) == set(df_2.columns), "The two sets should have the same columns to be compatible."

    # Concatenate the two dfs and rearrange columns (1 and 2 have videos in different order).
    df_12 = pd.concat([df_1, df_2], ignore_index=True)  # Concatenate the two sub-tests.
    df_12.columns = [c.split('\n')[0] for c in df_12.columns]  # Clean up test names (drop '\nIntroduce tu evaluación de esta componente.")

    return df_12

def generate_rates_only_df(df, test_id, n_tests, verbose=True):
    '''
    Generate a version with ONLY rater info and data points.

    'df'        (DataFrame) The test with all the columns. 
    'test_id'   (str) Indentifier of the test (e.g. "A" or "B").
    'n_tests'   (int) Number tests in the survey (e.g. 24 in 2023 Emotional attribution study).
    '''
    # Generate the names of the columns with data points and dummy IDs for the raters.
    test_columns = [f"({test_id}{n+1:02}) {d}" for n in range(n_tests) for d in ['Placer', 'Activación', 'Dominancia']]  # '(A01) Placer', '(A01) Activación', '(A01) Dominancia', '(A02) Placer'...
    rater_names = [f"Rater {test_id}{n+1:02}" for n in range(len(df))]  # 'Rater A01', 'Rater A02'...

    # Drop useless columns and reorder the remaining ones:
    df_rates = pd.DataFrame(df[test_columns])
    df_rates['Rater'] = rater_names
    df_rates = df_rates[['Rater'] + test_columns]

    # See what it looks like:
    if verbose:
        n_raters = df_rates.shape[0]
        print(f"Shape: {df_rates.shape}: {n_raters} raters x {n_tests} tests")
        df_rates.head()
    
    return df_rates

def adapt_study_df_to_test(df):
    '''
    Reformat a dataframe with the result of a test for its use with 'pingouin' statistics library.

    Input: a DataFrame with the result of a test in this format:
        Rater	    (A07) Placer	(A07) Activación	(A07) Dominancia	(A05) Placer	(A05) Activación	(A05) Dominancia
    0	Rater A01	7	            5	                4	                1	            8	                2
    1	Rater A02	6	            8	                3	                2	            8	                8

    Output:
        test	            rater	    rating
    0	(A07) Placer	    Rater A01	7
    1	(A07) Activación	Rater A01	5
    2	(A07) Dominancia	Rater A01	4
    3	(A05) Placer	    Rater A01	1
    4	(A05) Activación	Rater A01	8
    5	(A05) Dominancia	Rater A01	2
    6	(A07) Placer	    Rater A02	6
    7	(A07) Activación	Rater A02	8
    8	(A07) Dominancia	Rater A02	3
    9	(A05) Placer	    Rater A02	2
    10	(A05) Activación	Rater A02	8
    11	(A05) Dominancia	Rater A02	8

    '''
    test_names = list(df.columns[1:])  # Names of the videos rated (3 for each video): ['(A07) Placer', '(A07) Activación', '(A07) Dominancia', '(A05) Placer'...]
    n_tests = len(test_names)
    assert len(test_names)%3 == 0, "Wrong columns: the number of tests should be a multiple of 3."
    assert df.columns[0] == "Rater", "Wrong column name: the top-left column should be 'Rater'."

    rater_names = list(df['Rater'])  # Identifiers of the participants in the test.
    n_raters = len(rater_names)

    # Replicate the list of tests x number of raters.
    test_values = test_names * n_raters

    # Replicate the rater names x number of tests.
    rater_values = list(np.repeat(rater_names, n_tests))

    # Reassemble the rates in a column.
    l = [list(df.loc[n])[1:] for n in range(len(df))]
    rating_values = [item for sublist in l for item in sublist]

    new_df = pd.DataFrame({
        'test': test_values,
        'rater': rater_values,
        'rating': rating_values,
    })

    return new_df

def run_icc_test(df_rates, type=None):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for the samples in a dataframe.
    If one of the supported types is passed in 'type', the results of that case of the test
    are displayed as a table.
    """
    # Rearrange the df to a format suitable for the ICC library.
    df_test = adapt_study_df_to_test(df_rates)
    df_test.insert(
        loc=0, column='dimension',
        value=df_test.apply(lambda x: x['test'] [x['test'].find(' ')+1:], axis=1)
    )
    # df_test['dimension'] = df_test.apply(lambda x: x['test'] [x['test'].find(' ')+1:], axis=1)

    # Get slices of each dimension to test:
    df_test_pleasure = df_test[df_test['dimension'] == 'Placer']
    df_test_arousal = df_test[df_test['dimension'] == 'Activación']
    df_test_dominance = df_test[df_test['dimension'] == 'Dominancia']

    icc_pleasure = pg.intraclass_corr(
        data=df_test_pleasure,
        targets='test',
        raters='rater',
        ratings='rating',
    )
    icc_arousal = pg.intraclass_corr(
        data=df_test_arousal,
        targets='test',
        raters='rater',
        ratings='rating',
    )
    icc_dominance = pg.intraclass_corr(
        data=df_test_dominance,
        targets='test',
        raters='rater',
        ratings='rating',
    )

    if type in list(icc_pleasure['Type']):
        df1 = pd.concat([
            icc_pleasure[icc_pleasure['Type'] == type],
            icc_arousal[icc_pleasure['Type'] == type],
            icc_dominance[icc_pleasure['Type'] == type],
        ])
        df1.insert(loc=0, column='Dimension', value=['Pleasure', 'Arousal', 'Dominance'])
        display(df1.style.hide(axis='index'))

    return icc_pleasure, icc_arousal, icc_dominance, df_test
    
def my_two_sample_t2_test(sample1, rm_pad_df, rm_idx, method=None, verbose=True):
    '''
    Compare a sample of data points associated to a certain emotion with referencial
    PAD statistics reported in a study from the literature.
    A special implementation is required because we don't have all the samples from
    the second distribution, only means, st-devs, and sometimes the 'N'.

    Arguments:
    sample1         (DataFrame) Sample of many PAD values with the same emotion.
                        pleasure	 arousal	dominance
                    0	-1.00	     0.50	    -0.75
                    1	-1.00	     1.00	    -1.00
                    2	 0.00	     0.00	     0.00
                    3	-0.50	    -0.50	    -0.50
                    4	-0.50	    -0.25	    -0.75
                    ...	...	        ...	        ...
    rm_pad_df       (Dataframe) Referencial PAD values for a list of terms, e.g:
                            Number	Term	N	Pleasure-mean	Pleasure-sd	Arousal-mean	Arousal-sd	Dominance-mean	Dominance-sd
                        0	1	    Bold	27	0.44	        0.32	    0.61	        0.24	    0.66	        0.30
                        1	2	    Useful	27	0.70	        0.20	    0.44	        0.28	    0.47	        0.40
                        2	3	    Mighty	27	0.48	        0.37	    0.51	        0.28	    0.69	        0.3
                        ...
    rm_idx          (int) Index of 'rm_pad_df' to compare against.
    method          (str) What assumption to make regarding the unknown covariance
                    matrix of the referential emotion term. Values:
                    'independent'   Assume independent variables (diagonal
                                    covariance matrix with standard deviations).
                    'same'          Assume the same covariance matrix as sample1.
    '''
    # Calculate stats for Sample 1.
    n1, p = sample1.shape
    means1 = np.array(np.mean(sample1, axis=0))  # (3,)
    s1 = np.cov(sample1, rowvar=False)  # Covariance Matrix of the sample (3, 3).

    # Calculate stats for Sample 2.
    if 'N' in  rm_pad_df.columns:
        n2 = rm_pad_df.iloc[rm_idx]['N']
    else:
        n2 = n1  # Assumption in case the reference PAD distribution doesn't report 'N'.

    means2 = np.array([
        rm_pad_df.iloc[rm_idx]['Pleasure-mean'],
        rm_pad_df.iloc[rm_idx]['Arousal-mean'],
        rm_pad_df.iloc[rm_idx]['Dominance-mean'],
    ])
    # Estimate unknown covariance matrix of Sample 2.
    if not method or method == 'independent':
        # Assume independent variables.
        variances_s2 = np.array([
            rm_pad_df.iloc[rm_idx]['Pleasure-sd'],
            rm_pad_df.iloc[rm_idx]['Arousal-sd'],
            rm_pad_df.iloc[rm_idx]['Dominance-sd'],
        ]) ** 2
        s2 = np.diag(variances_s2)
    elif method == 'same':
        # Assume the same covar. matrix as Sample 1.
        s2 = s1.copy()
    else:
        print(f"Error in my_two_sample_t2_test(): Unkonwn 'method': {method}")
        sys.exit(1)

    # Run two-sample Hotelling's T-squared test.
    s = ((n1 - 1)*s1 + (n2 - 1)*s2)/(n1 + n2 - 2)

    delta_mean = np.array(means1 - means2)
    delta_mean_trans = np.array(means1 - means2).T
    s_inv = np.linalg.inv(s)

    t_square = (n1*n2)/(n1 + n2) * np.matmul(np.matmul(delta_mean_trans, s_inv), delta_mean)

    # Statistic test.
    statistic = t_square * (n1+n2-p-1)/(p*(n1+n2-2))
    F = f(p, n1+n2-p-1)
    p_value = 1 - F.cdf(statistic)

    if verbose:
        print(f"Test statistic: {statistic}\nDegrees of freedom: {p} and {n1+n2-p-1}\np-value: {p_value}")

    return statistic, F, p_value


def my_euclidean_distance(sample1,rm_pad_df, rm_idx, verbose=False):
    '''
    Compare the mean values of a sample of data points associated to a certain emotion with 
    the mean values of referencial PAD statistics reported in a study from the literature.

    Arguments:
    sample1         (DataFrame) Sample of many PAD values with the same emotion.
                        pleasure	 arousal	dominance
                    0	-1.00	     0.50	    -0.75
                    1	-1.00	     1.00	    -1.00
                    2	 0.00	     0.00	     0.00
                    3	-0.50	    -0.50	    -0.50
                    4	-0.50	    -0.25	    -0.75
                    ...	...	        ...	        ...
    rm_pad_df       (Dataframe) Referencial PAD values for a list of terms, e.g:
                            Number	Term	N	Pleasure-mean	Pleasure-sd	Arousal-mean	Arousal-sd	Dominance-mean	Dominance-sd
                        0	1	    Bold	27	0.44	        0.32	    0.61	        0.24	    0.66	        0.30
                        1	2	    Useful	27	0.70	        0.20	    0.44	        0.28	    0.47	        0.40
                        2	3	    Mighty	27	0.48	        0.37	    0.51	        0.28	    0.69	        0.3
                        ...
    rm_idx          (int) Index of 'rm_pa_df' to compare against.
    '''
    # Calculate stats for Sample 1.
    n1, p = sample1.shape
    means1 = np.array(np.mean(sample1, axis=0))  # (3,)

    # Calculate stats for Sample 2.
    if 'N' in  rm_pad_df.columns:
        n2 = rm_pad_df.iloc[rm_idx]['N']
    else:
        n2 = n1  # Assumption in case the reference PAD distribution doesn't report 'N'.

    means2 = np.array([
        rm_pad_df.iloc[rm_idx]['Pleasure-mean'],
        rm_pad_df.iloc[rm_idx]['Arousal-mean'],
        rm_pad_df.iloc[rm_idx]['Dominance-mean'],
    ])  # (3,)

    # Statistic test.
    statistic = np.sqrt(np.sum(np.square(means1 - means2)))

    return statistic


def match_sample_vs_rm_pad_values(emotion_to_match, df_3d_norm, rm_pad_df, method='independent'):
    '''
    Compare the statistical distribution of the PAD values individually rated to one specific
    emotion with referencial PAD statistics reported in a study from the literature.

    Args:
    emotion_to_match    (int) The number of the emotion whose PAD samples are compared.
    df_3d_norm          (Dataframe) Individual PAD rates from each rater to each video, e.g:
                            pleasure	arousal	dominance	emotion	video	test_ab
                        0	-1.00	    0.50	-0.75	    0	    A01 	A
                        1	-1.00	    1.00	-1.00	    0	    A02	    A
                        2	0.00	    0.00	0.00	    0	    A03	    A
                        ...
    rm_pad_df           (Dataframe) Referencial PAD values for a list of terms, e.g:
                            Number	Term	N	Pleasure-mean	Pleasure-sd	Arousal-mean	Arousal-sd	Dominance-mean	Dominance-sd
                        0	1	    Bold	27	0.44	        0.32	    0.61	        0.24	    0.66	        0.30
                        1	2	    Useful	27	0.70	        0.20	    0.44	        0.28	    0.47	        0.40
                        2	3	    Mighty	27	0.48	        0.37	    0.51	        0.28	    0.69	        0.3
                        ...
    method              (str) What assumption to make regarding the unknown covariance
                        matrix of the referential emotion term:
                        'independent'   Assume independent variables (diagonal
                                        covariance matrix with standard deviations).
                        'same'          Assume the same covariance matrix as sample1.
                        'means'         Compare means only with Euclidean distance, ignoring
                                        dispersion statistics.
    Returns:
    rm_statistics_df    (Dataframe) The referencial emotions sorted by matching (best at top) 
                        and the statistic obtained (lower is better), e.g:
                            rm_idx	rm_term	    statistic
                        0	103	    Helpless	3.698365
                        1	100	    Fearful	    5.234994
                        2	97	    Insecure	5.793690
                        ...
    '''
    # Extract all the data points associated to the emotion to match.
    sample1 = df_3d_norm[df_3d_norm['emotion'] == emotion_to_match][['pleasure', 'arousal', 'dominance']]

    # Compare the sample against each of the referential PAD statistics.
    rm_indices = []
    statistic_values = []
    rm_emotion_terms = []
    for i in range(len(rm_pad_df)):
        if method in ('independent', 'same'):
            statistic, F, p_value = my_two_sample_t2_test(
                sample1, rm_pad_df, i, method=method, verbose=False,
            )
        elif method in ('means'):
            statistic = my_euclidean_distance(
                sample1, rm_pad_df, i, verbose=False,
            )
        rm_indices += [i]
        statistic_values += [statistic]
        rm_emotion_terms.append(rm_pad_df.iloc[i]['Term'])
    
    rm_statistics_df = pd.DataFrame({
        'rm_idx': rm_indices,
        'rm_term': rm_emotion_terms,
        'statistic': statistic_values,
    })
    rm_statistics_df = rm_statistics_df.sort_values(by='statistic').copy()
    rm_statistics_df.reset_index(drop=True, inplace=True)

    return rm_statistics_df
