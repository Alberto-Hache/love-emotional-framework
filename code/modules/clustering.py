import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from os.path import exists
import random
from math import ceil
import matplotlib

import time_series_tools as ts_tools

#------------------------------
# Clustering-related functions:
#------------------------------

def run_clustering_evaluation(encodings, clustering, histogram=False, verbose=True):
    """
    Example of output for DBSCAN:

    9 clusters found:
        0  1  4  8  2  5  6  7  3
    13944 16 16 14 14 14 13 11 10

    No. samples     Type  Silhouette score       (incl.)
        12261     core          0.078839          core
        1791  non-core          0.078210 core+non-core
        6168     noisy         -0.095471           all
    
    """
    labels_list = clustering.labels_
    n_clusters = len(set(labels_list)) - (1 if -1 in labels_list else 0)

    n_noisy_points = list(labels_list).count(-1)
    if hasattr(clustering, '.core_sample_indices_'):
        n_core_samples = len(clustering.core_sample_indices_)
    else:
        n_core_samples = len(labels_list) - n_noisy_points
    non_core_points = len(labels_list) - n_core_samples - n_noisy_points
    silh_score = silhouette_score(encodings, labels_list)

    df = pd.DataFrame(data=np.array(labels_list))
    cluster_count_series = df[0].value_counts()
    cluster_count_series_mask = cluster_count_series.index != -1
    cluster_count = pd.DataFrame(
        cluster_count_series[cluster_count_series_mask]).T
    cluster_count_list = list(cluster_count.loc[0])

    # Partial metric for VALID labels (no outliers):
    valid_samples_mask = labels_list != -1
    encodings_valid = encodings[valid_samples_mask]
    labels_valid = labels_list[valid_samples_mask]
    silh_score_valid = silhouette_score(encodings_valid, labels_valid)

    # Partial metric for CORE labels:
    if hasattr(clustering, '.core_sample_indices_'):
        core_samples_mask = np.zeros_like(labels_list, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        encodings_core = encodings[core_samples_mask]
        labels_core = labels_list[core_samples_mask]
        silh_score_core = silhouette_score(encodings_core, labels_core)
    else:
        silh_score_core = None

    # Display results:
    if verbose:
        if hasattr(clustering, '.core_sample_indices_'):
            print_df = pd.DataFrame({
                'No. samples': [n_core_samples, non_core_points, n_noisy_points],
                'Type': ['core', 'non-core', 'noisy'],
                'Silhouette score': [silh_score_core, silh_score_valid, silh_score],
                '(incl.)': ['core', 'core+non-core', 'all'],
            })
        else:
            print_df = pd.DataFrame({
                'No. samples': [n_core_samples, n_noisy_points],
                'Type': ['core', 'noisy'],
                'Silhouette score': [silh_score_valid, silh_score],
                '(incl.)': ['core', 'all'],
            })
        print(f"{n_clusters} clusters found:")
        print(cluster_count.to_string(index=False))
        print("")
        print(print_df.to_string(index=False))

    if histogram:
        plt.hist(clustering.labels_, bins=n_clusters)

    return n_clusters, n_noisy_points, non_core_points, cluster_count_list, silh_score, silh_score_valid, silh_score_core

def hyperparameter_search(from_val, to_val, step_val, encodings, algorithm, algorithm_name=None):
    if not algorithm_name:
        algorithm_name = str(algorithm.func).split('\'')[-2].split('.')[-1]
    n_clusters_list, silh_score_list, silh_score_valid_list, silh_score_core_list = [], [], [], []
    eps_list = np.arange(from_val, to_val, step_val)
    print(f"{algorithm_name} search:")
    for eps in eps_list:
        print(f"eps = {eps:.3f}...", end='')
        dbscan_clustering = algorithm(eps=eps).fit(encodings)
        n_clusters, n_noisy_points, non_core_points, cluster_count_list, silh_score, silh_score_valid, silh_score_core = \
            run_clustering_evaluation(encodings, dbscan_clustering, verbose=False)
        n_clusters_list.append(n_clusters)
        silh_score_list.append(silh_score)
        silh_score_valid_list.append(silh_score_valid)
        silh_score_core_list.append(silh_score_core)
        print(f"\reps = {eps:.3f}, silhouette score = {silh_score:.3f}, {n_clusters} clusters: {cluster_count_list}, {n_noisy_points} outliers")
    
    report_df = pd.DataFrame({
        'eps': eps_list,
        'n_clusters': n_clusters_list,
        'silh_score': silh_score_list,
        'silh_score_valid': silh_score_valid_list,
        'silh_score_core': silh_score_core_list,
    })
    print(report_df.transpose())

    return eps_list, n_clusters_list, silh_score_list, silh_score_valid_list, silh_score_core_list

def kmedoids_hyperparameter_search(hyperparams_values, encodings, algorithm, algorithm_name=None):
    if not algorithm_name:
        algorithm_name = str(algorithm.func).split('\'')[-2].split('.')[-1]
    print(f"{algorithm_name} search:")

    assert len(hyperparams_values['random_state']) == 1 or algorithm.keywords['init'] == 'random', \
        f"Error: 'random_state' values were given, but should only be used when init='random'"

    model_list, n_clusters_list, random_state_list, silh_score_list, inertia_list = [], [], [], [], []
    for n_clusters in hyperparams_values['n_clusters']:
        for random_state in hyperparams_values['random_state']:
            print(f"n_clusters = {n_clusters}, random_state = {random_state}...", end='')
            kmedoids = algorithm(n_clusters=n_clusters, random_state=random_state).fit(encodings)
            n_clusters, _, _, cluster_count_list, silh_score, _, _ = \
                run_clustering_evaluation(encodings, kmedoids, verbose=False)
            inertia = kmedoids.inertia_

            model_list.append(kmedoids)
            n_clusters_list.append(n_clusters)
            random_state_list.append(random_state)
            silh_score_list.append(silh_score)
            inertia_list.append(inertia)  # Valid for KMedoids only.
            print(f"\rn_clusters = {n_clusters}, random_state = {random_state}, silhouette score = {silh_score:.3f}, inertia = {inertia}, {n_clusters} clusters: {cluster_count_list}")

    report_df = pd.DataFrame({
        'n_clusters': n_clusters_list,
        'random_state': random_state_list,
        'silh_score': silh_score_list,
        'inertia': inertia_list,
    })
    print(report_df.transpose())

    return model_list, n_clusters_list, random_state_list, silh_score_list, inertia_list

def find_closest_point_in_latent_space(encoding, all_encodings):
    return (all_encodings - encoding).abs().sum(axis=1).idxmin()

def moving_avg_cluster_allocation(cluster_probs, n_components, window=3, renormalize=True):
    """
    Apply a moving average to the probabilities assigned to each cluster stored in the dataframe passed.
    It DOES CHANGE the original probabilities passed, both in cluster columns ('0', '1', ...) and in 'Probability'.
    The process is run separately for each trajectory (with different seed-episode).

    Arguments:
    'cluster_probs'         (DataFrame) The original clustering allocation and its
                            respective probability at each step.
    'n_components'          (int) Number of cluster columns, starting from 0.
    'window'                (int) Size of the moving average window.
    'renormalize'           (bool) Whether probabilities must be normalized after applying mov. avg.
                            on each row.

    Returns:                (DataFrame) A copy of the original DataFrame passed with updated columns
                            'Cluster' and 'Probability', reflecting the new cluster allocation and
                            its probability.

    Example of 'cluster_probs'
        0	        1	        2	        3	        4	           5	           6	Step	Seed	Episode	Cluster	Probability
    0	0.120327	0.000284	0.877684	0.001704	0.0	1.588217e-26	1.241352e-09	19	    30	    4	    2	    0.877684
    1	0.109456	0.000539	0.888497	0.001508	0.0	3.594956e-25	2.756639e-13	20	    30  	4	    2   	0.888497
    2	0.089138	0.021327	0.888326	0.001209	0.0	2.242748e-24	8.353532e-18	21	    30	    4	    2   	0.888326
    3	0.086344	0.007172	0.905223	0.001262	0.0	1.834754e-23	1.515223e-23	22	    30	    4   	2   	0.905223
    4	0.089533	0.002595	0.906501	0.001371	0.0	2.613286e-22	6.295237e-40	23	    30	    4	    2	    0.906501 
    """
    list_of_cluster_cols = list(np.arange(n_components))
    list_of_non_cluster_cols = list(set(cluster_probs.columns).difference(set(np.arange(n_components))))

    # Loop over each unique seed-episode combination.
    s_e_pairs = cluster_probs[['Seed', 'Episode']].drop_duplicates()
    list_of_subseries = []
    for i in range(len(s_e_pairs)):
        # Run a moving average over the columns with cluster numbers of the trajectory.
        seed, episode = s_e_pairs.iloc[i]['Seed'], s_e_pairs.iloc[i]['Episode']
        subseries = cluster_probs[(cluster_probs['Seed'] == seed) & (cluster_probs['Episode'] == episode)][list_of_cluster_cols].\
            rolling(window=window, min_periods=1).mean()
        list_of_subseries.append(subseries)
    # Concatenate the separate trajectories and renormalize probabilities at each step if required.
    df_probs_smooth = pd.concat(list_of_subseries, ignore_index=True)
    if renormalize:
        df_probs_smooth = df_probs_smooth.div(df_probs_smooth.sum(axis=1), axis=0)

    # Add the original non-cluster columns.
    df_probs_smooth[list_of_non_cluster_cols] = cluster_probs[list_of_non_cluster_cols]

    # Now reassign 'Cluster' and 'Probability' column to the whole set of trajectories.
    df_probs_smooth['Cluster'] = df_probs_smooth[list_of_cluster_cols].idxmax(axis=1)
    df_probs_smooth['Probability'] = df_probs_smooth[list_of_cluster_cols].max(axis=1)

    return df_probs_smooth

def smoothen_cluster_allocation(cluster_probs, min_threshold=0.15, min_failed_surpasses=2, min_prob_to_count=0.0):
    """
    Refine the cluster allocation passed according to some smoothing criteria that prevents
    unstability in the allocation (column 'Cluster').
    It DOES NOT CHANGE the original probabilities: 
    The process is run separately for each trajectory (with different seed-episode).
    
    Two cases:

    a) The cluster allocation includes the probability of ALL the clusters at each step (e.g.
       Gaussian mixture clustering). Example with two clusters:
             0    1  Cluster  Probability
        0  0.2  0.8        1          0.8
        1  0.3  0.7        1          0.7
        2  0.4  0.6        1          0.6
        3  0.9  0.1        0          0.9
        4  0.6  0.4        0          0.6
        5  0.8  0.2        0          0.8
        6  0.2  0.8        1          0.8

    b) The cluster allocation only includes the estimated probability of the assigned cluster at
       each step (e.g. k-medoids clustering, using silhouette value for the data point).
       THIS CASE IS NOT SUPPORTED YET (I hope I'll only need Gaussian Mixture clustering!)

    Arguments:
    'cluster_probs'         (DataFrame) The original clustering allocation and its
                            respective probability at each step.
    'min_threshold'         (float) Minimal probability delta to overcome to assign a new cluster.
    'min_failed_surpasses'  (int) Minimal number of consecutive steps to overcome previous
                            cluster for less than threshold.
    'min_prob_to_count'     (float) Minimal probability for a cluster to count as selected.

    Returns:                (DataFrame) A copy of the original DataFrame passed with updated columns
                            'Cluster' and 'Probability', reflecting the new cluster allocation and
                            its probability.
    """
    cluster_probs_smooth = cluster_probs.copy()
    smooth_clusters, smooth_probs = [], []
    n_failed_surpasses = 0
    failed_cluster_candidate = None
    last_cluster_selected = cluster_probs_smooth.iloc[0]['Cluster']
    last_seed_episode = cluster_probs_smooth.iloc[0]['Seed'], cluster_probs_smooth.iloc[0]['Episode']
    
    for i in range(len(cluster_probs_smooth)):
        current_seed_episode = cluster_probs_smooth.iloc[i]['Seed'], cluster_probs_smooth.iloc[i]['Episode']
        if current_seed_episode != last_seed_episode:
            # A new trajectory starts. Reset variables.
            n_failed_surpasses = 0
            failed_cluster_candidate = None
            last_cluster_selected = cluster_probs_smooth.iloc[i]['Cluster']

        original_cluster = cluster_probs_smooth.iloc[i]['Cluster']
        original_prob = cluster_probs_smooth.iloc[i]['Probability']
        last_cluster_prob = cluster_probs_smooth.iloc[i][last_cluster_selected]  # Column of 'last_cluster'
        if ((original_cluster == last_cluster_selected) or  # Same cluster.
            (original_prob >= min_prob_to_count) and
            (
                ((original_prob - last_cluster_prob) > min_threshold) or  # New cluster is sufficiently better.
                ((n_failed_surpasses + 1 >= min_failed_surpasses) and (original_cluster == failed_cluster_candidate))  # New cluster is a bit better for enough steps.
            )
        ):
            # Assigned probability is good enough: Take it.
            new_cluster = original_cluster
            new_prob = original_prob
            last_cluster_prob = new_prob
            last_cluster_selected = original_cluster
            n_failed_surpasses = 0
            failed_cluster_candidate = None
        else:
            # Assigned cluster is NOT good enough: Ignore it.
            new_cluster = last_cluster_selected
            new_prob = last_cluster_prob
            if (original_cluster == failed_cluster_candidate) and (original_prob >= min_prob_to_count):
                n_failed_surpasses += 1
            else:
                n_failed_surpasses = 1
                failed_cluster_candidate = original_cluster
        last_seed_episode = current_seed_episode

        smooth_clusters.append(int(new_cluster))
        smooth_probs.append(new_prob)
        
    cluster_probs_smooth['Cluster'] = smooth_clusters
    cluster_probs_smooth['Probability'] = smooth_probs

    return cluster_probs_smooth

def obtain_stable_sequences(cluster_df, min_probability=0.8, min_length=0, seed=None, episode=None, cluster_value=None):
    '''Given:
    cluster_df    (DataFrame) A dataframe with sequences and the cluster probabilities for each time step of the sequence, in which:
        - column 'Seed' is the seed of the sequence,
        - column 'Episode' is the episode number with that seed,
        - column 'Cluster' is the cluster number at that time step,
        - column 'Probability' is the probability of belonging to that cluster,
        - column 'Step' is the time step of the sequence.

    min_probability (float) The minimum probability of belonging to a cluster for a time step to be considered stable.
    min_length      (int)   The minimum length of a sequence to be considered.
    cluster_value   (int)   The cluster value to consider. If None, all clusters are considered.
    seed            (int)   The seed of the sequence to consider. If None, all seeds are considered.
    episode         (int)   The episode number of the sequence to consider. If None, all episodes are considered.
                            Both seed and episode must be None or not None at the same time.

    Returns:

    stable_sequences_df (DataFrame) A dataframe with the consecutive stable sequences within each seed-episode combination, in which:
        - column 'Seed' is the seed of the sequence,
        - column 'Episode' is the episode number with that seed,
        - column 'Cluster' is the cluster number,
        - column 'Start' is the time step in which the sequence starts,
        - column 'End' is the time step in which the sequence ends.
        - column 'Length' is the length of the sequence,
        - column 'Avg_Probability' is the average probability of the steps in the sequence.
    in which ALL the time steps of the sequence have a probability of belonging to that cluster bigger or equal to min_probability.

    The sequences are ordered firstly by cluster, starting with 0, and then by length, from highest to lowest.
    '''
    # Obtain the sequences of stable clusters:
    stable_sequences = []

    # Generate the list of seed-episode combinations to consider:
    if seed is not None and episode is not None:
        seed_episode_list = [(seed, episode)]
    elif seed is None and episode is None:
        seed_episode_list = cluster_df[['Seed', 'Episode']].drop_duplicates().values
    else:
        raise ValueError('Both seed and episode must be None or not None at the same time.')

    # Iterate over unique seed-episode combinations:
    for seed, episode in seed_episode_list:
        # Obtain the cluster probabilities for that seed-episode combination:
        seed_episode_df = cluster_df.loc[(cluster_df['Seed'] == seed) & (cluster_df['Episode'] == episode)]
        # If only one cluster is considered, drop the other clusters:
        if cluster_value is not None:
            seed_episode_df = seed_episode_df.loc[seed_episode_df['Cluster'] == cluster_value]
        # Create a helper column with a boolean indicating if the probability is bigger than min_probability:
        seed_episode_df['Stable'] = seed_episode_df['Probability'] >= min_probability
        # Create another helper column 'crossing' with consecutive numbers for each time step in which 'Stable' changes:
        seed_episode_df['Crossing'] = (seed_episode_df['Stable'] != seed_episode_df['Stable'].shift()).cumsum()
        # Create another helper column 'count':
        seed_episode_df['Count'] = seed_episode_df.groupby(['Stable', 'Crossing']).cumcount(ascending=False) + 1
        seed_episode_df.loc[seed_episode_df['Stable'] == False, 'Count'] = 0
        # Obtain the stable sequences:
        stable_sequences.append(seed_episode_df.loc[seed_episode_df['Count'] > 0])
    # Concatenate the stable sequences:
    stable_sequences_df = pd.concat(stable_sequences)
    # Obtain the length of each sequence with equal seed, episode and crossing:
    stable_sequences_df['Length'] = stable_sequences_df.groupby(['Seed', 'Episode', 'Crossing'])['Count'].transform('max')
    # Drop the sequences with length smaller than min_length:
    stable_sequences_df = stable_sequences_df.loc[stable_sequences_df['Length'] >= min_length]
    # Obtain the start and end value of 'Step' of each sequence with equal seed, episode and crossing:
    stable_sequences_df['Start'] = stable_sequences_df.groupby(['Seed', 'Episode', 'Crossing'])['Step'].transform('min')
    stable_sequences_df['End'] = stable_sequences_df.groupby(['Seed', 'Episode', 'Crossing'])['Step'].transform('max')
    # Obtain the average value of 'Probability' of each sequence with equal seed, episode and crossing:
    stable_sequences_df['Avg_Probability'] = stable_sequences_df.groupby(['Seed', 'Episode', 'Crossing'])['Probability'].transform('mean')
    # Drop unnecessary rows with the same seed, episode and crossing, keeping only the maximum 'Count':
    stable_sequences_df = stable_sequences_df.loc[stable_sequences_df.groupby(['Seed', 'Episode', 'Crossing'])['Count'].idxmax()]
    # Drop unnecessary columns:
    stable_sequences_df = stable_sequences_df[['Seed', 'Episode', 'Cluster', 'Start', 'End', 'Length', 'Avg_Probability']]

    # Order the sequences by cluster, starting with 0:
    stable_sequences_df.sort_values(by=['Cluster', 'Length'], ascending=[True, False], inplace=True)

    # Reset the index:
    stable_sequences_df.reset_index(drop=True, inplace=True)

    return stable_sequences_df

def create_transition_matrix(df_probs, n_components):
    # Initialize transition matrix with n+2 x n+2 with zeros ("Start" at 0, "End" at index n+1).
    transition_matrix = np.zeros((n_components + 2, n_components + 2), dtype=int)
    
    # Group by 'Seed' and 'Episode', then iterate through each group:
    for _, group in df_probs.groupby(['Seed', 'Episode']):
        # Adjust sequence to include 'Start' and 'End'.
        clusters = [0] + [cluster_id + 1 for cluster_id in group['Cluster'].tolist()] + [n_components + 1]        
        # Count transitions
        for start, end in zip(clusters[:-1], clusters[1:]):
            transition_matrix[start, end] += 1
            
    return transition_matrix


#--------------------
# Drawing functions:
#--------------------

def draw_clustering_evaluation(
    eps_list, n_clusters_list,
    silh_score_list, silh_score_valid_list, silh_score_core_list
):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(eps_list, n_clusters_list)
    plt.xlabel('eps')
    plt.ylabel('n clusters')

    plt.subplot(1, 2, 2)
    plt.plot(eps_list, silh_score_list, label='silhouette (all)')
    plt.plot(eps_list, silh_score_valid_list,
             label='silhouette (core + noncore)')
    plt.plot(eps_list, silh_score_core_list, label='silhouette (core)')

    plt.legend(loc="upper left")
    plt.xlabel('eps')
    plt.ylabel('silhouette score')
    plt.tight_layout()
    plt.draw()

def draw_centroid_patterns(centroids, ae_model, ae_columns, log_scale=False):
    fig, ax_list = plt.subplots(1, len(centroids), figsize=(25, 4), squeeze=True)  # squeeze=True
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
    for cluster_number, centroid in enumerate(centroids):
        c_traj = ae_model.regenerate(centroid, ae_columns, denormalize=False)[0]
        _draw_trajectory(
            c_traj.df,
            ax_list[cluster_number],
            ae_columns, title=f"centroid {cluster_number}",
            draw_legend=(cluster_number == 0),
            draw_axes_info=(cluster_number == 0),
            log_scale=log_scale,
        )

def draw_2d_plot_with_clusters(
    df, x_column, y_column, cluster_column, probability_column=None, min_probability=0.0,
    title=None, draw_legend=True, draw_axes_info=True,
    palette='tab10', cmap=None, alpha=0.1, s=5, width=None, height=None,
    cluster_highlights=None,
):
    if width is None and height is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(width, height), squeeze=True)

    if cmap is None:
        cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap(palette).colors[:df[cluster_column].nunique()])

    # If cluster_highlights is not None, generate a color map where all clusters are colored with the same color (grey)
    # except for the clusters in cluster_highlights, which are colored with the colors in cmap:
    if cluster_highlights is not None:
        cmap = matplotlib.colors.ListedColormap(
            [cmap.colors[0] if cluster_number not in cluster_highlights else cmap.colors[cluster_number]
             for cluster_number in range(df[cluster_column].nunique())]
        )

    # Draw all points that have a probability higher than or equal to min_probability:
    # (if probability_column is None, all points are drawn)
    if probability_column is None:
        ax.scatter(df[x_column], df[y_column], c=df[cluster_column], cmap=cmap, alpha=alpha, s=s)
    else:
        ax.scatter(
            df.loc[df[probability_column] >= min_probability, x_column],
            df.loc[df[probability_column] >= min_probability, y_column],
            c=df.loc[df[probability_column] >= min_probability, cluster_column],
            cmap=cmap, alpha=alpha, s=s
        )

    # Draw legend:
    if draw_legend:
        ax.legend(
            handles=[
                matplotlib.lines.Line2D(
                    [], [], marker='o', color='w', markerfacecolor=color, markersize=10,
                    label=f"cluster {cluster_number}"
                )
                for cluster_number, color in enumerate(cmap.colors)
            ],
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    # Draw axes info:
    if draw_axes_info:
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

    # Draw title:
    if title is not None:
        ax.set_title(title)

    return fig, ax

def draw_gm_model_2d_clusters(conv_model_path, gm_model, gm_labels, all_encodings, n_components, draw_centroids=True):
    # Read 2D version of latent space
    latent_space = pd.read_csv(
        f"{conv_model_path}/all_trajectories_enc_2D.txt",
        sep='\t', header=0, float_precision='round_trip',
        usecols=['0', '1'],
    )

    # Draw all points in latent space.

    # Choose COLORS (these are matplotlib's Qualitative color palettes:
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    if n_components <= 10:
        palette = 'tab10'
    else:
        palette = 'tab20'
    cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap(palette).colors[:n_components])

    fig = plt.figure(figsize=(14, 6))
    scatter = plt.scatter(
        latent_space['0'], latent_space['1'],
        s=1, alpha=0.4, c=gm_labels, cmap=cmap)

    # Draw centroids if requested.
    if draw_centroids:
        # Approximate centroids as _closest_ points in latent space.
        centroid_indices = [
            find_closest_point_in_latent_space(gm_model.means_[i], all_encodings) \
                for i in range(gm_model.means_.shape[0])
        ]
        # Draw centroid numbers.
        centroid_2D_coords = latent_space.iloc[centroid_indices]
        lx = list(centroid_2D_coords['0'])
        ly = list(centroid_2D_coords['1'])
        for i in range(n_components):
            plt.annotate(
                str(i), (lx[i] + 2, ly[i] + 0.7),
                fontsize='large', fontweight='medium',
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.5, 'linewidth': 0},
            )
        # Draw centroids.
        plt.scatter(
            centroid_2D_coords['0'], centroid_2D_coords['1'],
            s=110, c=np.arange(n_components), cmap=cmap,
            marker='o', linewidths=3.5, edgecolors='white',
        )

    # Draw Legend.
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=list(np.arange(n_components)), bbox_to_anchor=(1.005, 1))
    for h in range(n_components):
        legend.legendHandles[h]._legmarker.set_alpha(1)

def draw_centroid_patterns_and_cluster_members(
    centroids, ae_model, ae_columns, df_probs, traj_list, window_steps, min_probability=0, seed=None, num_members=None, log_scale=False
):
    NUMBER_OF_MEMBERS = 6
    
    if num_members is None:
        num_members = NUMBER_OF_MEMBERS

    for cluster_number, centroid in enumerate(centroids):
        fig, ax_list = plt.subplots(1, 1 + num_members, figsize=(25, 4), squeeze=True)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
        
        # Draw centroid on first ax, and find some points.
        c_traj = ae_model.regenerate(centroid, ae_columns, denormalize=False)[0]
        traj_dict_list, n_cluster_members = _find_some_points_in_this_cluster(  # Get 'n_cluster_members' before drawing them.
            df_probs, cluster_number, n=num_members, min_probability=min_probability, seed=seed,
        )
        _draw_trajectory(c_traj.df, ax_list[0], ae_columns, title=f"centroid {cluster_number} ({n_cluster_members} points)", log_scale=log_scale)

        # Draw cluster members.
        for ax_idx, traj_data in enumerate(traj_dict_list):
            tr = _find_trajectory_in_list(traj_list, traj_data['Seed'], traj_data['Episode'])
            norm_tr_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
                [tr], ae_model.mean_value_dict, ae_model.std_value_dict
            )
            _draw_trajectory(
                norm_tr_list[0].df.iloc[traj_data['Step'] - window_steps + 1 : traj_data['Step'] + 1],
                ax_list[ax_idx + 1],
                ae_columns, title=f"RND point {ax_idx} ({traj_data['Probability']:.2f})",
                draw_legend=False, draw_axes_info=False, log_scale=log_scale
            )

def draw_average_sequences_and_cluster_members(
    centroids, ae_model, ae_columns, df_probs, traj_list, window_steps, min_probability=0, seed=None, num_members=None, log_scale=False
):
    '''
    This draws:
    - the average sequence of points in each cluster
    - a number of cluster members of the cluster
    '''
    NUMBER_OF_MEMBERS = 6

    if num_members is None:
        num_members = NUMBER_OF_MEMBERS

    # Firstly, normalize the full list of trajectories
    norm_traj_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
        traj_list, ae_model.mean_value_dict, ae_model.std_value_dict
    )

    # Process each cluster to get average sequence and best cluster members.
    for cluster_number, _ in enumerate(centroids):
        fig, ax_list = plt.subplots(1, 1 + num_members, figsize=(25, 4), squeeze=True)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

        # Draw first the average + confidence interval sequence on the first ax.
        traj_dict_list, n_cluster_members = _find_some_points_in_this_cluster(  # Get all cluster members meeting conditions.
            df_probs, cluster_number,
            n=None,  # Get ALL the points meeting the conditions.
            min_probability=min_probability,
        )
        avg_norm_tr, std_norm_tr = _generate_average_sequence_from_list(traj_dict_list, norm_traj_list, window_steps, ae_columns)
        _draw_trajectory(
            avg_norm_tr,
            ax_list[0],
            ae_columns,
            title=f"Cluster {cluster_number} ({n_cluster_members} points)",
            log_scale=log_scale,
            confidence_interval=std_norm_tr
        )
        # Draw now some random cluster members.
        rnd_traj_dict_list, n_cluster_members = _find_some_points_in_this_cluster(  # Get 'n_cluster_members' before drawing them.
            df_probs, cluster_number, n=num_members, min_probability=min_probability, seed=seed,
        )
        for ax_idx, traj_data in enumerate(rnd_traj_dict_list):
            tr = _find_trajectory_in_list(norm_traj_list, traj_data['Seed'], traj_data['Episode'])
            _draw_trajectory(
                tr.df.iloc[traj_data['Step'] - window_steps + 1 : traj_data['Step'] + 1],
                ax_list[ax_idx + 1],
                ae_columns, title=f"RND point {ax_idx} ({traj_data['Probability']:.2f})",
                draw_legend=False, draw_axes_info=False, log_scale=log_scale
            )

def draw_centroid_and_average_sequence_in_clusters(
    centroids, ae_model, ae_columns, df_probs, traj_list, window_steps, min_probability=0.5, log_scale=False
):
    # Firstly, normalize the full list of trajectories
    norm_traj_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
        traj_list, ae_model.mean_value_dict, ae_model.std_value_dict
    )

    # Process each centroid for best cluster members.
    for cluster_number, centroid in enumerate(centroids):
        fig, ax_list = plt.subplots(1, 3, figsize=(8, 4), squeeze=True)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

        # Draw centroid on the first ax.
        c_traj = ae_model.regenerate(centroid, ae_columns, denormalize=False)[0]
        traj_dict_list, n_cluster_members = _find_some_points_in_this_cluster(  # Get 'n_cluster_members' before drawing them.
            df_probs, cluster_number,
            n=None,  # Get ALL the points meeting the conditions.
            min_probability=min_probability,
        )
        _draw_trajectory(c_traj.df, ax_list[0], ae_columns, title=f"Centroid {cluster_number} ({n_cluster_members} points)", log_scale=log_scale)

        # Draw the average sequence on the second ax.
        avg_norm_tr, std_norm_tr = _generate_average_sequence_from_list(traj_dict_list, norm_traj_list, window_steps, ae_columns)
        _draw_trajectory(
            avg_norm_tr,
            ax_list[1],
            ae_columns,
            title=f"Avg. sequence",
            draw_legend=False, draw_axes_info=False,
            log_scale=log_scale,
        )

        # Draw the average + confidence interval sequence on the third ax.
        avg_norm_tr, std_norm_tr = _generate_average_sequence_from_list(traj_dict_list, norm_traj_list, window_steps, ae_columns)
        _draw_trajectory(
            avg_norm_tr,
            ax_list[2],
            ae_columns,
            title=f"Avg. sequence (stdev)",
            draw_legend=False, draw_axes_info=False,
            log_scale=log_scale,
            confidence_interval=std_norm_tr
        )

def draw_average_sequence_in_clusters_in_a_row(
    centroids, ae_model, ae_columns, df_probs, traj_list, window_steps, min_probability=0.5, log_scale=False
):
    '''
    A version of draw_average_sequence_in_clusters that draws all the clusters in a row.'''
    # Firstly, normalize the full list of trajectories
    norm_traj_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
        traj_list, ae_model.mean_value_dict, ae_model.std_value_dict
    )

    # Process each centroid for best cluster members.
    fig, ax_list = plt.subplots(1, len(centroids), figsize=(len(centroids)*3, 4), squeeze=True)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

    for cluster_number, centroid in enumerate(centroids):
        # Get 'n_cluster_members' before drawing them.
        # Loop over all the points in the cluster till enough points are found, reducing the probability threshold each time.
        enough_points_found = False
        probability_threshold = min_probability
        while not enough_points_found:
            traj_dict_list, n_cluster_members = _find_some_points_in_this_cluster(
                df_probs, cluster_number,
                n=None,  # Get ALL the points meeting the conditions.
                min_probability=probability_threshold,
            )
            if n_cluster_members >= 2:
                enough_points_found = True
            else:
                probability_threshold -= 0.5
        
        # Draw the average sequence on the second ax.
        avg_norm_tr, std_norm_tr = _generate_average_sequence_from_list(traj_dict_list, norm_traj_list, window_steps, ae_columns)
        _draw_trajectory(
            avg_norm_tr,
            ax_list[cluster_number],
            ae_columns,
            title=f"Cluster {cluster_number} ({n_cluster_members} points)",
            draw_legend=(cluster_number == 0),
            draw_axes_info=(cluster_number == 0),
            log_scale=log_scale,
            confidence_interval=std_norm_tr,
            title_color='black' if probability_threshold >= min_probability else 'red'
        )

def get_average_sequences(
    centroids, ae_model, ae_columns, df_probs, traj_list, window_steps, min_probability=0.5
):
    '''
    Generate an average sequence for each of the classes whose centroids have been learned
    from the closest points to the centroids.

    Arguments:
    centroids (list): List of centroid vectors, each representing the center of a cluster.
    ae_model (object): Autoencoder model used for normalization only.
    ae_columns (list): List of column names in the autoencoder model to be used for sequence generation.
    df_probs (DataFrame): DataFrame containing probabilities of points belonging to each cluster.
    traj_list (list): List of trajectories, where each trajectory is a list of points.
    window_steps (int): Number of steps to consider in the window for averaging.
    min_probability (float, optional): Minimum probability threshold to consider a point as part of a cluster.
    
    Returns:
    list: A list of tuples, where each tuple contains the average normalized trajectory and its standard deviation for a cluster.

    '''
    sequence_list = []
    # Normalize the full list of trajectories using z-score normalization using the mean and 
    # std from the autoencoder model.
    norm_traj_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
        traj_list, ae_model.mean_value_dict, ae_model.std_value_dict
    )

    for cluster_number, centroid in enumerate(centroids):
        # Get 'n_cluster_members' before drawing them.
        # Loop over all the points in the cluster till enough points are found, reducing the probability threshold each time.
        enough_points_found = False
        probability_threshold = min_probability
        while not enough_points_found:
            traj_dict_list, n_cluster_members = _find_some_points_in_this_cluster(
                df_probs, cluster_number,
                n=None,  # Get ALL the points meeting the conditions.
                min_probability=probability_threshold,
            )
            if n_cluster_members >= 2:
                enough_points_found = True
            else:
                probability_threshold -= 0.5
        
        # Generate and store the average sequence.
        avg_norm_tr, std_norm_tr = _generate_average_sequence_from_list(traj_dict_list, norm_traj_list, window_steps, ae_columns)
        sequence_list.append([avg_norm_tr, std_norm_tr])

    return sequence_list


def draw_real_time_clustering_of_one_trajectory(
    traj_list, seed, episode, df_probs, n_components, ae_columns,
    cum_columns=[], annotate=True, cluster_names=None, log_scale=False,
):
    """
    Generate graphics for the step-level categorization for a given trajectory
    based on the previously generated clustering info.

    Grapfic 1:      The original trajectory.
    Graphic 2:      Cluster allocation for each step (a line).
    Graphic 3:      Probabilities for all clusters for each step.

    Arguments:
    'traj_list'     (list) A list with all the Trajectory instances.
    'seed'          (int) Reference to the trajectory to display: the seed it was created from.
    'episode'       (int) Reference to the trajectory to display: the episode within that seed.
    'df_probs'      (DataFrame) The probability given to each cluster for all steps x trajectories.
    'n_components'  (int) The number of possible clusters for each step.
    'ae_columns'    (list) The name of the columns to draw from the Trajectory passed.
    'cum_columns'   [OPTIONAL](list) The name of columns whose accumulated values to draw from the Trajectory passed.
    'annotate'      [OPTIONAL](bool) Whether trajectory annotations must be displayed on the Trajectory (if found).
    'cluster_names' [OPTIONAL](dict) Keys = cluster numbers (int), values = names (str).
    'log_scale'     [OPTIONAL](bool) Use logarithmic scale in y-axis of Trajectory figure.

    Returns:        (None)
    """

    tr = _find_trajectory_in_list(traj_list, seed, episode)
    if tr is not None:
        fig, (ax_traj, ax_clusters, ax_probs) = plt.subplots(3, sharex=True, figsize=(30, 15), gridspec_kw={'height_ratios': [2, 1, 1]})

        # 1. Trajectory:
        df = tr.df
        # 1.1. Trajectory annotations.
        if annotate and exists(f"{tr.path_to_log}episode_{episode}.ann"):
            annotations_df = pd.read_csv(
                f"{tr.path_to_log}episode_{episode}.ann",
                sep='\t', header=0, float_precision='round_trip')
            coords = list(annotations_df['Step'])
            coords.append(tr.len - 1)
            anno_cmap = plt.get_cmap('tab10')
            for i in range(len(coords) - 1):
                ax_traj.axvspan(coords[i], coords[i + 1], facecolor=anno_cmap(i%10), alpha=0.10)  # Cycle 10 colors.
        # 1.2. Solid lines.
        for col in ae_columns:
            ax_traj.plot(
                list(df['Step']), list(df[col]),
                linewidth=0.75, color=ts_tools.DEF_VARIABLE_COLOR[col],  #  Value, Reward, Delta...
                label=col)
        # 1.3. Dashed lines.
        for col in cum_columns:
            new_col_name = f"Cum{col}"
            df[new_col_name] = df[col].cumsum()
            ax_traj.plot(
                list(df['Step']), list(df[new_col_name]), '--', dashes=(5, 5),
                linewidth=0.75, color=ts_tools.DEF_VARIABLE_COLOR[col], alpha=0.5,  # CumRwd...
                label=new_col_name)

        if log_scale:
            ax_traj.set_yscale('symlog')
        ax_traj.grid(axis='y', linestyle='--', alpha=0.5)
        ax_traj.set_title(f"Trajectory (seed {seed}, episode {episode})")
        ax_traj.legend(loc='best')
        ax_traj.xaxis.set_tick_params(length=5, which='major')
        ax_traj.xaxis.set_tick_params(length=5, which='minor')

        # 2. Clustering:
        tr_cluster_probs = df_probs[(df_probs['Seed'] == seed) & (df_probs['Episode'] == episode)]
        # 2.1. Clustering line.
        ax_clusters.plot(
            list(tr_cluster_probs['Step']), list(tr_cluster_probs['Cluster']),
            linewidth=0.75,
        )
        # 2.2. Clustering annotations.
        coords_list, text_list, cluster_list = _generate_cluster_plot_annotations(tr_cluster_probs, cluster_names)
        for coords, text, cluster in zip(coords_list, text_list, cluster_list):
            ax_clusters.annotate(text, coords, fontsize=11, color='black', alpha=0.66)

        ax_clusters.set_ylim([-1, n_components])
        ax_clusters.set_yticks(np.arange(0, n_components, 1))
        ax_clusters.grid(axis='y', linestyle='--', alpha=0.5)
        ax_clusters.set_title(f"Clustering")
        ax_clusters.xaxis.set_tick_params(length=5, which='major')
        ax_clusters.xaxis.set_tick_params(length=5, which='minor')

        # 3. Probabilities:
        if n_components <= 10:
            palette = 'tab10'
        else:
            palette = 'tab20'
        col_list = matplotlib.cm.get_cmap(palette).colors[:n_components]

        if ('0' in tr_cluster_probs.columns) or (0 in tr_cluster_probs.columns):  # Check for individual cluster probabilities.
            for c in range(n_components):
                ax_probs.plot(
                    list(tr_cluster_probs['Step']), list(tr_cluster_probs[c]),
                    linewidth=0.9, label=c,
                    color=col_list[c]
                )
            ax_probs.set_title(f"Probabilities")
        elif 'Probability' in tr_cluster_probs.columns:
            ax_probs.plot(
                list(tr_cluster_probs['Step']), list(tr_cluster_probs['Probability']),
                linewidth=1.0, label='Probability')
            ax_probs.set_title(f"Probability")
        ax_probs.legend(loc='upper left')

        from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        ax_probs.xaxis.set_major_locator(MultipleLocator(100))
        ax_probs.xaxis.set_minor_locator(MultipleLocator(20))
        ax_probs.xaxis.set_tick_params(length=5, which='major')
        ax_probs.xaxis.set_tick_params(length=5, which='minor')
        
    else:
        print(f"Trajectory (seed {seed}, episode {episode} not found in 'traj_list'")

def _draw_trajectory(
    df, ax, ae_columns,
    title=None, draw_legend=True, draw_axes_info=True, log_scale=False,
    confidence_interval=None, title_color='black',
):
    for col in ae_columns:
        ax.plot(
            list(df['Step']), list(df[col]),
            linewidth=1.0, color=ts_tools.DEF_VARIABLE_COLOR[col],  #  Value, Reward, Delta...
            label=col)

    if confidence_interval is not None:
        for col in ae_columns:
            ax.fill_between(
                list(df['Step']),
                list(df[col] - confidence_interval[col]), list(df[col] + confidence_interval[col]),
                color=ts_tools.DEF_VARIABLE_COLOR[col],  #  Value, Reward, Delta...
                alpha=0.07,
            )

    ax.set_ylim([-10, 10])
    if log_scale:
        ax.set_yscale('symlog')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle=':')
    if not draw_axes_info:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().grid(True)
        ax.set_yticklabels([])
        # ax.get_yaxis().set_visible(False)
    if title is not None:
        # Draw title, in the color passed as argument.
        ax.set_title(title)
        ax.title.set_color(title_color)
    if draw_legend:
        ax.legend(loc='best')

def _generate_average_sequence_from_list(traj_dict_list, norm_traj_list, window_steps, ae_columns):
    """
    Arguments:
    'traj_dict_list':   (list) A list of dictionaries, each with data of an original trajectory at a certain step:
                        [{'Seed': 30, 'Episode': 4, 'Step': 29},
                         {'Seed': 30, 'Episode': 4, 'Step': 219},
                         {'Seed': 30, 'Episode': 4, 'Step': 353}]
    'norm_traj_list'    (list) A list of **normalized** Trajectory instances.
    'window_steps'      (int) The input size of the autoencoder.
    'ae_columns'        (list) Names of the columns to generate the average on.

    Returns:
                        (Trajectory) An instance representing the average of all the sequences passed at each step.
                        (Trajectory) An instance representing the st. dev. of all the sequences passed at each step.
    """
    sequence_list = []
    for traj_data in traj_dict_list:
        tr = _find_trajectory_in_list(norm_traj_list, traj_data['Seed'], traj_data['Episode'])
        sequence_df = tr.df.iloc[traj_data['Step'] - window_steps + 1 : traj_data['Step'] + 1]
        sequence_df.reset_index(drop=True, inplace=True)
        sequence_list.append(sequence_df[ae_columns])

    all_seq_df_mean = pd.concat(sequence_list).groupby(level=0).mean()
    all_seq_df_mean['Step'] = all_seq_df_mean.index
    all_seq_df_std = pd.concat(sequence_list).groupby(level=0).std()
    all_seq_df_std['Step'] = all_seq_df_std.index

    return all_seq_df_mean, all_seq_df_std

def _find_trajectory_in_list(traj_list, seed, episode):
    """
    Find a Trajectory instance in a list by its 'seed' and 'episode' values.

    Arguments:
    'traj_list' (list) As list of Trajectory instances.
    'seed'      (int) The seed from which the Trajectory was created.
    'episode'   (int) The episode of that seed.

    Return:
                (Trajectory) An instance of 'Trajectory' or None if not found.
    """
    for t in traj_list:
        tr_seed, tr_episode = t.get_seed_and_episode_from_trajectory()
        if (tr_seed, tr_episode) == (seed, episode):
            return t
    return None

def _find_some_points_in_this_cluster(df_probs, cluster_number, n=3, min_probability=0.0, seed=None):
    """
    Arguments:
    'df_probs'      (DataFrame) The cluster allocation for each point in the latent space. E.g. with 8 clusters:
                        0	    1       2       3       4       5	            6	    7	            Step   Seed	Episode	Cluster Probability
                    0	0.096	0.067	0.000	0.003	0.000	1.234261e-09	0.832	1.058730e-08	19	   30	4	    6       0.832
                    1	0.108	0.082	0.000	0.002	0.000	5.338216e-10	0.806	1.801850e-08	20	   30	4	    6       0.806
                    2	0.065	0.085	0.000	0.002	0.000	3.730458e-10	0.846	5.157544e-08	21	   30	4	    6       0.846
    'idx'           (int) The number of cluster sought.
    'n'             (int) The number of points to find. If 'None', ALL the points meeting conditions will be returned.
    'min_probability'   (float) The minimum value required in column 'Probability'.
    'seed'          (int) A random seed to select the points (with Python's random generation).

    Returns:
                    (list) A random list of dictionaries, each with data of an original trajectory at a certain step:
                        [{'Seed': 30, 'Episode': 4, 'Step': 29},
                         {'Seed': 30, 'Episode': 4, 'Step': 219},
                         {'Seed': 30, 'Episode': 4, 'Step': 353}] 
                    (int) The number of existing points meeting all conditions.
                        
    """
    cluster_members_df = df_probs[(df_probs['Cluster'] == cluster_number) & (df_probs['Probability'] >= min_probability)]
    n_cluster_members = len(cluster_members_df)
    if n is not None and n_cluster_members < n:
        print(f"Warning: can't find {n} points in cluster {cluster_number} with probability >= {min_probability} ({n_cluster_members} found).")
    elif n == 0:
        print(f"Warning: no points found in cluster {cluster_number} with probability >= {min_probability} ({n_cluster_members} found).")

    if n is None:
        # Return ALL the points found.
        traj_dict_list = cluster_members_df[['Seed', 'Episode', 'Step', 'Probability']].to_dict('records')
    elif n_cluster_members > 0:
        # Return 'n' points (at most).
        if seed is not None:
            random.seed(seed)
        rnd_indices = random.sample(range(n_cluster_members), min(n, n_cluster_members))
        traj_dict_list = cluster_members_df[['Seed', 'Episode', 'Step', 'Probability']].iloc[rnd_indices].to_dict('records')
    else:
        # No points found meeting conditions.
        traj_dict_list = []

    return traj_dict_list, n_cluster_members

def _generate_cluster_plot_annotations(cluster_probs, cluster_names=None):
    Y_OFFSET = 0.3

    coords_list = [(int(cluster_probs.iloc[0]['Step']), cluster_probs.iloc[0]['Cluster'] + Y_OFFSET)]
    text_list = [cluster_names[cluster_probs.iloc[0]['Cluster']]] if cluster_names else [int(cluster_probs.iloc[0]['Cluster'])]
    cluster_list = [int(cluster_probs.iloc[0]['Cluster'])]

    last_cluster = cluster_probs.iloc[0]['Cluster']
    for i in range(len(cluster_probs)):
        cluster = cluster_probs.iloc[i]['Cluster']
        if cluster != last_cluster:
            coords_list.append(
                (int(cluster_probs.iloc[i]['Step']), cluster_probs.iloc[i]['Cluster'] + Y_OFFSET)
            )
            txt_cluster = cluster_names[cluster_probs.iloc[i]['Cluster']] if cluster_names else int(cluster_probs.iloc[i]['Cluster'])
            text_list.append(txt_cluster)
            cluster_list.append(int(cluster_probs.iloc[i]['Cluster']))
            last_cluster = cluster
    
    return coords_list, text_list, cluster_list


def draw_emotion_transition_matrix(emotion_transition_matrix, emotional_attribution):
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjust the figure size as needed

    matrix_for_colormap = emotion_transition_matrix.astype(float)  # Convert to float to accept NaN
    mask = np.eye(emotion_transition_matrix.shape[0], dtype=bool)  # A mask for the diagonal elements
    matrix_for_colormap[mask] = np.nan  # Set diagonal values to NaN for the colormap scaling only
    max_val_for_color_scale = np.nanmax(matrix_for_colormap)
    cax = ax.matshow(matrix_for_colormap, cmap='viridis', vmin=0, vmax=max_val_for_color_scale)

    fig.colorbar(cax)  # Add colorbar

    # Define the cluster labels, replace numbers with names from the dictionary
    clusters = ['[Start]'] + [emotional_attribution.get(i, str(i)) for i in range(8)] + ['[End]']

    # Set the ticks to be at the center of each cell
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_yticks(np.arange(len(clusters)))

    # Set the labels for the ticks
    ax.set_xticklabels(clusters, rotation=90, ha='center')  # Rotate labels for horizontal axis
    ax.set_yticklabels(clusters)

    # Add text labels for each cell to increase clarity
    for (i, j), val in np.ndenumerate(emotion_transition_matrix):
        color = 'white' if val < max_val_for_color_scale / 2 else 'black'
        if mask[i, j]:  # Change color for diagonal values to maintain visibility
            color = 'black'
        ax.text(j, i, f'{int(val)}', ha='center', va='center', color=color)

    # Remove the frame around the heatmap
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Set the grid if you want it
    ax.set_xticks(np.arange(-.5, len(clusters), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(clusters), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Turn off the ticks
    ax.tick_params(which="both", bottom=False, left=False)

    plt.show()

