import numpy as np


def get_patterns_of_interval(interval_values, verbose=False):
    '''
    Given an interval as a series of values, calculate its basic patterns.
    
    Returns:
    - mean          (float) The mean value of the interval.
    - std           (float) The standerd deviation of the values of the interval.
    - slope         (float) The slope of the polynomial approximation of degree 1 of the interval.
    - intercept     (float) The intercept (constant avlue) of the polynomial approximation of degree 1 of the interval.
    - last_value    (float) The last value of the linear regression in the interval.
    '''
    mean = np.mean(interval_values)
    std = np.std(interval_values)
    
    polyn = np.polyfit(np.arange(len(interval_values)), interval_values, deg=1)  # Linear approximation of the values passed assuming x = 0, 1, ...
    slope = polyn[0]  # The linear coefficient of the polynomial.
    intercept = polyn[1]  # The intercept, or constant value of the polynomial (first value of the lin. regression in the interval).
    last_value = intercept + slope * (len(interval_values) - 1)  # Last value of the linear regression in the interval.

    if verbose:
        print(f"Mean: {mean:.4} (± {std:.4}). Slope: {slope:.4}, Intercept: {intercept:.4}, Last value: {last_value:.4}")

    return mean, std, slope, intercept, last_value

def get_slope_stats_of_many_normalized_ts(list_of_univar_ts, window_steps, verbose=True):
    '''
    Given a list of normalized univariate time series (mean = 0, std = 1), analyze stats of the slope
    (linear regression) of its intervals of size 'window_steps' across all elements.

    Returns:
    - mean_slope          (float) Mean value of the slope of all the intervals.
    - mean_slope_size     (float) Mean value of the *absolute value* of the slopes of the intervals.
    - mean_positive_slope (float) Mean value of the *positive* slopes of al the intervals.
    - mean_negative_slope (float) Mean value of the *negative* slopes of al the intervals.
    - std_slope           (float) Stdev of the slope of all the intervals.
    - std_slope_size      (float) Stdev of the *absolute value* of the slopes of the intervals.
    - std_positive_slope  (float) Stdev of the *positive* slopes of al the intervals.
    - std_negative_slope  (float) Stdev of the *negative* slopes of al the intervals.
    '''
    means_list, p_means_list, n_means_list, abs_means_list = [], [], [], []

    for ts in list_of_univar_ts:
        for i in range(len(ts) - window_steps + 1):
            _, _, slope, _, _ = get_patterns_of_interval(ts[i:i+window_steps])  # Keep slope only.
            means_list += [slope]
            abs_means_list += [abs(slope)]
            if slope > 0:
                p_means_list += [slope]
            else:
                n_means_list += [slope]
    
    mean_slope = np.mean(means_list)
    mean_slope_size = np.mean(abs_means_list)
    mean_positive_slope = np.mean(p_means_list)
    mean_negative_slope = np.mean(n_means_list)
    std_slope = np.std(means_list)
    std_slope_size = np.std(abs_means_list)
    std_positive_slope = np.std(p_means_list)
    std_negative_slope = np.std(n_means_list)

    if verbose:
        print(f"Slope stats of the list of TS's:")
        print(f"(-)mean = {mean_negative_slope:.4} (+/- {std_negative_slope:.4})")
        print(f"mean = {mean_slope:.4} (+/- {std_slope:.4})")
        print(f"(+)mean = {mean_positive_slope:.4} (+/- {std_positive_slope:.4})")
        print(f"(abs) mean = {mean_slope_size:.4} (+/- {std_slope_size:.4})")
        print("")

    return mean_slope, mean_slope_size, mean_positive_slope, mean_negative_slope, \
        std_slope, std_slope_size, std_positive_slope, std_negative_slope


def map_learned_pattern_to_love_pattern(
    mean, slope, std,   # Stats of the learned pattern (normalized).
    intercept,          # Intercept of the linear regression of the learned pattern.
    last_value,         # Last value of the linear regression of the learned pattern.
    mean_slope,         # Mean value of the slopes of the original MTS (normalized).
    std_slope,          # Stdev of the slopes of the original MTS (normalized).
    love_pattern=5      # 5 or 6 to detect LOVE x5 or LOVE x6 patterns respectively.
):
    # Heuristic mapping of the learned emotional pattern.
    # We use classification intervals of length 0.50 vs
    # a) referential mean slope ('mean_slope_size') to classify trend changes.
    # b) referential mean value (0.0, since the series is normalized) to classify stable patterns.
    SENSITIVITY_THRESHOLD = 0.50
    
    VALENCE_THRESHOLD = SENSITIVITY_THRESHOLD / 2  # (0.25) Stable values between [VALENCE_THRESHOLD, HIGH_VALENCE_THRESHOLD] => Positive / Negative
    HIGH_VALENCE_THRESHOLD = VALENCE_THRESHOLD + SENSITIVITY_THRESHOLD  # (0.75) Stable values between [HIGH_VALENCE_THRESHOLD, infinity] => high Positive / Negative
    SLOPE_VALENCE_THRESHOLD = SENSITIVITY_THRESHOLD  # (0.50) Slopes above SLOPE_VALENCE_THRESHOLD * mean_slope_size => Increased / Decreased
    
    assert love_pattern in (5, 6), f"Wrong value ({love_pattern}) for argument 'love_pattern'; expected 5 or 6."

    # 1. Trend changes prevail: check slope size.
    if (slope > mean_slope + SLOPE_VALENCE_THRESHOLD * std_slope) or (slope < mean_slope - SLOPE_VALENCE_THRESHOLD * std_slope):
        if slope > 0:
            pattern = 'Increased'
        else:  # 'Decreased' differentiates between x5 and x6 LOVE refferential patterns.
            if love_pattern == 5:
                # Love x5 has a single pattern for decreasing trends.
                pattern = 'Decreased'
            elif last_value > 0:
                # Love x6 has two patterns for decreasing trends.
                pattern = 'Decreased to avg.'
            else:
                pattern = 'Decreased to neg.'
    
    # 2. Stable, very large values (compared to std).
    elif mean >= HIGH_VALENCE_THRESHOLD:
        pattern = 'high Positive'
    elif mean <= -HIGH_VALENCE_THRESHOLD:
        pattern = 'high Negative'
    # 3. Stable, large values (compared to std.)
    elif mean >= VALENCE_THRESHOLD:
        pattern = 'Positive'
    elif mean <= -VALENCE_THRESHOLD:
        pattern = 'Negative'
    # 4. Stable, average values.
    else:
        pattern = 'Average'
    
    stats_diagn = f"mean = {mean:+.4f} (+/- {std:.4f})\t slope {slope:+.4f} vs mean_slope_size {mean_slope:.4f} (+/- {std_slope:.4f})"

    return pattern, stats_diagn


def obtain_love_pattern_of_sequences_based_on_historic_mts(
        mean_slope, std_slope, sequence_list, value_name, love_pattern=5, verbose=False, draw=True
):
    """
    Compare each of the learned sequences (emotion patterns) to referential LOVE
    (Latest Observed Values Encoding) patterns, generating a text report.

    Args:
    mean_slope          (float) Mean value of the slopes of the intervals.
    std_slope           (float) Stdev of the slopes of the intervals.
    sequence_list       (list) List with average sequences learned for the values.
    love_pattern        (int) 5 or 6 to detect LOVE x5 or LOVE x6 patterns respectively.
    value_name          (str) The column to match from 'sequence_list' ('Reward' or 'Value').
    verbose             (bool) Turn on/off diagnostics.

    Returns: (nothing)
    ...
    """
    # Obtain the stats for each of the learned emotional pattern to map.
    print(f"'{value_name}' LOVE x{love_pattern} patterns:")
    for emotion in range(len(sequence_list)):
        # Stats and Linear regression of the average sequence (first list) of the learned emotion.
        mean, std, slope, intercept, last_value = get_patterns_of_interval(sequence_list[emotion][0][value_name], verbose=verbose)
        # Standard deviation of the avg. seq. learned as avg. std. (second list) over all steps.
        mean_std, _, _, _, _ = get_patterns_of_interval(sequence_list[emotion][1][value_name], verbose=verbose)

        # Map the learned emotional pattern.
        pattern, stats_diagn = map_learned_pattern_to_love_pattern(mean, slope, mean_std, intercept, last_value, mean_slope, std_slope, love_pattern)
    
        print(f"- Cluster {emotion}: {pattern:<15}\t {stats_diagn}")