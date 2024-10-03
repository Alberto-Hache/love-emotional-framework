"""
Tools and utilities to work with multivariate time series.
"""

import pandas as pd
import numpy as np
import itertools
import glob
import copy
import os

# Bokeh library stuff
# Select palettes from https://docs.bokeh.org/en/latest/docs/reference/palettes.html#bokeh-palette
# Names of colors: https://cjdoris.github.io/Bokeh.jl/v0.4/colors/
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Band, DataRange1d, Legend
from bokeh.palettes import Category10
from bokeh.models import HoverTool


Y_AXIS_PADDING = 5

DEF_VARIABLE_COLOR = dict({
    'Value': 'orange',
    'Reward': 'green',
    'Delta': 'red',
    'EmaRwd': 'purple',
    'AvgRwd': 'blue',
    'CumRwd': 'green',
})
TABLEAU_10_COLORS = dict({  # Matches matplotlib.
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
})
BOKEH_CATEGORY10_COLORS = dict({  # Similar to tableau's.
    'blue': '#2678b2',
    'orange': '#fd7f28',
    'green': '#339f34',
    'red': '#d42a2f',
    'purple': '#946abb',
    'brown': '#8b564d',
    'pink': '#e17ac1',
    'gray': '#7f7f7f',
    'olive': '#bcbc35',
    'cyan': '#29bece',
})

DEFAULT_TIME_COLUMN = 'Step'


class Trajectory:
    """A multivariate time series from an RL episode's log file."""

    def __init__(self, path_to_log=None, log_file=None, df=None, experiment_id='Exp.', columns=None):
        """
        Create a Trajectory (multivariate time series) from:
        a) an episode's log file in the given path.
        b) a given pandas data frame.

        Parameters
        ----------
        path_to_log : str (or None)
            The path to the log file (e.g. '/Users/Alberto/exp-01/train_series_s0/').
        log_file : str (or None)
            The name of the log file (e.g. 'episode_3.txt').
        df : pandas DataFrame (or None)
            The prebuilt data frame to use (if passed, 'path_to_log' and 'log_file' are ignored).
        experiment_id : str, optional
            An identifier of the experiment this log belongs to (e.g. '03.01 - LunarLander-v2').
        columns: list, optional
            The list of columns to keep, removing any other one.
        """
        self.path_to_log = path_to_log
        self.log_file = log_file
        self.experiment_id = experiment_id
        if df is None:
            self.name = f"{experiment_id} ({log_file})"
            self.df = pd.read_csv(
                path_to_log + log_file, sep='\t', header=0, float_precision='round_trip')
        else:
            self.name = f"{experiment_id} (from a DataFrame)"
            self.df = df

        if columns is not None:
            self.df = self.df[columns]
        self._validate_ts()
        self._update_ts_data()

    def describe(self):
        print(self.stats)

    def get_seed_and_episode_from_trajectory(self):
        """
        Return string values of the seed and episode in which this Trajectory was recorded.
        Values are extracted from self.path_to_log and self.log_file.)

        Returns:
            (int)    Seed number (e.g. 30).
            (int)    Episode number (e.g. 4).
        """
        seed = int(self.path_to_log.split('/')[-2].split('_s')[-1])
        episode = int(self.log_file.split('.txt')[0].split('episode_')[-1])
        return seed, episode
        
    def plot(self, columns=None, aux_columns=[], on_notebook=True, plot_width=1200, plot_height=600, \
        legend_location=None, legend=True, plot_title=None, hide_toolbar=False, return_figure=False, y_range=None):
        """
        Parameters
        ----------
        TBA:
        ... 
        'legend_location'   (string) 'center_left', 'top_left', 'bottom_left'
        ...
        """
        palette = TABLEAU_10_COLORS
        default_palette = itertools.cycle(Category10[10])
        if not plot_title:
            plot_title = f"{self.experiment_id} ({self.log_file})"

        length = self.len
        if not y_range:
            y_range = self._calculate_y_range()

        p = figure(
            plot_width=plot_width, plot_height=plot_height,
            title=plot_title,
            # x_range=DataRange1d(0, length - 1, max_interval=1), y_range=y_range,
            x_range=(0, length - 1), y_range=y_range,
            output_backend='svg',  # Vectorial format
        )
        # p.xaxis.ticker = [i for i in range(length)]
        if not columns:
            columns = list(self.df.columns)
            if DEFAULT_TIME_COLUMN in columns:
                columns.remove(DEFAULT_TIME_COLUMN)

        for column in columns:
            p.line(
                np.arange(0, length),
                self.df[column],
                legend_label=column,
                color=palette.get(
                    DEF_VARIABLE_COLOR.get(column, "Unknown"),
                    next(default_palette)),
                line_alpha=0.75 if column in aux_columns else 1.0,
                line_dash='dashed' if column in aux_columns else 'solid',
                #line_width= 1.2,
            )

        if legend_location == "out":  # AH: My special string for "out of plot"
            p.add_layout(Legend(), 'right')
            p.legend.click_policy = 'mute'  # 'hide', 'mute'
        else:
            if legend_location:
                p.legend.location = legend_location
            else:
                p.legend.location = self._define_legend_location(y_range)
            p.legend.click_policy = 'mute'  # 'hide', 'mute'

        p.legend.visible = legend
        p.add_tools(HoverTool(tooltips="y: @y, x: @x",
                    mode="vline", muted_policy='ignore'))
        p.xgrid.minor_grid_line_color = 'navy'
        p.xgrid.minor_grid_line_alpha = 0.1
        p.xaxis.axis_label = DEFAULT_TIME_COLUMN
        """
        p.ygrid.minor_grid_line_color = 'navy'
        p.ygrid.minor_grid_line_alpha = 0.1
        """
        if hide_toolbar:
            p.toolbar.logo = None
            p.toolbar_location = None

        if not return_figure:
            if on_notebook:
                output_notebook(hide_banner=True)
            show(p)
            return None
        else:
            return p

    def _calculate_y_range(self):
        min_y = min(self.stats.loc['min'][self.value_columns])
        max_y = max(self.stats.loc['max'][self.value_columns])
        y_range = (min_y - Y_AXIS_PADDING, max_y + Y_AXIS_PADDING)
        return y_range

    def _define_legend_location(self, y_range):
        """A basic heuristic based on first values of the series."""
        first_values = list(
            self.df.iloc[self.first_valid_index])  # First valid value of each column.
        # Clean possible 'nan' values.
        first_values = [x for x in first_values if str(x) != 'nan']
        lower_gap = abs(y_range[0] - min(first_values))
        higher_gap = abs(y_range[1] - max(first_values))
        mid_gap = abs(y_range[1] - y_range[0] - lower_gap - higher_gap)
        if mid_gap > max([higher_gap, lower_gap])*1.20:
            location = 'center_left'
        else:
            location = 'top_left' if higher_gap >= lower_gap else 'bottom_left'
        return location

    def _update_ts_data(self):
        self.len = len(self.df)
        self.stats = self.df.describe().drop(['count', '25%', '50%', '75%'])
        self.value_columns = list(self.df.columns)
        self.value_columns.remove(DEFAULT_TIME_COLUMN)
        self.first_valid_index = self.df[self.value_columns[0]].first_valid_index(
        )

    def _validate_ts(self):
        """
        """
        time_column = DEFAULT_TIME_COLUMN
        expected_columns = set(
            ['Reward', 'Value', 'Delta'])
        cols = set(self.df.columns)
        assert (time_column in cols), \
            f"Error: Column {time_column} is missing in log file {self.log_file}."
        # Test no-longer required (columns used will vary over experiments).
        # if self.log_file is not None:  # Skip check for prebuilt dataframes.
        #     assert (expected_columns.issubset(cols)), \
        #         f"Error: Columns {expected_columns} missing in log file {self.log_file}."

    def __repr__(self):
        """Result from prompt on this class."""
        txt_1 = f"{self.name}"
        txt_2 = str(self.df.__repr__())
        return f"{txt_1}\n{txt_2}"


def plot_cross_trajectory_stats(df_means, columns=None, aux_columns=[], df_stds=None, sigma_ratio=1, on_notebook=True, experiment_id="RL Experiment"):
    """
    Parameters
    ----------
    TBA
    """
    if columns is not None:
        assert set(columns).issubset(set(df_means.columns)), \
            f"Error: 'columns' {columns} doesn't match 'df_means' ({df_means.columns})."
    else:
        columns = df_means.columns
    n_trajectories = df_means.shape[0]

    palette = TABLEAU_10_COLORS
    default_palette = itertools.cycle(Category10[10])
    if df_stds is None:
        plot_title = f"{experiment_id} - Episodic stats"
    else:
        plot_title = f"{experiment_id} - Episodic stats (sigma {sigma_ratio})"
    p = figure(
        plot_width=1200, plot_height=600,
        title=plot_title,
        x_range=(0, n_trajectories),
        output_backend='svg',  # Vectorial format
    )

    for col_name in columns:
        var_color = palette.get(
            DEF_VARIABLE_COLOR[col_name], next(default_palette))
        p.line(
            np.arange(n_trajectories),
            df_means[col_name],
            legend_label=f"Avg. {col_name}",
            color=var_color,
            line_alpha=0.75 if col_name in aux_columns else 1.0,
            line_dash='dashed' if col_name in aux_columns else 'solid',
        )
        if df_stds is not None:
            assert df_means.shape == df_stds.shape, f"Error: 'df_means' and 'df_stds' have different shape."
            df = pd.DataFrame(dict(
                x=np.arange(n_trajectories),
                means=df_means[col_name],
                stds=df_stds[col_name],
            ))
            df['y1'] = df.means - sigma_ratio*df.stds
            df['y2'] = df.means + sigma_ratio*df.stds
            source = ColumnDataSource(df)
            band = Band(
                base='x', lower='y1', upper='y2', source=source,
                fill_color=var_color, fill_alpha=0.25, line_width=0,
            )
            p.add_layout(band)

    p.legend.location = 'top_left'
    p.legend.click_policy = 'mute'  # 'hide', 'mute'
    p.add_tools(HoverTool(tooltips="y: @y, x: @x",
                mode="vline", muted_policy='ignore'))
    p.xgrid.minor_grid_line_color = 'navy'
    p.xgrid.minor_grid_line_alpha = 0.1
    p.xaxis.axis_label = 'Episode'

    if on_notebook:
        output_notebook(hide_banner=True)
    show(p)


def calculate_trajectories_stats(trajectory_list, at_step=None, window_steps=None):
    """
    Calculate the min/max/mean/std-dev of each trajectory feature
    across ALL the trajectories in the list passed.
    """
    ref_trajectory = trajectory_list[0]
    reference_columns = list(ref_trajectory.stats.columns)
    time_column = DEFAULT_TIME_COLUMN
    if time_column in reference_columns:
        reference_columns.remove(time_column)

    from_idx, to_idx = determine_scope_to_handle(
        at_step, window_steps, ref_trajectory.df)

    joined_series = dict([c, []] for c in reference_columns)
    for t in trajectory_list:
        for c in reference_columns:
            joined_series[c].extend(t.df[c][from_idx:to_idx])

    min_value_dict = dict([c, min(joined_series[c])]
                          for c in reference_columns)
    max_value_dict = dict([c, max(joined_series[c])]
                          for c in reference_columns)
    mean_value_dict = dict([c, np.mean(joined_series[c])]
                           for c in reference_columns)
    std_value_dict = dict([c, np.std(joined_series[c])]
                          for c in reference_columns)

    return min_value_dict, max_value_dict, mean_value_dict, std_value_dict


def z_score_normalize_trajectory_list(trajectory_list, mean_dict=None, std_dict=None):
    """
    Return a copy of the list of Trajectory instances passed
    that is z-score normalized (i.e. 'value_columns' have 0 mean and 1 std.dev).
    Mean and std-dev are calculated across the whole list for each column unless
    both are provided as arguments (e.g. to normalize test set with train set stats).

    Arguments:
    'trajectory_list'   (list) A list of Trajectory instances. All should have
                        the same columns (but may differ in length).
    'mean_dict'         (dict) [Optional] The precalculated mean of each column.
    'std_dict'          (dict) [Optional] The std. dev. of each column.

    Returns:
    (list)              A copy of the list with normalized values.
    (dict)              The mean of each column.
    (dict)              The std. dev. of each column.
    """
    assert (mean_dict is None and std_dict is None) or (mean_dict is not None and std_dict is not None), \
        f"'mean_dict' and 'std_dict' must be both None or both precalculated dictionaries."
    if mean_dict is None:
        _, _, mean_dict, std_dict = calculate_trajectories_stats(
            trajectory_list)

    norm_trajectory_list = []
    for t in trajectory_list:
        t_norm = copy.deepcopy(t)
        for c in t.value_columns:
            t_norm.df[c] = (t_norm.df[c] - mean_dict[c]) / std_dict[c]
        t_norm._update_ts_data()
        norm_trajectory_list.append(t_norm)
    return norm_trajectory_list, mean_dict, std_dict


def invert_normalization_of_trajectory_list(norm_trajectory_list, mean_dict, std_dict):
    trajectory_list = []
    for t in norm_trajectory_list:
        t_invert = copy.deepcopy(t)
        for c in t.value_columns:
            t_invert.df[c] = std_dict[c] * t_invert.df[c] + mean_dict[c]
        t_invert._update_ts_data()
        trajectory_list.append(t_invert)
    return trajectory_list


def invert_normalization_of_sample_list(norm_sample_list, mean_list, std_list):
    sample_list = []
    for s in norm_sample_list:
        s = np.array(s) * std_list + mean_list
        sample_list.append(s)
    return sample_list


def calculate_df_prediction_errors(
    actual_traj_df, predicted_traj_df,
    at_step=None, window_steps=None,
    verbose=False
):
    """
    Compare two trajectories, the actual one vs a prediction, passed as
    multivariate time series (DataFrame).
    If 'at_step' or 'window_steps' are passed, only a fraction of 'actual_df' is compared.
    """
    time_column = DEFAULT_TIME_COLUMN
    if time_column in actual_traj_df.columns:
        # Returns a copy.
        actual_traj_df = actual_traj_df.drop(time_column, axis=1)
    if time_column in predicted_traj_df.columns:
        # Returns a copy.
        predicted_traj_df = predicted_traj_df.drop(time_column, axis=1)

    from_idx, to_idx = determine_scope_to_handle(
        at_step, window_steps, actual_traj_df)
    # Trim original df to prediction's size.
    actual_df_segment = actual_traj_df[from_idx:to_idx]
    assert actual_df_segment.shape == predicted_traj_df.shape, \
        f"Two trajectories were passed of different shapes, \
        ({actual_df_segment.shape}) and ({predicted_traj_df.shape}), \
        'at_step'={at_step}, 'window_steps'={window_steps}."

    diff = predicted_traj_df - actual_df_segment
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(np.square(diff), axis=0)
    rmse = np.sqrt(np.mean(np.square(diff), axis=0))

    if verbose:
        report = pd.concat([mae, mse, rmse], axis=1)
        report.columns = (('MAE', 'MSE', 'RMSE'))
        print(report)

    return mae, mse, rmse


def calculate_prediction_errors_of_list(
    actual_traj_list, predicted_traj_list, at_step=None, window_steps=None, verbose=False, display_precision=None
):
    full_actual_df = pd.concat(actual_traj_list, ignore_index=True)
    full_pred_df = pd.concat(predicted_traj_list, ignore_index=True)
    mae, mse, rmse = calculate_df_prediction_errors(
        full_actual_df, full_pred_df, at_step, window_steps)

    if verbose:
        if display_precision:
            pd.set_option("display.precision", display_precision)
        df_errors = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE'])
        df_errors['MAE'] = mae
        df_errors['MSE'] = mse
        df_errors['RMSE'] = rmse
        print(df_errors)

    return mae, mse, rmse


def determine_scope_to_handle(at_step, window_steps, trajectory_df):
    """
    Return the two indices of a DataFrame slice ending at 'at_step' that is 'window_steps' wide.
    """
    if at_step is None:
        at_step = len(trajectory_df) - 1
    assert at_step < len(trajectory_df), \
        f"Error: 'at_step' {at_step} exceeds length of {len(trajectory_df)}."

    if window_steps is None:
        window_steps = min(len(trajectory_df), at_step + 1)
    assert window_steps <= at_step + 1, \
        f"Error: 'window_steps' {window_steps} should be <= {at_step + 1}"

    to_idx = at_step + 1
    from_idx = to_idx - window_steps
    return from_idx, to_idx


def polynomial_fit(df, deg=1, value_columns=None, at_step=None, window_steps=None):
    """
    Approximate a Trajectory to a polynomial function.
    If 'at_step' or 'window_steps' are passed, only a fraction of the trajectory is fit.
    Time column is assumed to be 'DEFAULT_TIME_COLUMN' or else the first value.

    Arguments:
    df              (DataFrame) The MTS to fit.
    deg             (int) The degree of the polynomial approximation.
    value_columns   (list) The series to approximate within the MTS. If None,
                    take all but DEFAULT_TIME_COLUMN.
    at_step         (int) The time step to fit, or the last one if None.
    window_steps    (int) The number of steps to use starting from 'at_step'.

    Returns:        (np.array) The approximations, with shape (deg + 1, number of columns to fit).
                    E.g. Fit 2 columns with degree 2 will return a shape (3, 2).
    """
    from_idx, to_idx = determine_scope_to_handle(at_step, window_steps, df)
    assert window_steps > deg, f"Error: can't approxiamte a deg {deg} polynomial from only {window_steps} values."

    if not value_columns:
        value_columns = list(df.columns)
    time_column = DEFAULT_TIME_COLUMN if DEFAULT_TIME_COLUMN in df.columns else value_columns[0]
    if time_column in value_columns:
        value_columns.remove(time_column)

    return np.polyfit(df[time_column][from_idx:to_idx], df[value_columns][from_idx:to_idx], deg=deg)


def polynomial_approximate(x, poly):
    """
    Approximate the value of a multivariate time series (MTS) at certain step 'x' with
    polynomial approximator 'poly'.

    Args:
    'x '        (float)
    'poly'      (np.array) The polynomial function, with shape (deg + 1, number of columns to calculate).

    Returns:    (list[float]) The approximated values of the MTS at the x passed.
    """
    return np.polyval(poly, x)


def polynomial_derivative(poly):
    """
    Calculate the first derivative of the polynomial at the given point.

    Args:
    'poly'      (np.array) The polynomial function, with shape (deg + 1, number of columns to calculate).

    Returns:    (np.array)
    """
    degree = poly.shape[0] - 1

    return np.array([poly[line] * (degree - line) for line in range(degree + 1)])[:-1]


def read_all_trajectories_in_path(path, experiment_id="No_exp_id", columns=None, verbose=False):
    """
    Recursively search the path passed for all files matching wildcard 'episode_*.txt'
    and return a list of Trajectory instances.

    Arguments:
    'path'          (string) The path to search from, accepting wildcards, e.g.
                    '/Users/Alberto/Code/spinningup-tests2/experiments/exp-01.02/test_series_s*/'
    'experiment_id' (string) [Optional] Descriptive string to assign to the Trajectory.
    'columns'       (list) [Optional] Name of the columns to keep from the trajectories read.
    'verbose'       (bool) Report on actions done.

    Return:
    (list)
    """
    list = []
    subdir_list = glob.glob(path)
    for subdir in subdir_list:
        subdir_file_list = glob.glob(f"{subdir}episode_*.txt")
        for file in subdir_file_list:
            t = Trajectory(
                subdir, file.split('/')[-1],
                experiment_id=experiment_id,
                columns=columns
            )
            list.append(t)

    if verbose:
        print(f"{len(list)} trajectories read.")
        print(f"Example: \n{list[0]}")

    return list


def read_all_trajectories_in_file_list(train_set_list, experiment_id="No_exp_id", columns=None):
    """
    Create a list of Trajectory instances from the list with full paths passed.

    Arguments:
    'train_set_list'    (list) The full paths to read Trajectories from, e.g.
                        ['/Users/Alberto/Code/spinningup-tests2/experiments/exp-01.02/test_series_s1/episode_4.txt']
    'experiment_id'     (string) [Optional] Descriptive string to assign to the Trajectory.
    'columns'           (list) [Optional] Name of the columns to keep from the trajectories read.

    Return:
    (list)
    """
    list = []
    for file_path in train_set_list:
        path, file = os.path.split(file_path)
        t = Trajectory(
            f"{path}/", file,
            experiment_id=experiment_id,
            columns=columns
        )
        list.append(t)
    return list


def create_sequences_from_trajectory_list(t_list, window_steps, columns=None):
    """
    Split a list of Trajectory instances in sequences of a given length.

    Arguments:
    't_list'        (list) Trajectory instances to sequence.
    'window_steps'  (int) Desired length of the sequences.
    'columns'       (list) String names of the columns to keep (all by default).

    Returns:
    (list)          A sequence of Dataframes of equal length from the list passed.
    """
    if not columns:
        columns = t_list[0].df.columns
    seq_list = []
    for traj in t_list:
        traj_cols_df = traj.df[columns]
        t_seq_list = [traj_cols_df[t:t+window_steps] for t in range(0, len(traj_cols_df) - window_steps + 1)]
        seq_list.extend(t_seq_list)
    return seq_list

