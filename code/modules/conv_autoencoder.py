"""
Convolutional autoencoder
"""
from operator import index
import keras
import numpy as np
import pandas as pd

import time_series_tools as ts_tools


class ConvolutionalAutoencoder():
    """
    """

    def __init__(
        self, window_steps, model_to_load=None, encoder_model_to_load=None, decoder_model_to_load=None,
        train_set_files_list_f=None, ae_columns=None):
        """
        Arguments:
        'window_steps'              (int) The number of time steps used by the AE to encode.
        'autoencoder_model_to_load' (str) Path of the conv. encoder to load (full AE).
        'encoder_model_to_load'     (str) Path of the conv. encoder to load (partial AE, 1st submodel).
        'decoder_model_to_load'     (str) Path of the conv. decoder to load (partial AE, 2nc submodel).

        (Optional: Fit model to get normalization stats from **training** set)
        'train_set_files_list_f'    (str) Path to file with normalization trajectories to use.
        'ae_columns'                (str) String names of the columns used by the AE.
        """
        self.window_steps = window_steps
        assert self.window_steps > 0, "Parameter 'window_steps' must be > 0"

        assert (encoder_model_to_load is None and model_to_load is None) or \
               (encoder_model_to_load is not None and model_to_load is not None), \
            "Both 'encoder_model_to_load' and 'autoencoder_model_to_load' must passed (or both be None)."

        if not encoder_model_to_load:
            self.conv_encoder = None
            self.conv_decoder = None
            self.conv_model = None
            self.trained = False
        else:
            self.conv_encoder = keras.models.load_model(encoder_model_to_load)
            self.conv_decoder = keras.models.load_model(decoder_model_to_load)
            self.conv_model = keras.models.load_model(model_to_load)
            assert self.conv_encoder.input_shape[1] == window_steps, \
                f"Error: the input shape 'conv_encoder' is {self.conv_encoder.input_shape}, not matching 'window_steps' of {window_steps}."

            self.trained = True
            if train_set_files_list_f is not None and ae_columns is not None:
                with open(train_set_files_list_f) as f:
                    train_set_list = f.readlines()
                    train_set_list = [line.rstrip() for line in train_set_list]
                _ = self.fit(train_set_list, ae_columns)  # NO training here.

    def fit(self, trajectory_file_list, ae_columns):
        """
        Arguments:
        'trajectory_file_list'  (list) Paths to the trajectory files (.txt).
        'ae_columns'            (list) String names of the columns used by the AE.
        """
        assert len(trajectory_file_list) > 0, "No trajectories were passed to 'fit()'"

        self.ae_columns = ae_columns
        trajectory_list = ts_tools.read_all_trajectories_in_file_list(
            trajectory_file_list)  # , columns=ae_columns

        self._capture_trajectories_stats(trajectory_list)
        if not self.trained:
            self.trained = self._train_autoencoder(trajectory_list)

        return self.trained

    def _capture_trajectories_stats(self, trajectory_list):
        self.min_value_dict, self.max_value_dict, self.mean_value_dict, self.std_value_dict = \
            ts_tools.calculate_trajectories_stats(trajectory_list)

    def _train_autoencoder(self, trajectory_list):
        # Deep AE training not implemented here. Model must be provided at creation.
        return False

    def encode_at_step(self, trajectory, at_step=None):
        """
        Encode a given unnormalized Trajectory starting at a certain step and looking a 
        number of steps in the past.

        Arguments:
        'trajectory'    (Trajectory)
        'at_step'       (int) Limit of the time window to encode. If None, the last step.

        Returns:        (list) A list of 'float' values with the mean values of each series
                        within the 'window_steps' of the encoder.
        """
        assert self.trained, "The autoencoder wasn't trained yet."

        normalized_traj_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
            [trajectory], self.mean_value_dict, self.std_value_dict
        )

        normalized_df = normalized_traj_list[0].df[self.ae_columns]
        if at_step is None:
            at_step = trajectory.len - 1

        from_idx, to_idx = ts_tools.determine_scope_to_handle(
            at_step, self.window_steps, normalized_df)
        normalized_seq = np.array([normalized_df[from_idx:to_idx]])
        encoding = self.conv_encoder.predict(normalized_seq)

        return encoding[0]  # .predict returned a list of predictions.

    def encode_full_trajectory(self, trajectory):
        """
        Encode a given unnormalized trajectory producing a new trajectory of equal length,
        applying the AE's normalization.
        The first steps below 'window_steps' are set to np.nan.

        Arguments:
        'trajectory'    (Trajectory) An unnormalized instance of Trajectory.

        Returns:
                        (Trajectory)
        """
        # Encode skipping first steps.
        # Get a 1-elem list.
        norm_traj_list, _, _ = ts_tools.z_score_normalize_trajectory_list(
            [trajectory], self.mean_value_dict, self.std_value_dict
        )
        norm_traj_seqs = ts_tools.create_sequences_from_trajectory_list(  # Get a list of sequences.
            norm_traj_list,
            self.window_steps,
            columns=self.ae_columns
        )
        encodings = self.conv_encoder.predict(np.array(norm_traj_seqs))
        # Dimension "borrowed" from first encoding in list.
        dim_encodings = len(encodings[0])

        # Include empty first steps.
        blank_encodings = np.empty((self.window_steps - 1, dim_encodings))
        blank_encodings[:] = np.nan
        data = np.append(blank_encodings, encodings, axis=0)

        # Pack it into a Trajectory.
        new_df = pd.DataFrame(
            data=data,
            columns=[str(i) for i in range(dim_encodings)]  # Placeholder column names.
        )
        new_df['Step'] = new_df.index
        new_trajectory = ts_tools.Trajectory(df=new_df)
        return new_trajectory

    def encode_list_of_trajectories(self, traj_list, verbose=True):
        """
        Encode each of the trajectories passed.
        
        Arguments:
        'traj_list'     (list) A list of unnormalized Trajectory instances with episodes of an agent.
        'verbose'       (bool) Produce diagnotic messages.

        Returns:
                        (list) A new list of Trajectory instances with encoded MTS.
                        (DataFrame) A concatenation of all encoded trajectories skipping
                        their initial NaN steps. 
        """
        encoded_trajectories = []
        for t in traj_list:
            encoded_t = self.encode_full_trajectory(t)
            encoded_trajectories.append(encoded_t)
        all_encodings = pd.concat(
            [t.df[self.window_steps -1:] for t in encoded_trajectories],
            ignore_index=True
        ).drop('Step', axis=1)

        if verbose:
            print(f"{len(traj_list)} Trajectory instances -> {len(all_encodings)} encodings.")

        return encoded_trajectories, all_encodings

    def encode_and_reference_list_of_trajectories(self, traj_list, verbose=True):
        """
        Encode each of the trajectories passed, adding step-level reference to each trajectory.
        
        Arguments:
        'traj_list'     (list) A list of unnormalized Trajectory instances with episodes of an agent.
        'verbose'       (bool) Produce diagnotic messages.

        Returns:
                        (DataFrame) A concatenation of all encoded trajectories skipping
                        their initial NaN steps.
                        (list) A list of new Trajectory instances containing encondings at each step.
        """
        encoded_trajectories = []
        for t in traj_list:
            encoded_t = self.encode_full_trajectory(t)
            seed, episode = t.get_seed_and_episode_from_trajectory()
            encoded_t.df['Seed'] = seed
            encoded_t.df['Episode'] = episode
            encoded_trajectories.append(encoded_t)
            
        all_encodings = pd.concat(
            [t.df[self.window_steps -1:] for t in encoded_trajectories],
            ignore_index=True,
        )
        
        if verbose:
            print(f"{len(traj_list)} Trajectory instances -> {len(all_encodings)} encodings.")

        return all_encodings, encoded_trajectories

    def regenerate(self, encoding, ae_columns, denormalize=True):
        """
        Regenerate a denormalized portion of the original Trajectory from the encoding passed.
        It's used to measure the regeneration error vs the original Trajectory.

        Arguments:
        'encoding'      (np.array) A single encoding, as a list of float,
                        e.g. [4.836371 , 3.7903073, 7.117735 , 5.4358926, 4.1817636])
        'ae_columns'    (list) A list of strings with the column names for the returned sequence.
                        (This is required to associate columns to their mean and std-dev by name.)

        Returns:        (Trajectory) A new unnormalized trajectory regenerated from the encoding.

        """
        assert self.conv_decoder is not None, "Error: decoder not loaded."

        if not isinstance(encoding, np.ndarray):
            encoding = np.array(encoding)

        normalized_dec = self.conv_decoder.predict(np.array([encoding]))
        new_df = pd.DataFrame(
            data=normalized_dec[0],
            columns=ae_columns,
        )
        new_df['Step'] = new_df.index

        normalized_tr = ts_tools.Trajectory(df=new_df)
        if denormalize:
            denormalized_tr = ts_tools.invert_normalization_of_trajectory_list(
                [normalized_tr], self.mean_value_dict, self.std_value_dict
            )
            return denormalized_tr
        else:
            return [normalized_tr]


    def regenerate_full_trajectory(self, trajectory, verbose=False):
        """
        NOT IMPLEMENTED
        """
        raise NotImplemented
