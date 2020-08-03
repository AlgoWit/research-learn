"""
It supports the design and execution of
machine learning experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter

from rich.progress import track
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from ..utils import check_datasets, check_oversamplers_classifiers
from ..model_selection import ModelSearchCV

GROUP_KEYS = ['Dataset', 'Oversampler', 'Classifier', 'params']


def select_results(
    imbalanced_results,
    oversamplers_names=None,
    classifiers_names=None,
    datasets_names=None,
    scoring_cols=None,
):
    """Select results of an imbalanced experiment."""

    # Check input parameters
    error_msg = (
        'Parameter `{}` should be `None` or a subset '
        'of the experiments corresponding attribute.'
    )
    if oversamplers_names is not None:
        unique_ovrs_names = imbalanced_results.reset_index()['Oversampler'].unique()
        try:
            if not set(oversamplers_names).issubset(unique_ovrs_names):
                raise ValueError(error_msg.format('oversamplers_names'))
        except TypeError:
            raise ValueError(error_msg.format('oversamplers_names'))
    if classifiers_names is not None:
        unique_clfs_names = imbalanced_results.reset_index()['Classifier'].unique()
        try:
            if not set(classifiers_names).issubset(unique_clfs_names):
                raise ValueError(error_msg.format('classifiers_names'))
        except TypeError:
            raise ValueError(error_msg.format('classifiers_names'))
    if datasets_names is not None:
        unique_ds_names = imbalanced_results.reset_index()['Dataset'].unique()
        try:
            if not set(datasets_names).issubset(unique_ds_names):
                raise ValueError(error_msg.format('datasets_names'))
        except TypeError:
            raise ValueError(error_msg.format('datasets_names'))
    if scoring_cols is not None:
        unique_scoring_cols = set([score[0] for score in imbalanced_results.columns])
        try:
            if not set(scoring_cols).issubset(unique_scoring_cols):
                raise ValueError(error_msg.format('scoring_cols'))
        except TypeError:
            raise ValueError(error_msg.format('scoring_cols'))

    # Extract results
    results = imbalanced_results.reset_index()

    # Define boolean masks
    mask_ovr = (
        results.Oversampler.isin(oversamplers_names)
        if oversamplers_names is not None
        else True
    )
    mask_clf = (
        results.Classifier.isin(classifiers_names)
        if classifiers_names is not None
        else True
    )
    mask_ds = (
        results.Dataset.isin(datasets_names) if datasets_names is not None else True
    )
    mask = mask_ovr & mask_clf & mask_ds
    if mask is True:
        mask = np.repeat(True, len(results)).reshape(-1, 1)
    else:
        mask = mask.values.reshape(-1, 1)
    if scoring_cols is None:
        scoring_cols = imbalanced_results.columns

    # Filter results
    filtered_results = imbalanced_results[mask][scoring_cols]

    return filtered_results


def combine_results(*imbalanced_results):
    """Combines the results of multiple experiments into a single one."""

    # Extract results
    results = pd.concat(imbalanced_results, axis=1, sort=True)
    if results.isna().any().any():
        scoring_cols = [scoring for scoring, _ in results.columns]
        if set(Counter(scoring_cols).values()) != {2 * len(imbalanced_results)}:
            raise ValueError(
                'Experiment with different oversamplers, classifiers or datasets '
                'should have the same scoring and vice-versa.'
            )
        values = np.apply_along_axis(
            arr=results.values,
            func1d=lambda row: row[~np.isnan(row)][: len(results.columns) // 2],
            axis=1,
        )
        results = pd.DataFrame(
            values, index=results.index, columns=results.columns[: values.shape[1]]
        )

    return results


class ImbalancedExperiment(BaseEstimator):
    """Define a classification experiment on multiple imbalanced datasets."""

    def __init__(
        self,
        oversamplers,
        classifiers,
        scoring=None,
        n_splits=5,
        n_runs=2,
        random_state=None,
        n_jobs=-1,
        verbose=0,
    ):
        self.oversamplers = oversamplers
        self.classifiers = classifiers
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_runs = n_runs
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _initialize(self):
        """Initialize experiment's parameters."""

        # Check oversamplers and classifiers
        self.estimators_, self.param_grids_ = check_oversamplers_classifiers(
            self.oversamplers, self.classifiers, self.random_state, self.n_runs
        )

        # Create model search cv
        self.mscv_ = ModelSearchCV(
            self.estimators_,
            self.param_grids_,
            scoring=self.scoring,
            refit=False,
            cv=StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            ),
            return_train_score=False,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        # Extract oversamplers and classifiers names
        self.oversamplers_names_, *_ = zip(*self.oversamplers)
        self.classifiers_names_, *_ = zip(*self.classifiers)

        # Extract scoring columns
        if isinstance(self.scoring, list):
            self.scoring_cols_ = self.scoring
        elif isinstance(self.scoring, str):
            self.scoring_cols_ = [self.scoring]
        else:
            self.scoring_cols_ = (
                ['accuracy']
                if self.mscv_.estimator._estimator_type == 'classifier'
                else ['r2']
            )

    def fit(self, datasets):
        """Fit experiment."""

        self.datasets_names_, _ = zip(*datasets)
        self._initialize()

        # Define empty results
        results = []

        # Populate results table
        datasets = check_datasets(datasets)
        for dataset_name, (X, y) in track(datasets, description='Datasets'):

            # Fit model search
            self.mscv_.fit(X, y)

            # Get results
            result = pd.DataFrame(self.mscv_.cv_results_)
            scoring_cols = [
                col for col in result.columns.tolist() if 'mean_test' in col
            ]
            result.rename(
                columns=dict(zip(scoring_cols, self.scoring_cols_)), inplace=True
            )
            result = result.loc[:, ['models', 'params'] + self.scoring_cols_]

            # Append dataset name column
            result = result.assign(Dataset=dataset_name)

            # Append result
            results.append(result)

        # Split models
        results = pd.concat(results, ignore_index=True)
        results.loc[:, 'models'] = results.loc[:, 'models'].apply(
            lambda model: model.split('|')
        )
        results[['Oversampler', 'Classifier']] = pd.DataFrame(
            results.models.values.tolist()
        )

        # Calculate results
        results.drop(columns='models', inplace=True)
        results['params'] = results['params'].apply(
            lambda param_grid: str(
                {
                    param: val
                    for param, val in param_grid.items()
                    if 'random_state' not in param
                }
            )
        )
        scoring_mapping = {
            scorer_name: [np.mean, np.std] for scorer_name in self.scoring_cols_
        }
        self.results_ = results.groupby(GROUP_KEYS).agg(scoring_mapping)

        return self
