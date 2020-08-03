"""
It contains functions to report the results of model
search and experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter
import warnings

from scipy.stats import friedmanchisquare, ttest_rel
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from sklearn.metrics import SCORERS
from sklearn.utils.validation import check_is_fitted

from ..utils import check_datasets
from ..model_selection import ModelSearchCV


def report_model_search_results(model_search_cv, sort_results=None):
    """Generate a basic model search report of results."""

    # Check model_search_cv parameter
    if not isinstance(model_search_cv, ModelSearchCV):
        raise ValueError(
            'Parameter `model_search_cv` should be a ModelSearchCV instance.'
        )

    # Check if object is fitted
    check_is_fitted(model_search_cv, 'cv_results_')

    # Select columns
    columns = ['models', 'params'] + [
        results_param
        for results_param in model_search_cv.cv_results_.keys()
        if 'mean_test' in results_param or results_param == 'mean_fit_time'
    ]

    # select results
    results = {
        results_param: values
        for results_param, values in model_search_cv.cv_results_.items()
        if results_param in columns
    }

    # Generate report table
    report = pd.DataFrame(results, columns=columns)

    # Sort results
    if sort_results is not None:

        # Use sort_results parameter as the sorting key
        try:
            report = report.sort_values(
                sort_results, ascending=(sort_results == 'mean_fit_time')
            ).reset_index(drop=True)

        # Key error
        except KeyError:

            # Define error message
            if isinstance(model_search_cv.scoring, list):
                options = ', '.join(
                    ['mean_test_%s' % sc for sc in model_search_cv.scoring]
                )
            else:
                options = 'mean_test_score'
            error_msg = (
                f'Parameter `sort_results` should be one of mean_fit_score, {options}.'
            )

            # Raise custom error
            raise KeyError(error_msg)

    return report


def summarize_datasets(datasets):
    """Create a summary of imbalanced datasets."""

    # Check datasets format
    check_datasets(datasets)

    # Define summary table columns
    summary_columns = [
        "Dataset name",
        "Features",
        "Instances",
        "Minority instances",
        "Majority instances",
        "Imbalance Ratio",
    ]

    # Define empty summary table
    datasets_summary = []

    # Populate summary table
    for dataset_name, (X, y) in datasets:
        n_instances = Counter(y).values()
        n_minority_instances, n_majority_instances = min(n_instances), max(n_instances)
        values = [
            dataset_name,
            X.shape[1],
            len(X),
            n_minority_instances,
            n_majority_instances,
            round(n_majority_instances / n_minority_instances, 2),
        ]
        datasets_summary.append(values)
    datasets_summary = pd.DataFrame(datasets_summary, columns=summary_columns)

    # Cast to integer columns
    datasets_summary[summary_columns[1:-1]] = datasets_summary[
        summary_columns[1:-1]
    ].astype(int)

    # Sort datasets summary
    datasets_summary = datasets_summary.sort_values('Imbalance Ratio').reset_index(
        drop=True
    )

    return datasets_summary


def calculate_optimal(imbalanced_results):
    """"Calculate optimal results of an imbalanced experiment across
    hyperparameters for any combination of datasets, overamplers, classifiers
    and metrics."""

    # Extract scoring columns
    mean_scoring_cols = [
        score for score in imbalanced_results.columns if score[1] == 'mean'
    ]
    scoring_cols, _ = zip(*mean_scoring_cols)

    # Select mean scores
    optimal = imbalanced_results[mean_scoring_cols]

    # Calculate maximum score per group key
    keys = ['Dataset', 'Oversampler', 'Classifier']
    agg_measures = {score: max for score in optimal.columns}
    optimal = optimal.groupby(keys).agg(agg_measures).reset_index()
    optimal.columns = optimal.columns.get_level_values(0)

    # Format as long table
    optimal = optimal.melt(
        id_vars=keys, value_vars=scoring_cols, var_name='Metric', value_name='Score'
    )

    # Cast to categorical columns
    optimal_cols = keys + ['Metric']
    for col in optimal_cols:
        optimal[col] = pd.Categorical(optimal[col], optimal[col].unique())

    # Sort values
    optimal = optimal.sort_values(optimal_cols)

    return optimal


def calculate_wide_optimal(imbalanced_results):
    """"Calculate optimal results in wide format of an imbalanced experiment across
    hyperparameters for any combination of datasets, overamplers, classifiers
    and metrics."""

    # Format as wide table
    wide_optimal = calculate_optimal(imbalanced_results).pivot_table(
        index=['Dataset', 'Classifier', 'Metric'],
        columns=['Oversampler'],
        values='Score',
    )
    wide_optimal.columns = wide_optimal.columns.tolist()
    wide_optimal.reset_index(inplace=True)

    return wide_optimal


def _return_row_ranking(row, sign):
    """Returns the ranking of values. In case of tie, each ranking value
    is replaced with its group average."""

    # Calculate ranking
    ranking = (sign * row).argsort().argsort().astype(float)

    # Break the tie
    groups = np.unique(row, return_inverse=True)[1]
    for group_label in np.unique(groups):
        indices = groups == group_label
        ranking[indices] = ranking[indices].mean()

    return ranking.size - ranking


def calculate_ranking(imbalanced_results):
    """Calculate the ranking of oversamplers for
    any combination of datasets, classifiers and
    metrics."""
    wide_optimal = calculate_wide_optimal(imbalanced_results)
    ranking_results = wide_optimal.apply(
        lambda row: _return_row_ranking(
            row[3:], SCORERS[row[2].replace(' ', '_').lower()]._sign
        ),
        axis=1,
    )
    ranking = pd.concat([wide_optimal.iloc[:, :3], ranking_results], axis=1)
    return ranking


def calculate_mean_sem_ranking(imbalanced_results):
    """Calculate the mean and standard error of oversamplers' ranking
    across datasets for any combination of classifiers
    and metrics."""
    ranking = calculate_ranking(imbalanced_results)
    mean_ranking = ranking.groupby(['Classifier', 'Metric']).mean().reset_index()
    sem_ranking = (
        ranking.drop(columns='Dataset')
        .groupby(['Classifier', 'Metric'])
        .sem()
        .reset_index()
    )
    return mean_ranking, sem_ranking


def calculate_mean_sem_scores(imbalanced_results):
    """Calculate mean and standard error of scores across datasets."""
    wide_optimal = calculate_wide_optimal(imbalanced_results)
    mean_scores = wide_optimal.groupby(['Classifier', 'Metric']).mean().reset_index()
    sem_scores = (
        wide_optimal.drop(columns='Dataset')
        .groupby(['Classifier', 'Metric'])
        .sem()
        .reset_index()
    )
    return mean_scores, sem_scores


def calculate_mean_sem_perc_diff_scores(imbalanced_results, compared_oversamplers=None):
    """Calculate mean and standard error scores' percentage difference."""

    # Calculate wide optimal results
    wide_optimal = calculate_wide_optimal(imbalanced_results)
    ovrs_names = wide_optimal.columns[3:]

    # Calculate percentage difference only for more than one oversampler
    if len(ovrs_names) < 2:
        warnings.warn(
            'More than one oversampler is required to '
            'calculate the mean percentage difference.'
        )

    # Extract oversamplers
    control, test = (
        compared_oversamplers if compared_oversamplers is not None else ovrs_names[-2:]
    )

    # Calculate percentage difference
    scores = wide_optimal[wide_optimal[control] > 0]
    perc_diff_scores = pd.DataFrame(
        (100 * (scores[test] - scores[control]) / scores[control]),
        columns=['Difference'],
    )
    perc_diff_scores = pd.concat([scores.iloc[:, :3], perc_diff_scores], axis=1)

    # Calulate mean and std percentage difference
    mean_perc_diff_scores = (
        perc_diff_scores.groupby(['Classifier', 'Metric']).mean().reset_index()
    )
    sem_perc_diff_scores = (
        perc_diff_scores.drop(columns='Dataset')
        .groupby(['Classifier', 'Metric'])
        .sem()
        .reset_index()
    )

    return mean_perc_diff_scores, sem_perc_diff_scores


def _extract_pvalue(df):
    """Extract the p-value."""
    results = friedmanchisquare(*df.iloc[:, 3:].transpose().values.tolist())
    return results.pvalue


def apply_friedman_test(imbalanced_results, alpha=0.05):
    """Apply the Friedman test across datasets for every
    combination of classifiers and metrics."""

    # Calculate ranking results
    ranking = calculate_ranking(imbalanced_results)
    ovrs_names = ranking.columns[3:]

    # Apply test for more than two oversamplers
    if len(ovrs_names) < 3:
        warnings.warn(
            'More than two oversamplers are required apply the Friedman test.'
        )

    # Calculate p-values
    friedman_test = (
        ranking.groupby(['Classifier', 'Metric'])
        .apply(_extract_pvalue)
        .reset_index()
        .rename(columns={0: 'p-value'})
    )

    # Compare p-values to significance level
    friedman_test['Significance'] = friedman_test['p-value'] < alpha

    return friedman_test


def apply_holms_test(imbalanced_results, control_oversampler=None):
    """Use the Holm's method to adjust the p-values of a paired difference
    t-test for every combination of classifiers and metrics using a control
    oversampler."""

    # Calculate wide optimal results
    wide_optimal = calculate_wide_optimal(imbalanced_results)
    ovrs_names = list(wide_optimal.columns[3:])

    # Apply test for more than one oversampler
    if len(ovrs_names) < 2:
        warnings.warn('More than one oversampler is required to apply the Holms test.')

    # Use the last if no control oversampler is provided
    if control_oversampler is None:
        control_oversampler = ovrs_names[-1]
    ovrs_names.remove(control_oversampler)

    # Define empty p-values table
    pvalues = pd.DataFrame()

    # Populate p-values table
    for name in ovrs_names:
        pvalues_pair = wide_optimal.groupby(['Classifier', 'Metric'])[
            [name, control_oversampler]
        ].apply(lambda df: ttest_rel(df[name], df[control_oversampler])[1])
        pvalues_pair = pd.DataFrame(pvalues_pair, columns=[name])
        pvalues = pd.concat([pvalues, pvalues_pair], axis=1)

    # Corrected p-values
    holms_test = pd.DataFrame(
        pvalues.apply(
            lambda col: multipletests(col, method='holm')[1], axis=1
        ).values.tolist(),
        columns=ovrs_names,
    )
    holms_test = holms_test.set_index(pvalues.index).reset_index()

    return holms_test
