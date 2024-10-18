import pandas as pd
import numpy as np
from toolz import curry

from typing import List, Dict, Optional, Any, Callable


wise_colors = dict(
    dark_green='#163300',
    light_green='#9FE870',
    orange='#FFC091',
    yellow='#FFEB69',
    blue='#A0E1E1',
    pink='#FFD7EF'
)


@curry
def linear_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: str = None
):
    """
    Computes the linear coefficient from regressing the outcome on the prediction

    params
    ------
    test_data: pd.DataFrame
        A pandas DataFrame with with target and prediction.
    prediction_column: str
        The name of the column in `test_data` with the prediction.
    target_column: str
        The name of the column in `test_data` with the continuous target.
    eval_name: Optional[Union[str, None]]
        the name of the evaluator as it will appear in the logs.
    
    return
    ------
    log: dict
        a log-like dictionary with the linear coefficient from regressing the outcome on the prediction
    """

    if eval_name is None:
        eval_name = "linear_coefficient_evaluator__" + target_column

    cov_mat = test_data[[prediction_column, target_column]].cov()

    A = cov_mat.iloc[0, 1] 
    B = cov_mat.iloc[0, 0]

    score = np.divide(A, B, out=np.zeros_like(A), where=B!=0)

    return {eval_name: score}

def _apply_effect(
    evaluator: Callable[..., Dict[str, float]],
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str
) -> float:
    return evaluator(df, treatment_column, outcome_column, eval_name="effect")["effect"]

@curry
def linear_effect(df: pd.DataFrame, treatment_column: str, outcome_column: str) -> float:
    """
    Computes the linear coefficient from regressing the outcome on the treatment: cov(outcome, treatment)/var(treatment)
    params
    ------
    df: pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.
    treatment_column: str
        The name of the treatment column in `df`.
    outcome_column: str
        The name of the outcome column in `df`
    return
    ------
    effect: float
        The linear coefficient from regressing the outcome on the treatment: cov(outcome, treatment)/var(treatment)
    """
    return _apply_effect(linear_coefficient_evaluator, df, treatment_column, outcome_column)


@curry
def cumulative_effect_curve(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    prediction: str,
    min_rows: int = 30,
    steps: int = 100,
    effect_fn = linear_effect
) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the cumulative effect curve according to that ordering
    
    params
    ------
    df: pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.
    treatment: str
        The name of the treatment column in `df`.
    outcome: str
        The name of the outcome column in `df`.
    prediction: str
        The name of the prediction column in `df`.
    min_rows: int
        Minimum number of observations needed to have a valid result.
    steps: int
        The number of cumulative steps to iterate when accumulating the effect
    effect_fn: function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or List[int]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    return
    ------
    cumulative effect curve: np.array
        The cumulative treatment effect according to the predictions ordering.
    """
    size = df.shape[0]
    ordered_df = df.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    
    return np.array([
        effect_fn(ordered_df.head(rows), treatment, outcome) 
        for rows in n_rows
    ])


@curry
def relative_cumulative_gain_curve(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    prediction: str,
    min_rows: int = 30,
    steps: int = 100,
    effect_fn = linear_effect
) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the relative cumulative gain curve curve according to that ordering.
    The relative gain is simply the cumulative effect minus the Average Treatment Effect (ATE) times the relative
    sample size.
    
    params
    ------
    df: pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.
    treatment: str
        The name of the treatment column in `df`.
    outcome: str
        The name of the outcome column in `df`.
    prediction: str
        The name of the prediction column in `df`.
    min_rows: int
        Minimum number of observations needed to have a valid result.
    steps: int
        The number of cumulative steps to iterate when accumulating the effect
    effect_fn : function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or List[int]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    return
    ------
    relative cumulative gain curve: float
        The relative cumulative gain according to the predictions ordering.
    """

    ate = effect_fn(df, treatment, outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(
        df=df, 
        treatment=treatment, 
        outcome=outcome, 
        prediction=prediction,
        min_rows=min_rows, 
        steps=steps, 
        effect_fn=effect_fn
    )

    return np.array([
        (effect - ate) * (rows / size) 
        for rows, effect in zip(n_rows, cum_effect)
    ])

@curry
def area_under_the_relative_cumulative_gain_curve(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    prediction: str,
    min_rows: int = 30,
    steps: int = 100,
    effect_fn = linear_effect
) -> float:
    """
    Orders the dataset by prediction and computes the area under the relative cumulative gain curve, according to that
    ordering.
    
    params
    ------
    df : pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.
    treatment: str
        The name of the treatment column in `df`.
    outcome: str
        The name of the outcome column in `df`.
    prediction: str
        The name of the prediction column in `df`.
    min_rows: int
        Minimum number of observations needed to have a valid result.
    steps: int
        The number of cumulative steps to iterate when accumulating the effect
    effect_fn: function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or Array of int
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.
    return
    ------
    area under the relative cumulative gain curve: float
        The area under the relative cumulative gain curve according to the predictions ordering.
    """

    ate = effect_fn(df, treatment, outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    step_sizes = [min_rows] + [t - s for s, t in zip(n_rows, n_rows[1:])]

    cum_effect = cumulative_effect_curve(
        df=df, 
        treatment=treatment, 
        outcome=outcome, 
        prediction=prediction,
        min_rows=min_rows, 
        steps=steps, 
        effect_fn=effect_fn
    )

    return abs(
        sum([
            (effect - ate) * (rows / size) * (step_size / size)
            for rows, effect, step_size in zip(n_rows, cum_effect, step_sizes)
        ])
    )
