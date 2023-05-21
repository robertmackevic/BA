from typing import Dict

import numpy as np

from src.config import SCORES


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    results = {}
    for score in SCORES:
        results[score.__name__] = score(y_true, y_pred)

    if verbose:
        print(results)
        print('\n')

    return results
