# sklearn-genetic

Genetic feature selection module for scikit-learn

## Installation

```bash
pip install git+https://github.com/manuel-calzolari/sklearn-genetic.git
```

## Requirements
* Python >= 2.7
* scikit-learn >= 0.17.1
* DEAP >= 1.0.2

## Example

```python
import numpy as np
from sklearn import datasets, linear_model

from genetic_selection import GeneticSelectionCV


def main():
    iris = datasets.load_iris()

    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

    X = np.hstack((iris.data, E))
    y = iris.target

    estimator = linear_model.LogisticRegression()

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=True,
                                  scoring="accuracy",
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print selector.support_


if __name__ == "__main__":
    main()

```