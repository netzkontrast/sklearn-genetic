# sklearn-genetic - Genetic feature selection module for scikit-learn
# Copyright (C) 2016  Manuel Calzolari
#
# modified (C) 2018 Michael Schimmer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Genetic algorithm for feature selection"""

import multiprocessing
import random
import copy
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.feature_selection.base import SelectorMixin
from sklearn.externals.joblib import cpu_count
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from functools import partial
import signal
import sys
import joblib
import os
import time

creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)


def _init_selected_features(n_features=None, lower_percentage=5, higher_percentage=60):
    if n_features is not None:
        percentage = np.random.randint(lower_percentage, higher_percentage)
        k = n_features * percentage // 100
        indices = random.sample(range(n_features), k)
        feat_gene = np.zeros(n_features, dtype=bool)
        feat_gene[indices] = True
        return feat_gene
    raise ValueError('we need the total feature counr')


def _eval_function(individual, gaobject, estimator, X, y, cv,
                   scorer, verbose, fit_params, caching, test_data=None):
    individual_sum = np.sum(individual, axis=0)
    if individual_sum == 0:
        return -10000, individual_sum
    individual_tuple = tuple(individual)
    if caching and individual_tuple in gaobject.scores_cache:
        return gaobject.scores_cache[individual_tuple], individual_sum
    x_selected = X[:, np.array(individual, dtype=np.bool)]
    scores = []

    if fit_params['eval_set'] is not None:
        eval_set_params = copy.deepcopy(fit_params)
        for i, valid_data in enumerate(eval_set_params['eval_set']):
            x_eval, y_eval = check_X_y(valid_data[0], valid_data[1], "csr")
            x_eval_selected = x_eval[:, np.array(individual, dtype=np.bool)]
            eval_set_params['eval_set'][i][0] = x_eval_selected
    else:
        eval_set_params = fit_params

    fold = 0
    test_selected = test_data[:, np.array(individual, dtype=np.bool)]
    oof_train = np.zeros((x_selected.shape[0],))
    oof_test = np.zeros((test_selected.shape[0],))
    oof_test_skf = np.empty((cv.get_n_splits, test_selected.shape[0]))

    for train, test in cv.split(X, y):
        score = _fit_and_score(estimator=estimator, X=x_selected, y=y, scorer=scorer,
                               train=train, test=test, verbose=verbose, parameters=None,
                               fit_params=eval_set_params)
        scores.append(score)

        # if it is not empty - we want oof predictions
        if test_data is not None:
            oof_train[test] = estimator.booster_.predict(x_selected[test],
                                                         num_iteration=estimator.best_iteration_)
            oof_test_skf[fold, :] = estimator.booster_.predict(test_selected,
                                                               num_iteration=estimator.best_iteration_)
            fold += 1

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_train = oof_train.reshape(-1, 1)
    oof_test = oof_test.reshape(-1, 1)
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)

    data_dict = {
        'holdout_score': float(estimator.best_score_['oof']['auc']),
        'oof_score': scorer(y, oof_train),
        'oof_test_folds': oof_test_skf,
        'oof_train': oof_train,
        'oof_test_mean': oof_test,
        'estimator_params': estimator.get_params(),
        'estimator_feature_importance': estimator.feature_importances_,
        'estimator_best_iteration': int(estimator.best_iteration_),
        'estimator_n_features_': estimator.n_features_,
        'original_n_features': X.shape[0],
        'cv_scores': scores,
        'cv_score': scores_mean,
        'cv_score_std': scores_std,
        'folds': fold,
        'individual': individual,
        'individual_hash': str(hash(individual)),
        'time': time.time()
    }

    name = '{:.5f}_{}_{}_oof_data'.format(
        data_dict['holdout_score'],
        data_dict['time'],
        data_dict['individual_hash']
    )

    save_oof_predictions(name, data_dict)

    if caching:
        gaobject.scores_cache[individual_tuple] = scores_mean
        if random.randint(0, 4) == 0:
            filename = os.path.join(os.getcwd(), 'cache.z')
            joblib.dump(gaobject.scores_cache, filename, compress=True)
    return scores_mean, individual_sum


def save_oof_predictions(name, data):
    working_dir = os.getcwd()
    working_dir = os.path.join(working_dir, 'oof_predictions')
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    joblib.dump(data, os.path.join(working_dir, name + '.z'), compress=True)


class GeneticSelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    n_population : int, default=300
        Number of population for the genetic algorithm.

    crossover_proba : float, default=0.5
        Probability of crossover for the genetic algorithm.

    mutation_proba : float, default=0.2
        Probability of mutation for the genetic algorithm.

    n_generations : int, default=40
        Number of generations for the genetic algorithm.

    crossover_independent_proba : float, default=0.1
        Independent probability of crossover for the genetic algorithm.

    mutation_independent_proba : float, default=0.05
        Independent probability of mutation for the genetic algorithm.

    tournament_size : int, default=3
        Tournament size for the genetic algorithm.

    caching : boolean, default=False
        If True, scores of the genetic algorithm are cached.

    filename : string, default='genetics.z'
        filename of restore file

    restore : boolean, default=False
        If True, and there is a file in current directory, we restore
        the session.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    generation_scores_ : array of shape [n_generations]
        The maximum cross-validation score for each generation.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    An example showing genetic feature selection.

    >>> import numpy as np
    >>> from sklearn import datasets, linear_model
    >>> from genetic_selection import GeneticSelectionCV
    >>> iris = datasets.load_iris()
    >>> E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    >>> X = np.hstack((iris.data, E))
    >>> y = iris.target
    >>> estimator = linear_model.LogisticRegression()
    >>> selector = GeneticSelectionCV(estimator, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True  True  True  True False False False False False False False False
           False False False False False False False False False False False False], dtype=bool)
    """

    def __init__(self, estimator, cv=None, scoring=None, fit_params=None, verbose=0, n_jobs=1,
                 n_population=300, crossover_proba=0.5, mutation_proba=0.2, n_generations=40,
                 crossover_independent_proba=0.1, mutation_independent_proba=0.05,
                 tournament_size=3, caching=False, filename='genetics.z', restore=True,
                 test_data=None
                 ):

        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_population = n_population
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.n_generations = n_generations
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba = mutation_independent_proba
        self.tournament_size = tournament_size
        self.caching = caching
        self.scores_cache = {}
        self.filename = filename
        self.test_data = test_data
        self.restore = restore

        if self.caching:
            cache_file_name = self.get_storage_path('cache.z')
            if restore and os.path.isfile(cache_file_name):
                self.scores_cache = joblib.load(cache_file_name)
                if self.verbose:
                    print('{} cache entries restored'
                          .format(len(self.scores_cache)))

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @staticmethod
    def get_storage_path(filename):
        return os.path.join(os.getcwd(), filename)

    def save(self):
        if self.caching:
            filename = self.get_storage_path('cache.z')
            joblib.dump(self.scores_cache, filename, compress=True)

        joblib.dump(self, self.get_storage_path(self.filename), compress=True)

    @staticmethod
    def restore(filename):
        print('restoring')
        object = joblib.load(GeneticSelectionCV.get_storage_path(filename))
        print('{} individuals restored'.format(len(object.scores_cache)))
        return object

    def signal_handler(self, sig, frame):
        print('WAIT, we store what we have done so far')
        self.save()
        sys.exit(0)

    def fit(self, X, y):
        """Fit the GeneticSelectionCV model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        return self._fit(X, y)

    def _fit(self, X, y):
        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        estimator = clone(self.estimator)

        # Genetic Algorithm
        toolbox = base.Toolbox()
        init_features = partial(_init_selected_features, n_features=n_features)
        toolbox.register("individual", tools.initIterate, creator.Individual, init_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _eval_function, gaobject=self, estimator=estimator, X=X, y=y,
                         cv=cv, scorer=scorer, verbose=self.verbose, fit_params=self.fit_params,
                         caching=self.caching, test=self.test_data)
        toolbox.register("mate", tools.cxUniform, indpb=self.crossover_independent_proba)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_independent_proba)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        if self.n_jobs > 1:
            pool = multiprocessing.Pool(processes=self.n_jobs)
            toolbox.register("map", pool.map)
        elif self.n_jobs < 0:
            pool = multiprocessing.Pool(processes=max(cpu_count() + 1 + self.n_jobs, 1))
            toolbox.register("map", pool.map)

        pop = toolbox.population(n=self.n_population)
        hof = tools.HallOfFame(5, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", self.rounded_mean)
        stats.register("std", self.rounded_std)
        stats.register("min", self.rounded_min)
        stats.register("max", self.rounded_max)

        if self.verbose > 0:
            print("Selecting features with genetic algorithm.")

        _, log = algorithms.eaSimple(pop, toolbox, cxpb=self.crossover_proba,
                                     mutpb=self.mutation_proba, ngen=self.n_generations,
                                     stats=stats, halloffame=hof, verbose=self.verbose)
        if self.n_jobs != 1:
            pool.close()
            pool.join()

        print('done')
        # Set final attributes
        support_ = np.array(hof, dtype=np.bool)[0]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, support_], y)

        self.generation_scores_ = np.array([score for score, _ in log.select("max")])
        self.n_features_ = support_.sum()
        self.support_ = support_

        return self

    @staticmethod
    def rounded_std(value, decimals=6):
        std = np.std(value, axis=0)
        return [np.round(std[0], decimals=decimals), np.round(std[1], decimals=1)]

    @staticmethod
    def rounded_mean(value, decimals=6):
        mean = np.mean(value, axis=0)
        return [np.round(mean[0], decimals=decimals), int(mean[1])]

    @staticmethod
    def rounded_min(value, decimals=6):
        min = np.min(value, axis=0)
        return [np.round(min[0], decimals=decimals), int(min[1])]

    @staticmethod
    def rounded_max(value, decimals=6):
        max = np.max(value, axis=0)
        return [np.round(max[0], decimals=decimals), int(max[1])]

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))
