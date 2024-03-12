import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.base import BaseEstimator
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.aggr import Aggregation
import optuna

class OptimizingExplainer:
    def __init__(self, n_trials=10, n_cluster_max=20):
        self.n_trials = n_trials
        self.n_cluster_max = n_cluster_max
        self.expl = None

    def fit(self, data, model):
        n_leaves = sum(1 for layer in model.modules() if len(list(layer.children())) == 0)
        
        def objective(trial):
            params = {
                f'n_clusters_{i}' : trial.suggest_int(f'n_clusters_{i}', 2, self.n_cluster_max)
                for i in range(n_leaves)
            }
            expl = Explainer(**params).fit(data, model)
            return expl.accuracy(data, split='val')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        self.expl = Explainer(**study.best_params).fit(data, model)
        return self
        
    def predict(self, data):
        return self.expl.predict(data)
    
    def accuracy(self, data, split='full'):
        return self.expl.accuracy(data, split=split)
        


class Explainer:
    def __init__(self, **params):
        self.params = params

    def fit(self, data, model):
        train_mask = data.train_mask.numpy()
        adj = to_scipy_sparse_matrix(data.edge_index)
        values, agg = values_agg_in_order(model, data)
        self.agg = agg
        self.clusterings = get_clusterings(values, **self.params)
        self.agglomerate_concepts, self.agglomerate_explanation = \
            agglomerate_concepts(self.clusterings, **self.params)
        self.agglomerate_concepts = [np.array(concepts).T for concepts in self.agglomerate_concepts]
        
        if agg[0]:
            raise ValueError('First layer cannot be aggregation layer')
        self.nb = [
            CategoricalNB(min_categories=2).fit(
                data.x.numpy(),
                self.clusterings[0].labels_
            )
        ]
        for i in range(1, len(self.clusterings)):
            x = np.concatenate(
                [data.x.numpy()] + self.agglomerate_concepts[:i], axis=-1
            )
            if agg[i]:
                cat_mask = np.array([True] * x.shape[1] + [False] * x.shape[1] * 2)
                int_mask = np.array([False] * x.shape[1] + [True] * x.shape[1] * 2)
                x = np.concatenate(
                    [x, adj @ x, adj @ (1 - x)], axis=-1
                )
                self.nb.append(
                    MixedNB(cat_mask, int_mask, min_categories=2).fit(
                        x[train_mask], self.clusterings[i].labels_[train_mask]
                    )
                )
            else:
                self.nb.append(
                    CategoricalNB(min_categories=2).fit(
                        x[train_mask], self.clusterings[i].labels_[train_mask]
                    )
                )
        x = np.concatenate(
            [data.x.numpy()] + self.agglomerate_concepts, axis=-1
        )
        self.nb.append(
            CategoricalNB(min_categories=2).fit(
                x[train_mask], data.y.numpy()[train_mask]
            )
        )
        return self

    def predict(self, data):
        adj = to_scipy_sparse_matrix(data.edge_index)
        x = data.x.numpy()
        for i in range(len(self.nb) - 1):
            x_temp = x
            if self.agg[i]:
                x_temp = np.concatenate(
                    [x, adj @ x, adj @ (1 - x)], axis=-1
                )
            pred = self.nb[i].predict(x_temp)
            x_new = np.array([
                [
                    prediction in explanation
                    for explanation in self.agglomerate_explanation[i]  
                ]
                for prediction in pred
            ])
            x = np.concatenate([x, x_new], axis=-1)
        return self.nb[-1].predict(x)

    def accuracy(self, data, split='full'):
        if split == 'train':
            mask = data.train_mask.numpy()
        elif split == 'val':
            mask = data.val_mask.numpy()
        elif split == 'test':
            mask = data.test_mask.numpy()
        elif split == 'full':
            mask = np.ones(data.y.size(0), dtype=bool)
        return (self.predict(data) == data.y.numpy())[mask].mean()



def concepts(clusterings, **params):
    return [
        [
            np.array(clustering.labels_ == label)
            for label in range(params[f'n_clusters_{i}'])
        ]
        for i, clustering in enumerate(clusterings)
    ]


def agglomerate_concepts(clusterings, **params):
    agglomerations = [
        AgglomerativeClustering(
            n_clusters=2, linkage='single'
        ).fit(clustering.cluster_centers_)
        for i, clustering in enumerate(clusterings)
    ]
    result = concepts(clusterings, **params)
    explanation = [
        [{label} for label in range(params[f'n_clusters_{i}'])]
        for i in range(len(clusterings))
    ]
    for i, agglomeration in enumerate(agglomerations):
        for a, b in agglomeration.children_:
            result[i].append(
                np.logical_or(result[i][a], result[i][b])
            )
            explanation[i].append(
                explanation[i][a] | explanation[i][b]
            )
    return result, explanation


def get_clusterings(values, **params):
    standardized = [
        StandardScaler().fit_transform(value) for value in values
    ]
    clusterings = [
        MiniBatchKMeans(
            n_clusters=params[f'n_clusters_{i}'], n_init='auto'
        ).fit(standardized)
        for i, standardized in enumerate(standardized)
    ]
    return [clustering for clustering in clusterings]


def values_agg_in_order(model, input):
    values = []
    agg = []
    modules = leaf_modules_in_order(model, input)
    hooks = []
    for module in modules:
        hooks += [module.register_forward_hook(
            lambda self, input, output: values.append(output.numpy())
        ), module.register_forward_hook(
            lambda self, input, output: agg.append(isinstance(self, Aggregation))
        )]
    model.eval()
    with torch.no_grad():
        model(input)
    for hook in hooks:
        hook.remove()
    return values, agg


def leaf_modules_in_order(model, input):
    hooks = []
    modules_in_order = []
    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(
                module.register_forward_hook(
                    lambda self, *_: modules_in_order.append(self)
                )
            )
    with torch.no_grad():
        model(input)
    for hook in hooks:
        hook.remove()
    return modules_in_order


class MixedNB(BaseEstimator):
    def __init__(self, categorical_mask, integral_mask, min_categories=None):
        super().__init__()
        self.min_categories = min_categories
        self.categorical_mask = categorical_mask
        self.has_categorical = np.any(categorical_mask)
        self.integral_mask = integral_mask
        self.has_integral = np.any(integral_mask)
        if not (self.has_categorical or self.has_integral):
            raise ValueError('At least one feature must be used')

    def fit(self, X, y):
        if self.has_categorical:
            self.categorical_nb = CategoricalNB(min_categories=self.min_categories)
            self.categorical_nb.fit(X[:, self.categorical_mask], y)
            self.classes_ = self.categorical_nb.classes_
            self.class_count_ = self.categorical_nb.class_count_
            self.class_log_prior_ = self.categorical_nb.class_log_prior_
        if self.has_integral:
            self.integral_nb = MultinomialNB()
            self.integral_nb.fit(X[:, self.integral_mask], y)
            self.classes_ = self.integral_nb.classes_
            self.class_log_prior_ = self.integral_nb.class_log_prior_
            self.class_count_ = self.integral_nb.class_count_
        return self

    def predict_log_proba(self, X):
        log_proba = np.zeros((X.shape[0], self.classes_.shape[0]))
        if self.has_categorical:
            log_proba += self.categorical_nb.predict_log_proba(X[:, self.categorical_mask])
        if self.has_integral:
            log_proba += self.integral_nb.predict_log_proba(X[:, self.integral_mask])
        if self.has_categorical and self.has_integral:
            log_proba /= self.class_log_prior_
        return log_proba

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)
        