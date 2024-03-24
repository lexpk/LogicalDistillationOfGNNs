import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import AgglomerativeClustering
import torch
from torch.nn import ReLU
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
import optuna


class OptimizingExplainer:
    def __init__(self, max_depth=10, max_ccp_alpha=2e-2, lmbd=1e-4, n_trials=100, callbacks=[]):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.lmbd = lmbd
        self.n_trials = n_trials
        self.callbacks = callbacks
        self.explainer = None
    
    def fit(self, batch, model):
        datalist = batch.to_data_list()
        half = len(datalist) // 2
        train = Batch.from_data_list(datalist[:half])
        val = Batch.from_data_list(datalist[half:])
        values, aggr = get_values(train, model)

        def objective(trial):
            params = {
                f'max_depth_{i}' : trial.suggest_int(f'max_depth_{i}', 1, self.max_depth)
                for i in range(len(values) + 1)
            } | {
                f'ccp_alpha_{i}' : trial.suggest_float(f'ccp_alpha_{i}', 0, self.max_ccp_alpha)
                for i in range(len(values) + 1)
            }
            expl = Explainer(aggr, **params).fit(train, values)
            return expl.accuracy(val) - self.lmbd * (sum(
                dt.tree_.node_count for dt in expl.dt) + expl.out_dt.tree_.node_count)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, callbacks=self.callbacks, show_progress_bar=True)

        values, aggr = get_values(batch, model)
        self.explainer = Explainer(aggr, **study.best_params).fit(batch, values)
        return self

    def predict(self, batch):
        return self.explainer.predict(batch)

    def accuracy(self, batch):
        return self.explainer.accuracy(batch)


class Explainer:
    def __init__(self, aggr, **params):
        self.aggr = aggr
        self.params = params
        self.dt = None
        self.clustering = None
        self.out_dt = None

    def fit(self, batch, values):
        adj = to_scipy_sparse_matrix(batch.edge_index)
        feature_set = FeatureSet(batch.x.numpy())
        self.dt = []
        self.clustering = []
        for i, (value, aggr) in enumerate(zip(values, self.aggr)):
            if aggr:
                feature_set.deepen(adj)
                dt, clustering = feature_set.iterate(
                    value, max_depth=self.params[f'max_depth_{i}'], ccp_alpha=self.params[f'ccp_alpha_{i}'])
            else:
                dt, clustering = feature_set.iterate(
                    value, max_depth=self.params[f'max_depth_{i}'], ccp_alpha=self.params[f'ccp_alpha_{i}'])
            self.dt.append(dt)
            self.clustering.append(clustering)
        feature_set.mean_pool(batch.batch)
        self.out_dt = DecisionTreeClassifier(
            max_depth=self.params[f'max_depth_{len(self.aggr)}'],
            ccp_alpha=self.params[f'ccp_alpha_{len(self.aggr)}']
        ).fit(feature_set.x, batch.y.numpy())
        return self

    def predict(self, batch):
        adj = to_scipy_sparse_matrix(batch.edge_index)
        feature_set = FeatureSet(batch.x.numpy())
        for dt, clustering, aggr in zip(self.dt, self.clustering, self.aggr):
            if aggr:
                feature_set.deepen(adj)
            feature_set.apply(dt, clustering)
        feature_set.mean_pool(batch.batch)
        result = self.out_dt.predict(feature_set.x)
        return result

    def accuracy(self, batch):
        prediction = self.predict(batch)
        y = batch.y.numpy()
        return (prediction == y).mean()


class ExplainerLayer:
    def __init__(self, max_depth, max_ccp_alpha):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.dt = None
        self.n_features_in = None
        self.leaf_indices = None
        self.combinations = None
        self.leaf_values = None
        self.aggregation = False

    def fit(self, x, y, adj=None):
        self.n_features_in = x.shape[1]
        if adj is not None:
            self.aggregation = True
            x = np.concatenate([
                x, adj @ x, adj @ (1 - x), (adj @ x) / (adj.sum(axis=1)).clip(1e-6, None),
            ], axis=1)
        self.dt = DecisionTreeRegressor(max_depth=self.max_depth, ccp_alpha=self.max_ccp_alpha)
        self.dt.fit(x, y)
        leaves = _leaves(self.dt.tree_)
        if len(self.leaf_indices) == 1:
            self.combinations = [{1}]
        else:
            self.aggregation = True
            clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
            clustering.fit(self.leaf_values)
            self.combinations = _agglomerate_labels(len(leaves), clustering)
        self.leaf_indices = [leaves.index(i) if i in leaves else -1 for i in range(self.dt.tree_.node_count)]
        self.leaf_values = np.array(
            [i in combination for combination in self.combinations]
            for i in self.leaf_indices if i != -1
        )
        return self

    def predict(self, x, adj=None):
        if self.aggregation:
            x = np.concatenate([
                x, adj @ x, adj @ (1 - x), (adj @ x) / (adj.sum(axis=1)).clip(1e-6, None),
            ], axis=1)
        pred = self.dt.apply(x)
        return self.leaf_values[self.leaf_indices[pred]]

    def fit_predict(self, x, y, adj=None):
        self.fit(x, y, adj)
        return self.predict(x, adj)

        
def _agglomerate_labels(n_labels, clustering):
    agglomerated_features = [{i} for i in range(n_labels)]
    for i, j in clustering.children_:
        agglomerated_features.append(agglomerated_features[i] | agglomerated_features[j])
    return agglomerated_features


class FeatureSet:
    def __init__(self, x):
        self.x = x

    def apply(self, dt, clustering):
        leaves = _leaves(dt.tree_)
        pred = dt.apply(self.x)
        features = [pred == leaf for leaf in leaves]
        features = agglomerate_features(clustering, features)
        self.x = np.array(features).T
        return self.x

    def iterate(self, y, max_depth=7, ccp_alpha=0.0):
        new_features, dt, clustering = _iterate(self.x, y, max_depth=max_depth, ccp_alpha=ccp_alpha)
        self.x = new_features
        return dt, clustering

    def deepen(self, adj):
        x_neigh = adj @ self.x
        not_x_neigh = adj @ (1 - self.x)
        self.x = np.concatenate([self.x, x_neigh, not_x_neigh], axis=1)

    def mean_pool(self, batch):
        self.x = global_mean_pool(torch.tensor(self.x), batch).numpy()

    def __getitem__(self, idx):
        return self.x[idx]


def _iterate(x, y, max_depth=7, ccp_alpha=0.0):
    dt = DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha)
    dt.fit(x, y)
    leaves = _leaves(dt.tree_)
    if len(leaves) == 1:
        return np.ones((x.shape[0], 1)), dt, TrivialClustering()
    dt_out = dt.apply(x)
    new_features = [dt_out == leaf for leaf in leaves]
    clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    clustering.fit(dt.tree_.value[leaves].squeeze(axis=2))
    new_features = agglomerate_features(clustering, new_features)
    
    return np.array(new_features).T, dt, clustering


def _leaves(tree, node_id=0):
    if tree.children_left[node_id] == tree.children_right[node_id] and \
        (node_id in tree.children_left or node_id in tree.children_right or node_id == 0)
        return [node_id]
    else:
        left = _leaves(tree, tree.children_left[node_id])
        right = _leaves(tree, tree.children_right[node_id])
        return left + right


def get_values(batch, model):
    values = []
    hooks = []
    modules = [module for module in model.modules() if isinstance(module, ReLU) or isinstance(module, Aggregation)]
    for layer in modules:
        hooks.append(layer.register_forward_hook(
            lambda mod, inp, out: values.append(out.detach().numpy())
        ))
    with torch.no_grad():
        model(batch).detach().numpy()
    for hook in hooks:
        hook.remove()
    return values, [isinstance(module, Aggregation) for module in modules]


class TrivialClustering:
    def __init__(self):
        self.n_clusters = 1
        self.children_ = np.array([])

    def fit(self, x):
        return self

    def fit_predict(self, x):
        return np.zeros(x.shape[0])

    def predict(self, x):
        return np.zeros(x.shape[0])

