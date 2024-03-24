import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
import torch
from torch.nn import ReLU
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch
import optuna


class Optimizer:
    def __init__(self, max_depth=10, max_ccp_alpha=2e-2, lmb=1e-4, n_trials=100):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.lmb = lmb
        self.n_trials = n_trials
        self.explainer = None
    
    def optimize(self, batch, model):
        datalist = batch.to_data_list()
        half = len(datalist) // 2
        train = Batch.from_data_list(datalist[:half])
        val = Batch.from_data_list(datalist[half:])
        values, aggr = _get_values(train, model)

        def objective(trial):
            params = {
                f'max_depth_{i}' : trial.suggest_int(f'max_depth_{i}', 1, self.max_depth)
                for i in range(len(values) + 1)
            } | {
                f'ccp_alpha_{i}' : trial.suggest_float(f'ccp_alpha_{i}', 0, self.max_ccp_alpha)
                for i in range(len(values) + 1)
            }
            expl = Explainer(aggr, **params).fit(train, values)
            return expl.accuracy(val) - self.lmb * (sum(
                layer.dt.tree_.node_count for layer in expl.layer) + expl.out_layer.dt.tree_.node_count)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        values, aggr = _get_values(batch, model)
        self.explainer = Explainer(aggr, **study.best_params).fit(batch, values)
        return self.explainer

    def predict(self, batch):
        return self.explainer.predict(batch)

    def accuracy(self, batch):
        return self.explainer.accuracy(batch)


class Explainer:
    def __init__(self, aggr, **params):
        self.aggr = aggr
        self.params = params
        self.layer = []
        self.out_layer = None

    def fit(self, batch, values):
        adj = to_scipy_sparse_matrix(batch.edge_index)
        x, y = batch.x.numpy(), batch.y.numpy()
        for i, (value, aggr) in enumerate(zip(values, self.aggr)):
            self.layer.append(ExplainerLayer(self.params[f'max_depth_{i}'], self.params[f'ccp_alpha_{i}']))
            x = self.layer[-1].fit_predict(x, value, adj if aggr else None)
        self.out_layer = ExplainerPoolingLayer(
            self.params[f'max_depth_{len(values)}'],
            self.params[f'ccp_alpha_{len(values)}']
        )
        y = self.out_layer.fit(x, batch.batch, y)
        return self

    def predict(self, batch):
        adj = to_scipy_sparse_matrix(batch.edge_index)
        x = batch.x.numpy()
        for layer, aggr in zip(self.layer, self.aggr):
            x = layer.predict(x, adj if aggr else None)
        return self.out_layer.predict(x, batch.batch)

    def accuracy(self, batch):
        prediction = self.predict(batch)
        return (prediction == batch.y.numpy()).mean()

    def prune(self):
        relevant = self.out_layer._relevant()


class ExplainerLayer:
    def __init__(self, max_depth, max_ccp_alpha):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.dt = None
        self.n_features_in = None
        self.leaf_indices = None
        self.leaf_values = None
        self.leaf_formulas = None
        self.aggregation = False

    def fit(self, x, y, adj=None):
        self.n_features_in = x.shape[1]
        if adj is not None:
            self.aggregation = True
            x = np.asarray(np.concatenate([
                x, adj @ x, adj @ (1 - x), (adj @ x) / (adj.sum(axis=1)).clip(1e-6, None),
            ], axis=1))
        self.dt = DecisionTreeRegressor(max_depth=self.max_depth, ccp_alpha=self.max_ccp_alpha)
        self.dt.fit(x, y)
        leaves = _leaves(self.dt.tree_)
        leaf_values = [self.dt.tree_.value[i][0][0] for i in leaves]
        if len(leaves) == 1:
            combinations = [{1}]
        else:
            self.aggregation = True
            clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
            clustering.fit(leaf_values)
            combinations = _agglomerate_labels(len(leaves), clustering)
        self.leaf_indices = np.array([leaves.index(i) if i in leaves else -1 for i in range(self.dt.tree_.node_count)])
        self.leaf_values = np.array(
            [i in combination for combination in combinations]
            for i in self.leaf_indices if i != -1
        )
        self.leaf_formulas = [
            [i for (i, b) in enumerate(self.leaf_values[j]) if b]
            for j in range(len(self.leaf_values))
        ]
        return self

    def predict(self, x, adj=None):
        if self.aggregation:
            x = np.concatenate([
                x, adj @ x, adj @ (1 - x), (adj @ x) / (adj @ np.ones(x.shape[0])).clip(1e-6, None)
            ], axis=1)
        pred = self.dt.apply(x)
        return self.leaf_values[self.leaf_indices[pred]]

    def fit_predict(self, x, y, adj=None):
        self.fit(x, y, adj)
        return self.predict(x, adj)

    def _relevant(self):
        return [feature for feature in self.dt.tree_.n_features if feature != -2]

    def _prune_irrelevant(self, relevant):
        self.leaf_formulas = [
            [i for i in formulas if i in relevant]
            for formulas in self.leaf_formulas
        ]

    def _remove_redunant(self):
        flag = True
        while flag:
            for parent, (left, right) in enumerate(zip(self.dt.tree_.children_left, self.dt.tree_.children_right)):
                if self.dt.tree_.children_left[parent] == -1 and self.dt.tree_.children_right[parent] == -1 and \
                    self.leaf_formulas[left] == self.leaf_formulas[right]:
                    self._merge_leaves(parent, left, right)
                    flag = True
                    break

    def _merge_leaves(self, parent, left, right):
        self.dt.tree_.children_left[parent] = -1
        self.dt.tree_.children_right[parent] = -1
        self.leave_indices[parent] = self.leave_indices[left]
        self.leave_indices[left] = -1
        self.leave_indices[right] = -1
        self.feature[parent] = -2
        self.threshold[parent] = -2


class ExplainerPoolingLayer:
    def __init__(self, max_depth, max_ccp_alpha):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.dt = None
        self.n_features_in = None
        self.leaf_indices = None
        self.leaf_values = None
        self.aggregation = False
    
    def fit(self, x, batch, y):
        self.n_features_in = x.shape[1]
        x = np.concatenate([
            global_mean_pool(torch.tensor(x), batch).numpy(),
            global_add_pool(torch.tensor(x), batch).numpy()
        ], axis=1)
        self.dt = DecisionTreeClassifier(max_depth=self.max_depth, ccp_alpha=self.max_ccp_alpha)
        self.dt.fit(x, y)

    def predict(self, x, batch):
        x = np.concatenate([
            global_mean_pool(torch.tensor(x), batch).numpy(),
            global_add_pool(torch.tensor(x), batch).numpy()
        ], axis=1)
        return self.dt.predict(x)

    def fit_predict(self, x, batch, y):
        self.fit(x, batch, y)
        return self.predict(x, batch)

        
def _agglomerate_labels(n_labels, clustering):
    agglomerated_features = [{i} for i in range(n_labels)]
    for i, j in clustering.children_:
        agglomerated_features.append(agglomerated_features[i] | agglomerated_features[j])
    return agglomerated_features


def _leaves(tree, node_id=0):
    if tree.children_left[node_id] == tree.children_right[node_id] and \
        (node_id in tree.children_left or node_id in tree.children_right or node_id == 0):
        return [node_id]
    else:
        left = _leaves(tree, tree.children_left[node_id])
        right = _leaves(tree, tree.children_right[node_id])
        return left + right


def _get_values(batch, model):
    values = []
    aggr = []
    hooks = []
    modules = [module for module in model.modules() if isinstance(module, ReLU) or isinstance(module, Aggregation)]
    for layer in modules:
        hooks.append(layer.register_forward_hook(
            lambda mod, inp, out: values.append(out.detach().numpy())
        ))
        hooks.append(layer.register_forward_hook(
            lambda mod, inp, out: aggr.append(isinstance(mod, Aggregation))
        ))
    with torch.no_grad():
        model(batch).detach().numpy()
    for hook in hooks:
        hook.remove()
    return values, aggr
