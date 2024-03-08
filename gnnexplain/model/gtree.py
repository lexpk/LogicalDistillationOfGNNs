import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.aggr import Aggregation


class Explainer:
    def __init__(self):
        self.feature_set = None


    def fit(self, data, model, max_depth=7):
        x = data.x.numpy()
        adj = to_scipy_sparse_matrix(data.edge_index)
        self.feature_set = FeatureSet(x)
        
        values = []
        hooks = []
        leaf_modules = leaf_modules_in_order(model, data)
        for layer in leaf_modules:
            hooks.append(layer.register_forward_hook(
                lambda mod, inp, out: values.append(out.detach().numpy())
            ))
        model.eval()
        with torch.no_grad():
            out = model(data).detach().numpy()
        for i, (value, hook, mod) in enumerate(zip(values, hooks, leaf_modules)):
            hook.remove()
            if isinstance(mod, Aggregation):
                self.feature_set.deepen(adj)
                self.feature_set.iterate(
                    value, mask=data.train_mask, max_depth=max_depth)
            else:
                self.feature_set.iterate(
                    value, mask=data.train_mask, max_depth=max_depth)
            self.dt = self.feature_set.iterate(
                data.y.numpy(), mask=data.train_mask, update=False, max_depth=max_depth)

    def accuracy(self, data, split='val'):
        if split == 'train':
            mask = data.train_mask.numpy()
        elif split == 'val':
            mask = data.val_mask.numpy()
        elif split == 'test':
            mask = data.test_mask.numpy()
        elif split == 'full':
            mask = np.ones(data.y.size(0), dtype=bool)
        x = data.x.numpy()
        y = data.y.numpy()
        if isinstance(self.dt, DecisionTreeRegressor):    
            (self.dt.predict(self.feature_set[mask]).argmax(-1) == y[mask]).mean()
        if isinstance(self.dt, DecisionTreeClassifier):
            return (self.dt.predict(self.feature_set[mask]) == y[mask]).mean()
        raise ValueError("Model has not been trained yet.")

class FeatureSet:
    def __init__(self, x):
        self.x = x
        self.x_neigh = np.zeros((x.shape[0], 0))
        self.not_x_neigh = np.zeros((x.shape[0], 0))

    def iterate(self, y, mask=None, update=True, max_depth=7):
        if mask is None:
            mask = np.ones(y.shape[0], dtype=bool)
        x = np.concatenate([self.x, self.x_neigh, self.not_x_neigh], axis=1)
        new_features, dt = _iterate(x, y, mask, max_depth=max_depth)
        if update:
            self.x = np.concatenate([self.x, new_features], axis=1)
        return dt

    def deepen(self, adj):
        self.x_neigh = adj @ self.x
        self.not_x_neigh = adj @ (1 - self.x)

    def __getitem__(self, idx):
        return np.concatenate([self.x, self.x_neigh, self.not_x_neigh], axis=1)[idx]


def _iterate(x, y, mask, max_depth=7):
    if np.issubdtype(y.dtype, np.floating):
        dt = DecisionTreeRegressor(max_depth=max_depth)
    else:
        dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(x[mask], y[mask])
    leaves = _leaves(dt.tree_)
    dt_out = dt.apply(x)
    new_features = [dt_out == leaf for leaf in leaves]
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=.1, linkage='ward')
    clustering.fit(dt.tree_.value[leaves].squeeze())
    new_features = agglomerate_features(clustering, new_features)
    
    return np.array(new_features).T, dt


def _leaves(tree, node_id=0):
    if tree.children_left[node_id] == tree.children_right[node_id]:
        return [node_id]
    else:
        left = _leaves(tree, tree.children_left[node_id])
        right = _leaves(tree, tree.children_right[node_id])
        return left + right


def agglomerate_features(clustering, features):
    for i, children in enumerate(clustering.children_):
        a, b = children
        features.append(np.logical_or(features[a], features[b]))
    return features


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
