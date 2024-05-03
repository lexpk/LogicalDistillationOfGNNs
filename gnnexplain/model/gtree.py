import re
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch
import optuna


class Optimizer:
    def __init__(self, max_depth=10, max_ccp_alpha=5e-3, lmb=1e-3, n_trials=100):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.lmb = lmb
        self.n_trials = n_trials
        self.explainer = None
    
    def optimize(self, batch, model_or_int, logger=None, progress_bar=True):
        datalist = batch.to_data_list()
        half = len(datalist) // 2
        train = Batch.from_data_list(datalist[:half])
        val = Batch.from_data_list(datalist[half:])
        values = _get_values(train, model_or_int)

        if logger is not None:
            logger.experiment.define_metric('gtree_step')
            logger.experiment.define_metric('gtree_train_acc', step_metric='gtree_step')
            logger.experiment.define_metric('gtree_opt_acc', step_metric='gtree_step')
            logger.experiment.define_metric('gtree_node_count', step_metric='gtree_step')
            logger.experiment.define_metric('gtree_regularized_acc', step_metric='gtree_step')
            

        def objective(trial, logger):
            params = {
                f'max_depth_{i}' : trial.suggest_int(f'max_depth_{i}', 1, self.max_depth)
                for i in range(len(values) + 1)
            } | {
                f'ccp_alpha_{i}' : trial.suggest_float(f'ccp_alpha_{i}', 0, self.max_ccp_alpha)
                for i in range(len(values) + 1)
            }
            expl = Explainer(**params).fit(train, values).prune()
            train_acc = expl.accuracy(train)
            opt_acc = expl.accuracy(val) 
            node_count =sum(layer.dt.tree_.node_count for layer in expl.layer) + expl.out_layer.dt.tree_.node_count
            regularized_acc = opt_acc - self.lmb * node_count
            if logger is not None:
                logger.experiment.log({
                    'gtree_step': trial.number,
                    'gtree_train_acc': train_acc,
                    'gtree_opt_acc': opt_acc,
                    'gtree_node_count': node_count,
                    'gtree_regularized_acc': regularized_acc,
                }, commit=True)
            return regularized_acc

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, logger), n_trials=self.n_trials, show_progress_bar=progress_bar)

        values = _get_values(batch, model_or_int)
        self.explainer = Explainer(**study.best_params).fit(batch, values).prune()
        return self.explainer

    def predict(self, batch):
        return self.explainer.predict(batch)

    def accuracy(self, batch):
        return self.explainer.accuracy(batch)

    def fidelity(self, batch, model):
        return self.explainer.fidelity(batch, model)

    def f1_score(self, batch):
        return self.explainer.f1_score(batch)


class Explainer:
    def __init__(self, **params):
        self.params = params
        self.layer = []
        self.out_layer = None

    def fit(self, batch, values):
        adj = to_scipy_sparse_matrix(batch.edge_index, num_nodes=batch.x.shape[0]).tobsr()
        x, y = batch.x.numpy(), batch.y.numpy()
        depth_indices = [x.shape[1]]
        for i, value in enumerate(values):
            for _ in range(10):
                self.layer.append(ExplainerLayer(self.params[f'max_depth_{i}'], self.params[f'ccp_alpha_{i}'], depth_indices))
                samples = np.random.choice(np.arange(len(batch)), size=min(10, x.shape[0]), replace=False)
                small_batch = Batch.from_data_list(batch[samples])
                small_batch_indices = torch.arange(len(batch.batch))[(batch.batch == torch.tensor(samples).view(-1, 1)).max(axis=0).values]
                self.layer[-1].fit(x[small_batch_indices], value[small_batch_indices], to_scipy_sparse_matrix(small_batch.edge_index, num_nodes=small_batch_indices.shape[0]).tobsr())
                x_new = self.layer[-1].predict(x, adj)
                depth_indices.append(x_new.shape[1])
                x = np.concatenate([x, x_new], axis=1)
        self.out_layer = ExplainerPoolingLayer(
            self.params[f'max_depth_{len(values)}'],
            self.params[f'ccp_alpha_{len(values)}'],
            depth_indices
        )
        y = self.out_layer.fit(x, batch.batch, y)
        return self

    def predict(self, batch):
        x = batch.x.numpy()
        adj = to_scipy_sparse_matrix(batch.edge_index, num_nodes=x.shape[0]).tobsr()
        for layer in self.layer:
            x = np.concatenate([x, layer.predict(x, adj)], axis=1)
        return self.out_layer.predict(x, batch.batch)

    def accuracy(self, batch):
        prediction = self.predict(batch)
        return (prediction == batch.y.numpy()).mean()

    def prune(self):
        relevant = self.out_layer._relevant()
        for i, layer in enumerate(self.layer[::-1]):
            _relevant = {index for (depth, index) in relevant if depth == len(self.layer) - i - 1}
            layer._prune_irrelevant(_relevant)
            layer._remove_redunant()
            relevant |= layer._relevant()
        return self

    def save_image(self, path):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=len(self.layer) + 1, figsize=(10, 4 * len(self.layer) + 4))
        for i, layer in enumerate(self.layer):
            layer.plot(axs[i], i)
        self.out_layer.plot(axs[-1], len(self.layer))
        fig.savefig(path)
        plt.close(fig)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=len(self.layer) + 1, figsize=(10, 4 * len(self.layer) + 4))
        for i, layer in enumerate(self.layer):
            layer.plot(axs[i], i)
        self.out_layer.plot(axs[-1], len(self.layer))
        plt.show()

    def fidelity(self, batch, model):
        prediction = self.predict(batch)
        with torch.no_grad():
            model_pred = model(batch).argmax(dim=1).numpy()
        return (prediction == model_pred).mean()

    def f1_score(self, batch):
        prediction = self.predict(batch)
        return f1_score(batch.y.numpy(), prediction, average='macro')


class ExplainerLayer:
    def __init__(self, max_depth, max_ccp_alpha, depth_indices):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None
        self.leaf_formulas = None

    def fit(self, x, y, adj):
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))
        x_neigh = adj @ x
        x = np.asarray(np.concatenate([
            x, x_neigh, x + x_neigh, x_neigh / (adj.sum(axis=1)).clip(1e-6, None), (x + x_neigh) / (1 + adj.sum(axis=1))
        ], axis=1))
        self.dt = DecisionTreeRegressor(max_depth=self.max_depth, ccp_alpha=self.max_ccp_alpha, splitter='best')
        self.dt.fit(x, y)
        leaves = _leaves(self.dt.tree_)
        leaf_values = [self.dt.tree_.value[i, :, 0] for i in leaves]
        if len(leaves) == 1:
            combinations = [{0}]
        else:
            clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
            clustering.fit(leaf_values)
            combinations = _agglomerate_labels(len(leaves), clustering)
        self.leaf_indices = np.array([
            leaves.index(i) if i in leaves else -1
            for i in range(self.dt.tree_.node_count)
        ])
        self.leaf_values = np.array([
            [i in combination for combination in combinations]
            for i in self.leaf_indices if i != -1
        ])
        self.leaf_formulas = [
            [i for (i, b) in enumerate(self.leaf_values[j]) if b]
            for j in range(len(self.leaf_values))
        ]
        return self

    def predict(self, x, adj=None):
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))
        x_neigh = adj @ x
        x = np.asarray(np.concatenate([
            x, x_neigh, x + x_neigh, x_neigh / (adj.sum(axis=1)).clip(1e-6, None), (x + x_neigh) / (1 + adj.sum(axis=1))
        ], axis=1))
        pred = self.dt.apply(x)
        return self.leaf_values[self.leaf_indices[pred]]

    def fit_predict(self, x, y, adj=None):
        self.fit(x, y, adj)
        return self.predict(x, adj)

    def _relevant(self):
        return {_feature_depth_index(feature, self.depth_indices) for feature in self.dt.tree_.feature if feature != -2}

    def _prune_irrelevant(self, relevant):
        self.leaf_formulas = [
            [i for i in formulas if i in relevant]
            for formulas in self.leaf_formulas
        ]

    def _remove_redunant(self):
        flag = True
        while flag:
            flag = False
            for parent, (left, right) in enumerate(zip(self.dt.tree_.children_left, self.dt.tree_.children_right)):
                if self.leaf_indices[left] != -1 and self.leaf_indices[right] != -1 and left != -1 and right != -1 and \
                        self.leaf_formulas[self.leaf_indices[left]] == self.leaf_formulas[self.leaf_indices[right]]:
                    self._merge_leaves(parent, left, right)
                    flag = True
                    break

    def _merge_leaves(self, parent, left, right):
        self.dt.tree_.children_left[parent] = -1
        self.dt.tree_.children_right[parent] = -1
        self.leaf_indices[parent] = self.leaf_indices[left]
        self.leaf_indices[left] = -1
        self.leaf_indices[right] = -1
        self.dt.tree_.feature[parent] = -2
        self.dt.tree_.threshold[parent] = -2

    def plot(self, ax, n=0):
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)
        leaf_counter = 0
        for obj in ax.properties()['children']:
            if type(obj) == matplotlib.text.Annotation:
                obj.set_fontsize(8)
                txt = obj.get_text().splitlines()[0]
                match = re.match(r'x\[(\d+)\] <= (\d+\.\d+)', txt)
                if match:
                    feature, threshold = match.groups()
                    feature = int(feature)
                    formula = _feature_formula(feature, self.depth_indices)
                    threshold = float(threshold)
                    if feature < self.n_features_in:
                        obj.set_text(fr'$I{formula} > 0$')
                    elif feature < 2 * self.n_features_in:
                        obj.set_text(fr'$A{formula} > {int(threshold)}$')
                    elif feature < 3 * self.n_features_in:
                        obj.set_text(fr'$(I + A){formula} > {int(threshold)}$')
                    elif feature < 4 * self.n_features_in:
                        obj.set_text(fr'$A{formula} > {threshold}$')
                    else:
                        obj.set_text(fr'$(I + A){formula} > {threshold}$')
                else:
                    txt = r"$" + r", ".join([
                        fr"M_{{{i}}}^{{{n}}}" for i in self.leaf_formulas[[i for i in self.leaf_indices if i != -1][leaf_counter]]
                    ]) + r"\:$"
                    obj.set_text(txt)
                    leaf_counter += 1


class ExplainerPoolingLayer:
    def __init__(self, max_depth, max_ccp_alpha, depth_indices):
        self.max_depth = max_depth
        self.max_ccp_alpha = max_ccp_alpha
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None
    
    def fit(self, x, batch, y):
        x = _pool(x, batch)
        self.dt = DecisionTreeClassifier(max_depth=self.max_depth, ccp_alpha=self.max_ccp_alpha)
        self.dt.fit(x, y)

    def predict(self, x, batch):
        x = _pool(x, batch)
        return self.dt.predict(x)

    def fit_predict(self, x, batch, y):
        self.fit(x, batch, y)
        return self.predict(x, batch)

    def _relevant(self):
        return {_feature_depth_index(feature, self.depth_indices) for feature in self.dt.tree_.feature if feature != -2}

    def plot(self, ax, n=0):
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)
        for obj in ax.properties()['children']:
            if type(obj) == matplotlib.text.Annotation:
                obj.set_fontsize(8)
                txt = obj.get_text().splitlines()[0]
                #match x[3] <= 13.1312
                match = re.match(r'x\[(\d+)\] <= (\d+\.\d+)', txt)
                if match:
                    feature, threshold = match.groups()
                    feature = int(feature)
                    formula = _feature_formula(feature, self.depth_indices)
                    threshold = float(threshold)
                    if feature < self.n_features_in:
                        obj.set_text(fr'$1{formula} > {threshold}$')
                    else:
                        obj.set_text(fr'$1{formula} > {int(threshold)}$')


def _pool(x, batch):
    return np.concatenate([
        global_mean_pool(torch.tensor(x), batch).numpy(),
        global_add_pool(torch.tensor(x), batch).numpy()
    ], axis=1)



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


def _get_values(batch, model_or_int):
    if isinstance(model_or_int, int):
        return _get_values_int(batch, model_or_int)
    if isinstance(model_or_int, torch.nn.Module):
        return _get_values_model(batch, model_or_int)
    raise ValueError(f"Unknown type {type(model_or_int)}")


def _get_values_int(batch, int):
    labels = batch.y.numpy().max() + 1
    values = []
    for datapoint in batch.to_data_list():
        onehot = np.eye(labels)[datapoint.y.numpy()]
        values.append(onehot.repeat(datapoint.x.shape[0], axis=0))
    values = np.concatenate(values, axis=0).squeeze()
    values = [values] * int
    return values


def _get_values_model(batch, model):
    values = []
    hook = model.act.register_forward_hook(
        lambda mod, inp, out: values.append(out.detach().numpy().squeeze())
    )
    with torch.no_grad():
        model(batch).detach().numpy()
    hook.remove()
    return values


def _feature_formula(index, depth_indices):
    depth, index = _feature_depth_index(index, depth_indices)
    if depth == -1:
        return fr'U_{{{index}}}'
    else:
        return fr'\chi_{{{index}}}^{{{depth}}}'

def _feature_depth_index(index, depth_indices):
    index = index % sum(depth_indices)
    depth = -1
    for i in depth_indices:
        if index < i:
            return depth, index
        index -= i
        depth += 1
