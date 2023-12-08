import causalnex
import pandas as pd
import networkx as nx

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from causalnex.plots import NODE_STYLE, EDGE_STYLE

# Load breast cancer dataset
# data = load_breast_cancer()
scaler = StandardScaler()

X, Y = make_classification(n_samples=2500, n_features=16, n_informative=12, n_redundant=4, n_classes=2, n_clusters_per_class=2, random_state=96)

X_scaled = scaler.fit_transform(X)

target = pd.Series(Y, name='target')
target.to_csv('target.csv')

# Convert node feature to pandas DataFrame
expression = pd.DataFrame(X_scaled)
expression['target'] = Y
# expression.to_csv('expression.csv')


# Create adjacency matrix
sm = causalnex.structure.notears.from_pandas(expression)
adj_matrix = nx.to_pandas_adjacency(sm)
non_zero_values = adj_matrix.values[adj_matrix.values != 0]
edge_threshold = np.mean(non_zero_values)
sm.remove_edges_below_threshold(edge_threshold)  # avoid weak edges

# search for the largest connected component
G = nx.DiGraph(sm)
largest_cc = max(nx.weakly_connected_components(G), key=len)
subgraph = G.subgraph(largest_cc)

expression_subgraph = expression.iloc[list(largest_cc)]
expression_subgraph.to_csv('expression_subgraph.csv')

viz = causalnex.plots.plot_structure(nx.DiGraph(subgraph), all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
viz.toggle_physics(False)
viz.show("connected.html")

adj_matrix = nx.to_pandas_adjacency(subgraph)
adj_matrix = adj_matrix.applymap(lambda x: 1 if x != 0 else 0)
adj_matrix.to_csv('adjacency_matrix.csv')
