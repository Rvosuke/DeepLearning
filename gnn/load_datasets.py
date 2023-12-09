import pandas as pd
import networkx as nx

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from causalnex.structure import notears
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Load breast cancer dataset
data = load_breast_cancer()
scaler = StandardScaler()

# X, Y = make_classification(n_samples=2500, n_features=16, n_informative=12, n_redundant=4, n_classes=2, n_clusters_per_class=2, random_state=96)
X, Y = data.data, data.target

X_scaled = scaler.fit_transform(X)

target = pd.Series(Y, name='target')
target.to_csv('target.csv')

# Convert node feature to pandas DataFrame
expression = pd.DataFrame(X_scaled)
expression['target'] = Y
# expression.to_csv('expression.csv')


# Create adjacency matrix
sm = notears.from_pandas(expression)
adj_matrix = nx.to_pandas_adjacency(sm)
non_zero_values = adj_matrix.values[adj_matrix.values != 0]
edge_threshold = np.mean(non_zero_values)
sm.remove_edges_below_threshold(edge_threshold)  # avoid weak edges

# search for the largest connected component
lagest_subgraph = sm.get_largest_subgraph()

expression_subgraph = expression.loc[:, list(lagest_subgraph.nodes())]
expression_subgraph.to_csv('expression.csv')

viz = plot_structure(nx.DiGraph(lagest_subgraph), all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
viz.toggle_physics(False)
viz.show("connected.html")

adj_matrix = nx.to_pandas_adjacency(lagest_subgraph)
adj_matrix = adj_matrix.applymap(lambda x: 1 if x != 0 else 0)
adj_matrix.to_csv('adjacency_matrix.csv')
