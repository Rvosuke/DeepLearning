import pandas as pd
import networkx as nx

from sklearn.datasets import load_breast_cancer
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Load breast cancer dataset
data = load_breast_cancer()
target = pd.Series(data.target, name='target')
target.to_csv('target.csv')

# Convert node feature to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('features.csv')

# Create adjacency matrix
sm = from_pandas(df)
sm.remove_edges_below_threshold(1)  # avoid weak edges

viz = plot_structure(sm, all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
viz.toggle_physics(False)
viz.show("connected.html")

adj_matrix = nx.to_pandas_adjacency(sm)
adj_matrix = adj_matrix.applymap(lambda x: 1 if x != 0 else 0)
adj_matrix.to_csv('adjacency_matrix.csv')
