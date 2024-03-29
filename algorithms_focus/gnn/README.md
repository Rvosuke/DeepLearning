# Cause discovery and graph neural network classification

This project uses the NOTEARS algorithm of the CausalNex library for causal discovery and performs graph-level classification tasks through graph neural networks. We used the breast_cancer data set in scikit-learn and the virtual data set generated by the make_classification function for testing.

## start using

Before running the algorithm, the data set needs to be prepared. Running the `load_datasets.py` script will automatically prepare and process the dataset (breast_cancer by default), which performs the following tasks:

1. Causal discovery - Use the NOTEARS algorithm to identify causal relationships in a data set.
2. Data processing - Generate the CSV files required for the graph neural network, including:
    - `expression.csv`: sample feature matrix
    - `target.csv`: label vector
    - `adjacency_matrix.csv`: adjacency matrix

After completing the data preparation, run `main.py` directly to start the classification task and obtain the classification results.
In `main.py`, you can affect the performance of the algorithm by adjusting different parameters, such as `no_pos` controls whether positional encoding is used, and `gcn_base_layers` sets the number of base layers of the graph convolution network.


## Dependency installation

The project's dependent libraries are listed in the `requirements.txt` file. Before installing dependent libraries, make sure your Python environment is ready. To install dependent libraries, use the following command:

```bash
pip install -r requirements.txt
```

## Test data set

This project uses the breast_cancer data set in scikit-learn by default. You can also construct a custom virtual data set for testing through scikit-learn's make_classification function.