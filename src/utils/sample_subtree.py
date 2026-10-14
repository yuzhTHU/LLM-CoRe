import nd2py as nd
import numpy as np

def sample_subtree(eqtree: nd.Symbol, max_nodes: int, random_state: int) -> nd.Symbol:
    """Sample a random subtree from the equation tree with at most max_nodes nodes."""
    raise DeprecationWarning("Please use gp_operators.GPOperators.sample_subtree instead.")
    rng = np.random.default_rng(random_state)

    nettype = eqtree.nettype

    subtrees = []
    for node in eqtree.iter_preorder():
        if len(node) > max_nodes:
            continue
        if node.nettype == nettype:
            subtrees.append(node)
        elif node.nettype == "scalar":
            subtrees.append(node)
        elif node.nettype == "node" and nettype == "scalar":
            subtrees.append(node)


