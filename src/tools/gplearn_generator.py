import logging
import nd2py as nd
import numpy as np
from ..utils import my_hash
from numpy.random import default_rng
from typing import Tuple, List, Literal, Set

_logger = logging.getLogger(__name__)


class GPLearnGenerator:
    def __init__(self, args, assistant, variables, binary, unary, edge_list, num_nodes):
        self.unary = unary
        self.binary = binary
        if args.no_mode_vars_for_GP:
            self.variables = [var for name, var in variables.items() if name not in assistant.modes]
        else:
            self.variables = [var for name, var in variables.items()]
        self.num_nodes = num_nodes
        self.edge_list = edge_list
        self.symbols = binary + unary
        self.full_prob = args.full_prob
        self.depth_range = args.depth_range
        self.const_range = args.const_range
        self.scalar_number_only = True
        self._rng = np.random.default_rng(seed := args.random_state + my_hash(self.__class__.__name__))
        _logger.debug(f"GPLearnGenerator initialized with RNG seed: {seed}, initial state: {self._rng.bit_generator.state}")

    def generate_node(self, nettype: Set[Literal["node", "edge", "scalar"]]) -> nd.Symbol:
        symbol_nettypes = []
        for sym in self.symbols:
            for nt in sorted(nettype & sym.nettype_range()):
                symbol_nettypes.append((sym, nt))
        symbol, nettype = self._rng.choice(symbol_nettypes)
        node = symbol(nettype=nettype)
        return node

    def generate_leaf(
        self, nettype: Set[Literal["node", "edge", "scalar"]]
    ) -> nd.Number | nd.Variable:
        leafs = [var for var in self.variables if var.nettype in nettype]

        if self.const_range is not None:
            const_range = self.const_range
        elif len(leafs) == 0:
            const_range = (-1, 1)
        else:
            const_range = None

        if const_range is not None:
            if nettype == "scalar" or self.scalar_number_only or self._rng.integers(2):
                number = nd.Number(self._rng.uniform(*const_range), nettype="scalar")
            elif nettype == "node":
                number = nd.Number(
                    self._rng.uniform(*const_range, (self.num_nodes,)), nettype="node"
                )
            elif nettype == "edge":
                number = nd.Number(
                    self._rng.uniform(*const_range, (len(self.edge_list[0]),)),
                    nettype="edge",
                )
            else:
                raise ValueError(
                    f"Unknown nettype: {nettype}. Supported types are 'scalar', 'node', and 'edge'."
                )
            leafs = leafs + [number]

        return self._rng.choice(leafs)

    def generate_eqtree(
        self, nettype: Set[Literal["node", "edge", "scalar"]]
    ) -> nd.Symbol:
        if isinstance(nettype, str):
            nettype = {nettype}

        full_tree = self._rng.random() < self.full_prob
        max_depth = self._rng.integers(*self.depth_range)
        op_prob = (
            1.0
            if full_tree
            else len(self.symbols) / (len(self.variables) + len(self.symbols))
        )

        # Start a eqtree with a function to avoid degenerative eqtrees
        eqtree = self.generate_node(nettype)
        empty_nodes_and_depth = [(i, 1) for i in eqtree.operands]

        while empty_nodes_and_depth:
            empty_node, depth = empty_nodes_and_depth.pop(0)
            if (depth < max_depth) and (self._rng.random() < op_prob):
                node = self.generate_node(empty_node.replaceable_nettype())
                eqtree.replace(empty_node, node)
                empty_nodes_and_depth.extend([(i, depth + 1) for i in node.operands])
            else:  # Variable or Number
                leaf = self.generate_leaf(empty_node.replaceable_nettype())
                eqtree.replace(empty_node, leaf)
        return eqtree
