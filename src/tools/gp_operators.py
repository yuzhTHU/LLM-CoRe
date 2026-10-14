import logging
import numpy as np
import nd2py as nd
import pandas as pd
from functools import reduce
from argparse import Namespace
from typing import List, Generator
from ..utils import my_uuid, my_hash
from ..core import Individual, Triplet
from .gplearn_generator import GPLearnGenerator
from ..assistant import Assistant

_logger = logging.getLogger(__name__)


class GPOperator:

    def __init__(self, args: Namespace, assistant: Assistant, generator: GPLearnGenerator):
        self.args = args
        self.assistant = assistant
        self.generator = generator
        self._rng = np.random.default_rng(seed := args.random_state + my_hash(self.__class__.__name__))
        _logger.debug(f"GPOperator initialized with RNG seed: {seed}, initial state: {self._rng.bit_generator.state}")
        self.method_probs = {
            "crossover": args.p_crossover,
            "subtree-mutation": args.p_subtree_mutation,
            "hoist-mutation": args.p_hoist_mutation,
            "point-mutation": args.p_point_mutation,
            "term-reduction": args.p_term_reduction,
            "collapse-reduction": args.p_collapse_reduction,
            "unwrap-reduction": args.p_unwrap_reduction,
            "modal-reduction": args.p_modal_reduction,
        }
        if (sum_p := sum(self.method_probs.values())) <= 1.0:
            self.method_probs['reproduction'] = 1.0 - sum_p
        else:
            raise ValueError(f"Sum of method probabilities {sum_p} exceeds 1.0")

    def gp_mutation(
        self, population: List[Individual], child_num: int, n_iter: int
    ) -> Generator[Individual, None, None]:
        """ 使用遗传编程方法对种群进行变异，生成新的个体 """
        loader = self.tournament(population, child_num)
        for parent in loader:
            ## Sample a method based on probabilities
            key_prob = list(self.method_probs.keys())
            cum_prob = np.cumsum(list(self.method_probs.values()))
            method = key_prob[np.searchsorted(cum_prob, self._rng.random())]
            ## Apply the method to create new coeff, f_sst, f_wwv
            parents = [parent]
            if method == "crossover":
                donor = self.tournament(population, 1)[0]
                coeff = self.crossover(parent.triplet.coeff, donor.triplet.coeff)
                f_sst = self.crossover(parent.triplet.f_sst, donor.triplet.f_sst)
                f_wwv = self.crossover(parent.triplet.f_wwv, donor.triplet.f_wwv)
                parents.append(donor)
            elif method == "subtree-mutation":
                coeff = self.subtree_mutation(parent.triplet.coeff)
                f_sst = self.subtree_mutation(parent.triplet.f_sst)
                f_wwv = self.subtree_mutation(parent.triplet.f_wwv)
            elif method == "hoist-mutation":
                coeff = self.hoist_mutation(parent.triplet.coeff)
                f_sst = self.hoist_mutation(parent.triplet.f_sst)
                f_wwv = self.hoist_mutation(parent.triplet.f_wwv)
            elif method == "point-mutation":
                coeff = self.point_mutation(parent.triplet.coeff)
                f_sst = self.point_mutation(parent.triplet.f_sst)
                f_wwv = self.point_mutation(parent.triplet.f_wwv)
            elif method == "reproduction":
                coeff = parent.triplet.coeff.copy()
                f_sst = parent.triplet.f_sst.copy()
                f_wwv = parent.triplet.f_wwv.copy()
            elif method == "term-reduction":
                coeff = parent.triplet.coeff.copy()
                f_sst = parent.triplet.f_sst.copy()
                f_wwv = parent.triplet.f_wwv.copy()
                target_attr = self._rng.choice(["coeff", "f_sst", "f_wwv"])
                if target_attr == "coeff": coeff = self.term_reduce(coeff)
                elif target_attr == "f_sst": f_sst = self.term_reduce(f_sst)
                elif target_attr == "f_wwv": f_wwv = self.term_reduce(f_wwv)
                else: raise ValueError(f"Unknown target attribute for reduction: {target_attr}")
            elif method == "collapse-reduction":
                coeff = parent.triplet.coeff.copy()
                f_sst = parent.triplet.f_sst.copy()
                f_wwv = parent.triplet.f_wwv.copy()
                target_attr = self._rng.choice(["coeff", "f_sst", "f_wwv"])
                if target_attr == "coeff": coeff = self.collapse_reduce(coeff)
                elif target_attr == "f_sst": f_sst = self.collapse_reduce(f_sst)
                elif target_attr == "f_wwv": f_wwv = self.collapse_reduce(f_wwv)
                else: raise ValueError(f"Unknown target attribute for reduction: {target_attr}")
            elif method == "unwrap-reduction":
                coeff = parent.triplet.coeff.copy()
                f_sst = parent.triplet.f_sst.copy()
                f_wwv = parent.triplet.f_wwv.copy()
                target_attr = self._rng.choice(["f_sst", "f_wwv"])
                if target_attr == "coeff": raise ValueError(f"Cannot reduce coeff with unwrap method")
                elif target_attr == "f_sst": f_sst = self.unwrap_reduce(f_sst)
                elif target_attr == "f_wwv": f_wwv = self.unwrap_reduce(f_wwv)
                else: raise ValueError(f"Unknown target attribute for unwrap reduction: {target_attr}")
            elif method == "modal-reduction":
                coeff = parent.triplet.coeff.copy()
                f_sst = parent.triplet.f_sst.copy()
                f_wwv = parent.triplet.f_wwv.copy()
                target_attr = self._rng.choice(["f_sst", "f_wwv"])
                if target_attr == "coeff": raise ValueError(f"Cannot reduce coeff with modal method")
                elif target_attr == "f_sst": f_sst = self.modal_reduce(f_sst)
                elif target_attr == "f_wwv": f_wwv = self.modal_reduce(f_wwv)
                else: raise ValueError(f"Unknown target attribute for modal reduction: {target_attr}")
            else:
                raise ValueError(f"Unknown method: {method}")
            ## Prevent generating too complex formulas
            if self.args.coeff_max_len_for_GP:
                coeff = self.reduce_tree(coeff, max_len=self.args.coeff_max_len_for_GP)
            if self.args.f_sst_max_len_for_GP:
                f_sst = self.reduce_tree(f_sst, max_len=self.args.f_sst_max_len_for_GP)
            if self.args.f_wwv_max_len_for_GP:
                f_wwv = self.reduce_tree(f_wwv, max_len=self.args.f_wwv_max_len_for_GP)
            triplet = Triplet(coeff=coeff, f_sst=f_sst, f_wwv=f_wwv)
            ## Generate Children
            try:
                triplet = self.assistant.triplet_to_reduction(triplet)
                child = Individual(
                    uuid=my_uuid(self._rng),
                    n_iter=n_iter,
                    source=method,
                    parents=[parent.uuid for parent in parents],
                    triplet=triplet,
                    program=self.assistant.triplet_to_program(triplet),
                    dynamics=self.assistant.triplet_to_dynamics(triplet),
                )
                yield child
            except Exception as exc:
                _logger.warning("Discarding an invalid GP candidate: %s", exc)
                yield None

    def tournament(self, population: List[Individual], num) -> List[Individual]:
        size = (num, self.args.tournament_size)
        groups = self._rng.choice(population, size=size, replace=True)
        winners = [max(g, key=lambda x: x.fitness) for g in groups]
        return winners

    def crossover(self, parent: nd.Symbol, donor: nd.Symbol) -> nd.Symbol:
        """Crossover: 用 donor 的某个子树替换 parent 的某个子树"""
        child = parent.copy()
        removed_subtree = self.sample_subtree(child)
        nettypes = removed_subtree.replaceable_nettype()
        donored_subtree = self.sample_subtree(donor, nettype=nettypes)
        child = child.replace(removed_subtree, donored_subtree.copy())
        return child

    def subtree_mutation(self, parent: nd.Symbol) -> nd.Symbol:
        """Subtree mutation: 用一个随机树替换某个子树"""
        child = parent.copy()
        removed_subtree = self.sample_subtree(child)
        nettype = removed_subtree.nettype
        donor_subtree = self.generator.generate_eqtree(nettype=nettype)
        child = child.replace(removed_subtree, donor_subtree.copy())
        return child

    def hoist_mutation(self, parent: nd.Symbol) -> nd.Symbol:
        """Hoist mutation: 用某个子树替换根节点"""
        child = self.sample_subtree(parent, nettype=parent.nettype).copy()
        return child

    def point_mutation(self, parent: nd.Symbol) -> nd.Symbol:
        """Point mutation: 用随机符号替换 / 插入某个节点"""
        child = parent.copy()  # TODO
        return child

    def sample_subtree(self, tree: nd.Symbol, nettype: nd.NetType = None) -> nd.Symbol:
        """
        follow the same approach as GPlearn and Koza (1992) to choose functions 90% of the time and leaves 10% of the time.
        """
        if isinstance(nettype, str):
            nettype = {nettype}
        if nettype is None:
            nodes = [op for op in tree.iter_preorder()]
        else:
            nodes = []
            for op in tree.iter_preorder():
                if op.nettype in nettype or op.nettype == "scalar":
                    nodes.append(op)
                elif op.nettype == "edge" and "node" in nettype:
                    if nd.Aggr in self.generator.unary:
                        nodes.append(nd.Aggr(op))
                    if nd.Rgga in self.generator.unary:
                        nodes.append(nd.Rgga(op))
                elif op.nettype == "node" and "edge" in nettype:
                    if nd.Sour in self.generator.unary:
                        nodes.append(nd.Sour(op))
                    if nd.Targ in self.generator.unary:
                        nodes.append(nd.Targ(op))
        if len(nodes) == 0:
            return self.generator.generate_eqtree(nettype=nettype)

        probs = np.array([0.9 if node.n_operands > 0 else 0.1 for node in nodes])
        idx = np.searchsorted(np.cumsum(probs / probs.sum()), self._rng.random())
        subtree = nodes[idx]
        return subtree

    def reduce_tree(self, tree: nd.Symbol, max_len=10, max_trials=100) -> nd.Symbol:
        """ 简单地通过选择子树来减少树的复杂度 """
        reduced_tree = tree.copy()
        for _ in range(max_trials):
            if len(reduced_tree) <= max_len:
                break
            reduced_tree = self.sample_subtree(reduced_tree, reduced_tree.nettype)
        return reduced_tree

    def term_reduce(self, parent: nd.Symbol) -> nd.Symbol:
        """ 分支修剪 (Branch Pruning)：随机拆解并丢弃一项 """
        terms = parent.split_by_add(split_by_sub=True, expand_aggr=False, expand_sour=False, expand_targ=False)
        if len(terms) <= 1:
            return parent.copy() # 如果只有一项，则无法进一步修剪
        
        # 检查并拆分 Aggr 项，同时构建可供丢弃的候选池
        parsed_terms = []
        candidate_indices = []  # 记录所有项的坐标：(在 parsed_terms 中的索引, 在 items 中的内部索引)
        for i, term in enumerate(terms):
            if isinstance(term, nd.Aggr): # 3. 检查是否为 nd.Aggr
                # 4. 拆解 Aggr 的内部项
                inner_terms = term.operands[0].split_by_add(split_by_sub=True, expand_aggr=True, expand_sour=True, expand_targ=True)
                parsed_terms.append({'is_aggr': True, 'items': inner_terms})
                for j in range(len(inner_terms)):
                    candidate_indices.append((i, j))
            else:
                parsed_terms.append({'is_aggr': False, 'items': [term]})
                candidate_indices.append((i, 0))
                
        # 从候选池中随机扔掉一个
        drop_struct_idx, drop_inner_idx = self._rng.choice(candidate_indices)
        parsed_terms[drop_struct_idx]['items'].pop(drop_inner_idx)
        
        # 重新合并
        final_terms = []
        for struct in parsed_terms:
            items = struct['items']
            if not items:
                continue # 如果该项（例如某个 Aggr 内部唯一的一项）被完全丢弃，则跳过
            if struct['is_aggr']:
                final_terms.append(nd.Aggr(reduce(lambda a, b: a + b, items)))
            else:
                final_terms.append(items[0])
        if not final_terms: # 兜底逻辑：如果全空则返回原树
            return parent.copy()
            
        final_merged = reduce(lambda a, b: a + b, final_terms)
        return final_merged

    def collapse_reduce(self, parent: nd.Symbol) -> nd.Symbol:
        """节点坍塌 (Node Collapse)：将子树替换为 0, 1 或常数"""
        child = parent.copy()
        nodes = [op for op in child.iter_preorder()]
        # weights = np.array([1.0 / (node.eval(df, **eval_args).std() + 1) for node in nodes])
        # probs = weights / weights.sum()
        # target_node = self._rng.choice(nodes, p=probs)
        target_node = self._rng.choice(nodes) # 先不考虑权重，纯随机选择一个节点
        collapse_val = self._rng.choice([0.0, 1.0]) 
        constant_node = nd.Number(collapse_val, nettype='scalar', fitable=False) # 替换为你们实际的常数构建方法
        child = child.replace(target_node, constant_node)
        return child

    def unwrap_reduce(self, parent: nd.Symbol) -> nd.Symbol:
        """算子剥离 (Operator Unwrapping)：消除非线性算子或进行泰勒展开"""
        child = parent.copy()
        
        # 找到所有属于一元非线性算子的节点
        unary_nodes = [node for node in child.iter_preorder() if type(node) in self.generator.unary and type(node) not in [nd.Sour, nd.Targ, nd.Aggr, nd.Rgga, nd.Readout]]
        if not unary_nodes:
            return child
            
        target_node = self._rng.choice(unary_nodes)
        inner_expr = target_node.operands[0].copy()
        
        strategy = self._rng.choice(['linear', 'taylor_1st'])
        if strategy == 'linear':
            replacement = inner_expr
        elif strategy == 'taylor_1st':
            # 针对特定的非线性算子进行一阶近似
            if type(target_node) is nd.Exp:
                # exp(x) ≈ x + 1
                # replacement = inner_expr + 1
                replacement = inner_expr
            elif type(target_node) is nd.Sin:
                # sin(x) ≈ x
                replacement = inner_expr
            elif type(target_node) is nd.Cos:
                # cos(x) ≈ 1
                # replacement = nd.Number(1.0)
                replacement = inner_expr # 示意
            else:
                replacement = inner_expr
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
                
        child = child.replace(target_node, replacement)
        return child

    def modal_reduce(self, parent: nd.Symbol) -> nd.Symbol:
        """模态限制 (Modal Masking)：强制项仅对特定气候模态生效"""
        modes = ['ENSO', 'NPMM', 'SPMM', 'IOD', 'SIOD', 'IOB', 'ATL3', 'SASD', 'TNA']
        terms = self.split_by_add(parent)
        
        if not terms:
            return parent.copy()
            
        # 随机挑选一个相加的项
        target_idx = self._rng.integers(0, len(terms))
        target_term = terms[target_idx]
        
        # 随机挑选 1 到 2 个气候模态
        num_modes = self._rng.integers(1, 3) # 1 或 2 个模态
        selected_modes = self._rng.choice(modes, size=num_modes, replace=False)
        
        # 构造模态的 Variable，并相加（如果是多个）
        mode_expr = nd.Variable(selected_modes[0], nettype='node')
        for m in selected_modes[1:]:
            mode_expr = nd.Add(mode_expr, nd.Variable(m, nettype='node'))
            
        # 将原有的项乘上这个模态限制器
        masked_term = nd.Mul(target_term.copy(), mode_expr)
        terms[target_idx] = masked_term
        
        # 重新组装
        child = terms[0]
        for term in terms[1:]:
            child = nd.Add(child, term)
        return child
