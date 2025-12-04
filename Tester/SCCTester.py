from typing import Callable, Any
import random

import networkx as nx

from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter
from Tester.BaseTester import BaseTester, MetamorphicMutator


class SCCTesterAlgorithms:
    @staticmethod
    def _wrapper(result):
        return set(map(frozenset, list(result)))

    @staticmethod
    def default(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(nx.strongly_connected_components(graph))

    @staticmethod
    def recursive(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(
            nx.strongly_connected_components_recursive(graph)
        )

    @staticmethod
    def kosaraju(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(
            nx.kosaraju_strongly_connected_components(graph)
        )

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph_default()
        return set(
            frozenset(map(int, graph_ig.vs[component]["_nx_name"]))
            for component in graph_ig.components(mode="STRONG")
        )


class SCCMetamorphicMutator(MetamorphicMutator):
    def mutate(
        self, graph: nx.Graph, input: Any, result: set[frozenset[int]]
    ) -> tuple[nx.Graph, Any, Callable[[set[frozenset[int]]], bool]]:
        if len(graph.nodes) == 0:
            return graph, input, (lambda _: True)
        graph_mutated = self.add_edge_inside_component(graph, result)
        checker = lambda res: (len(res) == len(result))
        return graph_mutated, input, checker

    def add_edge_inside_component(self, graph: nx.DiGraph, result):
        component = list(random.choice(list(result)))
        start, end = (random.choice(component), random.choice(component))
        graph_mutated = graph.copy()
        graph_mutated.add_edge(start, end)
        return graph_mutated

    def remove_edge_between_components(self, graph, result):
        graph_mutated = graph.copy()
        for _ in range(100):
            out_component = random.choice(result)
            start_node = random.choice(list(out_component))
            edge = random.choice(list(graph.edges(start_node)))
            if edge[1] in out_component:
                continue
            graph_mutated.remove_edge(edge[0], edge[1])
            return graph_mutated
        return graph_mutated


class SCCTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="scc_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "default": SCCTesterAlgorithms.default,
            "recursive": SCCTesterAlgorithms.recursive,
            "kosaraju": SCCTesterAlgorithms.kosaraju,
            "igraph": SCCTesterAlgorithms.igraph,
        }

    @staticmethod
    def get_metamorphic_mutator():
        return SCCMetamorphicMutator()
