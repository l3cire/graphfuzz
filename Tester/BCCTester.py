from typing import Callable, Any
import random

import networkx as nx

from Tester.BaseTester import BaseTester, TestMetamorphism
from Utils.GraphConverter import GraphConverter


class BCCTesterAlgorithms:
    @staticmethod
    def networkx(graph: nx.DiGraph):
        return set(map(frozenset, list(nx.biconnected_components(graph))))

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph_default()
        return set(
            frozenset(map(int, graph_ig.vs[component]["_nx_name"]))
            for component in graph_ig.biconnected_components()
        )


class BCCTestMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: set[frozenset[int]]
    ) -> tuple[nx.Graph, Any, Callable[[set[frozenset[int]]], bool]]:
        if len(result) == 0:
            return graph, input, lambda _: True
        new_graph = graph.copy()
        new_node = max(list(graph.nodes)) + 1
        new_graph.add_node(new_node)

        comp = random.choice(list(result))
        n_neighbours = random.randint(1, len(list(comp)))
        neighbours = random.sample(list(comp), n_neighbours)
        new_graph.add_edges_from([(new_node, node) for node in neighbours])

        def check(components: set[frozenset[int]]):
            if n_neighbours == 1:
                return comp in components
            else:
                return comp.union({new_node}) in components

        return new_graph, input, check


class BCCTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="bcc_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "networkx": BCCTesterAlgorithms.networkx,
            "igraph": BCCTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return BCCTestMetamorphism()
