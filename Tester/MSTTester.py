import numpy as np
from typing import Callable, Any
import random

import networkx as nx

from Tester.BaseTester import BaseTester, TestMetamorphism
from Utils.GraphConverter import GraphConverter


class MSTTesterAlgorithms:
    @staticmethod
    def _weight_sum(mst_edges):
        return sum(
            data["weight"] if "weight" in data else 1 for _, _, data in list(mst_edges)
        )

    @staticmethod
    def kruskal(graph: nx.DiGraph):
        return MSTTesterAlgorithms._weight_sum(
            nx.minimum_spanning_edges(graph, algorithm="kruskal", data=True)
        )

    @staticmethod
    def prim(graph: nx.DiGraph):
        return MSTTesterAlgorithms._weight_sum(
            nx.minimum_spanning_edges(graph, algorithm="prim", data=True)
        )

    @staticmethod
    def boruvka(graph: nx.DiGraph):
        return MSTTesterAlgorithms._weight_sum(
            nx.minimum_spanning_edges(graph, algorithm="boruvka", data=True)
        )

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph_default()
        if "weight" not in graph_ig.es.attribute_names():
            graph_ig.es["weight"] = 1
        return sum(
            edge["weight"] for edge in graph_ig.spanning_tree(weights="weight").es
        )


class MSTTestMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: int
    ) -> tuple[nx.Graph, Any, Callable[[int], bool]]:
        if len(graph.nodes) <= 1:
            return graph, input, (lambda _: True)
        graph_mutated, expected_result = self.compose_methods(graph, result)
        checker = lambda res: (res == expected_result)
        return graph_mutated, input, checker

    def compose_methods(self, graph: nx.DiGraph, result: int, max_compositions=4):
        n_compositions = random.randint(1, max_compositions)
        all_methods = [
            self.add_node_single_edge,
        ]
        if nx.is_connected(graph):
            all_methods.extend(
                [self.add_edge_large_weight, self.add_node_multiple_edges]
            )
        new_graph, new_result = graph, result
        for _ in range(n_compositions):
            method = random.choice(all_methods)
            new_graph, new_result = method(new_graph, new_result)
        return new_graph, new_result

    def add_edge_large_weight(self, graph: nx.Graph, result: int):
        weight = result + 1
        nodes = list(graph.nodes)
        for _ in range(1000):
            u, v = random.sample(nodes, 2)
            if graph.has_edge(u, v):
                continue
            new_graph = graph.copy()
            new_graph.add_edge(u, v, weight=weight)
            return new_graph, result
        return graph, result

    def add_node_single_edge(self, graph: nx.Graph, result: int):
        new_node = max(list(graph.nodes)) + 1
        weight = random.randint(1, 20)
        some_node = random.choice(list(graph.nodes))
        new_graph = graph.copy()
        new_graph.add_node(new_node)
        new_graph.add_edge(new_node, some_node, weight=weight)
        return new_graph, result + weight

    def add_node_multiple_edges(self, graph: nx.Graph, result: int, max_n_edges=10):
        new_node = max(list(graph.nodes)) + 1
        n_edges = random.randint(1, min(len(list(graph.nodes)), max_n_edges))

        weights = [random.randint(result, result + 100) for _ in range(n_edges)]
        nodes = random.sample(list(graph.nodes), n_edges)

        new_graph = graph.copy()
        new_graph.add_node(new_node)
        for weight, old_node in zip(weights, nodes):
            new_graph.add_edge(new_node, old_node, weight=weight)

        return new_graph, result + min(weights)


class MSTTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="mst_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "kruskal": MSTTesterAlgorithms.kruskal,
            "prim": MSTTesterAlgorithms.prim,
            "boruvka": MSTTesterAlgorithms.boruvka,
            "igraph": MSTTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return MSTTestMetamorphism()

    def preprocess_weights(self, graph: nx.DiGraph):
        if not nx.is_weighted(graph):
            for u, v in graph.edges():
                graph[u][v]["weight"] = 1
        else:
            for u, v, data in graph.edges(data=True):
                if np.isnan(data.get("weight", 0)):
                    graph[u][v]["weight"] = 0

    def test(self, G, timestamp):
        self.preprocess_weights(G)
        return super().test(G, timestamp=timestamp)
