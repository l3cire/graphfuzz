import numpy as np
from typing import Callable, Any

import networkx as nx

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancies, save_discrepancy
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


class MSTTester(BaseTester):

    def __init__(self, corpus_path, discrepancy_filename="mst_discrepancy"):
        super().__init__(corpus_path, discrepancy_filename)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "kuskal": MSTTesterAlgorithms.kruskal,
            "prim": MSTTesterAlgorithms.prim,
            "boruvka": MSTTesterAlgorithms.boruvka,
            "igraph": MSTTesterAlgorithms.igraph,
        }

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
