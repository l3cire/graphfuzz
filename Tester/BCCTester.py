from typing import Callable, Any

import networkx as nx

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancy
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


class BCCTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="bcc_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "networkx": BCCTesterAlgorithms.networkx,
            "igraph": BCCTesterAlgorithms.igraph,
        }
