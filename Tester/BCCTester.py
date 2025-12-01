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

    def __init__(self, corpus_filename="bcc_corpus.pkl"):
        super().__init__(corpus_filename)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "networkx": BCCTesterAlgorithms.networkx,
            "igraph": BCCTesterAlgorithms.igraph,
        }

    def test(self, G, timestamp):
        discrepancy_msg, discrepancy_graph = self.test_algorithms(G)
        if discrepancy_msg:
            save_discrepancy(
                (discrepancy_msg, discrepancy_graph, timestamp),
                f"bcc_discrepancy_{self.uuid}.pkl",
            )
        return discrepancy_msg, discrepancy_graph
