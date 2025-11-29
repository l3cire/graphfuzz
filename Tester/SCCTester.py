from typing import Callable, Any

import networkx as nx

from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter
from Tester.BaseTester import BaseTester

class SCCTesterAlgorithms:
    @staticmethod
    def _wrapper(result):
        return set(map(frozenset, list(result)))

    @staticmethod
    def default(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(nx.strongly_connected_components(graph))

    @staticmethod
    def recursive(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(nx.strongly_connected_components_recursive(graph))
    
    @staticmethod
    def kosaraju(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(nx.kosaraju_strongly_connected_components(graph))
    
    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph_default()
        return set(frozenset(map(int, graph_ig.vs[component]['_nx_name'])) for component in graph_ig.components(mode='STRONG'))


class SCCTester(BaseTester):

    def __init__(self, discrepancy_filename="scc_discrepancy.pkl"):
        super().__init__(discrepancy_filename)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            'default': SCCTesterAlgorithms.default,
            'recursive': SCCTesterAlgorithms.recursive,
            'kosaraju': SCCTesterAlgorithms.kosaraju,
            'igraph': SCCTesterAlgorithms.igraph
        }

    def test(self, graph: nx.DiGraph, timestamp):
        discrepancy_msg, discrepancy_graph = self.test_algorithms(graph)
        if discrepancy_msg:
            save_discrepancy((discrepancy_msg, discrepancy_graph, timestamp),
                             f"scc_discrepancy_{self.uuid}.pkl")
        return discrepancy_msg, discrepancy_graph
