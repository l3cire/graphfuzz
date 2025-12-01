import os
import uuid
from typing import Callable, Any
import networkx as nx
import igraph as ig

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancies, save_discrepancy
from Utils.GraphConverter import GraphConverter


class MaxMatchingTesterAlgorithms:
    @staticmethod
    def hopcroft_karp(graph: nx.Graph):
        return len(nx.algorithms.bipartite.matching.hopcroft_karp_matching(graph))
    
    @staticmethod
    def eppstein(graph: nx.Graph):
        return len(nx.algorithms.bipartite.matching.eppstein_matching(graph))
    
    @staticmethod
    def igraph(graph: nx.Graph):
        if not nx.is_bipartite(graph):
            raise ValueError("Provided NetworkX graph is not bipartite")

        # Get the two sets of the bipartite graph
        sets = nx.bipartite.sets(graph)
        types = [node in sets[0] for node in graph.nodes()]

        # Convert NetworkX graph to iGraph
        converter = GraphConverter(graph)
        g = converter.to_igraph()

        # Set the 'type' attribute for each vertex in iGraph
        g.vs['type'] = types

        # Compute the maximum bipartite matching
        matching = g.maximum_bipartite_matching()
        matching_size = sum(1 for i in range(g.vcount()) if matching.is_matched(i))
        return matching_size


class MaxMatchingTester(BaseTester):

    def __init__(self, corpus_filename="max_matching_corpus.pkl"):
        super().__init__(corpus_filename)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            'hopcroft_karp': MaxMatchingTesterAlgorithms.hopcroft_karp,
            'eppstein': MaxMatchingTesterAlgorithms.eppstein,
            'igraph': MaxMatchingTesterAlgorithms.igraph
        }

    def test(self, G):
        discrepancy_msg, discrepancy_graph = self.test_algorithms(G)
        if discrepancy_msg:
            save_discrepancy((discrepancy_msg, discrepancy_graph),
                             f"max_matching_discrepancy_{self.uuid}.pkl")
        return discrepancy_msg, discrepancy_graph
