from itertools import permutations

import networkx as nx

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter


class AdamicAdarTesterAlgorithms:
    @staticmethod
    def networkx(graph: nx.DiGraph):
        nodes = list(graph.nodes())
        all_pairs = list(permutations(nodes, 2))
        nx_pairs = nx.adamic_adar_index(graph, ebunch=all_pairs)
        return {(u, v): aa for u, v, aa in nx_pairs}

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph()
        return graph_ig.similarity_inverse_log_weighted(mode="all")


class AdamicAdarTester(BaseTester):

    def __init__(self, corpus_path, discrepancy_filename="aa_discrepancy"):
        super().__init__(corpus_path, discrepancy_filename)

    def test_algorithms(self, G):
        """Test Adamic-Adar index between networkx and igraph."""

        nx_aa_dict = AdamicAdarTesterAlgorithms.networkx(G)
        ig_aa_matrix = AdamicAdarTesterAlgorithms.igraph(G)

        for i, node1 in enumerate(G.nodes()):
            for j, node2 in enumerate(G.nodes()):
                if node1 != node2:
                    # Retrieve Adamic-Adar score from iGraph
                    ig_score = ig_aa_matrix[i][j]
                    nx_score = nx_aa_dict.get((node1, node2), 0)

                    if not self.approximately_equal(nx_score, ig_score):
                        discrepancy_msg = (
                            f"Results of networkx and igraph are different for a graph!"
                        )
                        return discrepancy_msg, G

        return None, None

    @staticmethod
    def approximately_equal(a, b, tol=1e-3):
        """Check if two centrality values are approximately equal considering a tolerance."""
        return abs(a - b) <= tol
