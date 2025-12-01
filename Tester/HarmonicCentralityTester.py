import math
from typing import Callable, Any

import networkx as nx

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter


class HarmonicCentralityTesterAlgorithms:
    @staticmethod
    def networkx(graph: nx.DiGraph):
        return nx.harmonic_centrality(graph, distance="weight")

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph()

        mode = "in" if graph.is_directed() else "all"
        # Compute harmonic centrality with igraph, taking into account if the graph is directed
        # Check if the iGraph has 'weight' attribute for edges
        if "weight" in graph_ig.es.attribute_names():
            ig_centrality = graph_ig.harmonic_centrality(
                mode=mode, weights="weight", normalized=False
            )
        else:
            # If no weights, compute centrality without weights
            ig_centrality = graph_ig.harmonic_centrality(mode=mode, normalized=False)

        # In igraph, the result is a list, map it to vertex ids
        return {
            graph_ig.vs["name"][v.index]: c for v, c in zip(graph_ig.vs, ig_centrality)
        }


class HarmonicCentralityTester(BaseTester):

    def __init__(self, corpus_filename="hc_corpus.pkl"):
        super().__init__(corpus_filename)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "networkx": HarmonicCentralityTesterAlgorithms.networkx,
            "igraph": HarmonicCentralityTesterAlgorithms.igraph,
        }

    def test(self, G):
        discrepancies = self.test_algorithms(G)
        if discrepancies:
            discrepancy_count = len(discrepancies)  # Count the discrepancies
            discrepancy_msg = (
                f"Results of NetworkX and iGraph are different for a graph!"
            )
            save_discrepancy((discrepancy_msg, G), f"hc_discrepancy_{self.uuid}.pkl")
            return discrepancy_msg, G, discrepancy_count
        return None, None, None

    def test_algorithms(self, G):
        """Test harmonic centrality between networkx and igraph."""

        def contains_negative_or_nan_weight(graph):
            for u, v, data in graph.edges(data=True):
                weight = data.get("weight", 0)
                if weight <= 0 or math.isnan(weight):
                    return True
            return False

        if contains_negative_or_nan_weight(G):
            return []

        discrepancies = []
        nx_centrality_dict = self.algorithms["networkx"](G)
        ig_centrality_dict = self.algorithms["igraph"](G)

        for node, nx_score in nx_centrality_dict.items():
            ig_score = ig_centrality_dict.get(str(node))
            if ig_score is None or not self.approximately_equal(nx_score, ig_score):
                discrepancies.append((node, nx_score, ig_score))

        return discrepancies

    @staticmethod
    def approximately_equal(a, b, tol=1e-6):
        """Check if two centrality values are approximately equal considering a tolerance."""
        return abs(a - b) <= tol
