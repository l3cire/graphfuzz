import math
from typing import Callable, Any
import random

import networkx as nx

from Tester.BaseTester import BaseTester, TestMetamorphism
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


class HarmonicCentralityMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: dict[int, float]
    ) -> tuple[nx.Graph, Any, Callable[[dict[int, float]], bool]]:
        scaling_factor = max(0.1, random.random() * 10)
        new_graph = graph.copy()
        for _, _, data in new_graph.edges(data=True):
            data["weight"] = data.get("weight", 1) * scaling_factor

        def check(res: dict[int, float]):
            for node, value in result.items():
                expected_result = value / scaling_factor
                if abs(expected_result - res[node]) > 1e-3:
                    return False
            return True

        return new_graph, input, check


class HarmonicCentralityTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="hc_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "networkx": HarmonicCentralityTesterAlgorithms.networkx,
            "igraph": HarmonicCentralityTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return HarmonicCentralityMetamorphism()

    def test_algorithms(self, G):
        """Test harmonic centrality between networkx and igraph."""

        def contains_negative_or_nan_weight(graph):
            for _, _, data in graph.edges(data=True):
                weight = data.get("weight", 0)
                if weight <= 0 or math.isnan(weight):
                    return True
            return False

        if contains_negative_or_nan_weight(G):
            return None, None

        nx_centrality_dict = self.algorithms["networkx"](G)
        ig_centrality_dict = self.algorithms["igraph"](G)

        for node, nx_score in nx_centrality_dict.items():
            ig_score = ig_centrality_dict.get(str(node))
            if ig_score is None or not self.approximately_equal(nx_score, ig_score):
                discrepancy_msg = (
                    f"Results of networkx and igraph are different for a graph!"
                )
                return discrepancy_msg, G

        return None, None

    @staticmethod
    def approximately_equal(a, b, tol=1e-6):
        """Check if two centrality values are approximately equal considering a tolerance."""
        return abs(a - b) <= tol
