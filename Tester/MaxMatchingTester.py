from typing import Callable, Any
import random

import networkx as nx

from Tester.BaseTester import BaseTester, TestMetamorphism
from Utils.GraphConverter import GraphConverter


class MaxMatchingTesterAlgorithms:
    @staticmethod
    def is_graph_supported(graph: nx.Graph) -> bool:
        return (
            (len(list(graph.nodes)) >= 2)
            and nx.is_connected(graph)
            and nx.is_bipartite(graph)
        )

    @staticmethod
    def hopcroft_karp(graph: nx.Graph):
        if not MaxMatchingTesterAlgorithms.is_graph_supported(graph):
            return 0
        return len(nx.algorithms.bipartite.matching.hopcroft_karp_matching(graph))

    @staticmethod
    def eppstein(graph: nx.Graph):
        if not MaxMatchingTesterAlgorithms.is_graph_supported(graph):
            return 0
        return len(nx.algorithms.bipartite.matching.eppstein_matching(graph))

    @staticmethod
    def igraph(graph: nx.Graph):
        if not MaxMatchingTesterAlgorithms.is_graph_supported(graph):
            return 0

        # Get the two sets of the bipartite graph
        sets = nx.bipartite.sets(graph)
        types = [node in sets[0] for node in graph.nodes()]

        # Convert NetworkX graph to iGraph
        converter = GraphConverter(graph)
        g = converter.to_igraph()

        # Set the 'type' attribute for each vertex in iGraph
        g.vs["type"] = types

        # Compute the maximum bipartite matching
        matching = g.maximum_bipartite_matching()
        matching_size = sum(1 for i in range(g.vcount()) if matching.is_matched(i))
        return matching_size


class MaxMatchingMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: int
    ) -> tuple[nx.Graph, Any, Callable[[int], bool]]:
        if not MaxMatchingTesterAlgorithms.is_graph_supported(graph):
            return graph, input, lambda _: True
        mutation_type = random.choice([0, 1])
        new_graph = graph.copy()
        if mutation_type == 0:
            left, right = map(list, nx.bipartite.sets(new_graph))
            u, v = random.choice(left), random.choice(right)
            new_graph.add_edge(u, v)
            return new_graph, input, lambda x: (x >= result)
        elif mutation_type == 1:
            u, v = random.choice(list(new_graph.edges))
            new_graph.remove_edge(u, v)
            return new_graph, input, lambda x: (x <= result)


class MaxMatchingTester(BaseTester):

    def __init__(
        self,
        corpus_path,
        discrepancy_filename="max_matching_discrepancy",
        *args,
        **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "hopcroft_karp": MaxMatchingTesterAlgorithms.hopcroft_karp,
            "eppstein": MaxMatchingTesterAlgorithms.eppstein,
            "igraph": MaxMatchingTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return MaxMatchingMetamorphism()
