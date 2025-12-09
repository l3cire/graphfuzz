from itertools import permutations
from typing import Any, Callable
import random

import networkx as nx

from Tester.BaseTester import BaseTester, TestMetamorphism
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
        similarity_matrix = graph_ig.similarity_inverse_log_weighted(mode="all")
        res = {}
        for u, v in list(permutations(list(graph.nodes), 2)):
            u_ig = graph_ig.vs.find(name=str(u)).index
            v_ig = graph_ig.vs.find(name=str(v)).index
            res[(u, v)] = similarity_matrix[u_ig][v_ig]
        return res


class AdamicAdarTestMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: dict[tuple[int, int], float]
    ) -> tuple[nx.Graph, Any, Callable[[tuple[int, int], float], bool]]:
        if len(graph.nodes) < 2:
            return graph, input, lambda _: True
        if random.choice([True, False]):
            return self.mutate_add_neighbour_not_common(graph, input, result)
        else:
            return self.mutate_add_common_neighbour(graph, input, result)

    def mutate_add_neighbour_not_common(
        self, graph: nx.Graph, input: Any, result: dict[tuple[int, int], float]
    ) -> tuple[nx.Graph, Any, Callable[[tuple[int, int], float], bool]]:
        all_nodes = set(graph.nodes)
        for _ in range(100):
            u, v = random.sample(list(graph.nodes), 2)
            u_neighbours = set(graph.neighbors(u))
            v_neighbours = set(graph.neighbors(v))
            all_neighbours = u_neighbours.union(v_neighbours).difference(
                set([u, v])
            )  # u, v not counted for Adamic-Adar
            free_nodes = all_nodes.difference(all_neighbours)
            if len(free_nodes) == 0:
                continue
            new_graph = graph.copy()
            new_graph.add_edge(u, free_nodes.pop())
            checker = lambda res: (abs(res[(u, v)] - result[(u, v)]) < 1e-3)
            return new_graph, input, checker
        return graph, input, lambda _: True

    def mutate_add_common_neighbour(
        self, graph: nx.Graph, input: Any, result: dict[tuple[int, int], float]
    ) -> tuple[nx.Graph, Any, Callable[[tuple[int, int], float], bool]]:
        for _ in range(100):
            u, v = random.sample(list(graph.nodes), 2)
            u_neighbours = set(graph.neighbors(u))
            v_neighbours = set(graph.neighbors(v))
            diff = v_neighbours.difference(u_neighbours).difference(set([u, v]))
            if len(diff) == 0:
                continue
            new_graph = graph.copy()
            new_graph.add_edge(u, diff.pop())
            checker = lambda res: (res[(u, v)] > result[(u, v)])
            return new_graph, input, checker
        return graph, input, lambda _: True


class AdamicAdarTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="aa_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms = {
            "networkx": AdamicAdarTesterAlgorithms.networkx,
            "igraph": AdamicAdarTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return AdamicAdarTestMetamorphism()

    def test_algorithms(self, G):
        """Test Adamic-Adar index between networkx and igraph."""

        nx_aa_dict = AdamicAdarTesterAlgorithms.networkx(G)
        ig_aa_dict = AdamicAdarTesterAlgorithms.igraph(G)

        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    # Retrieve Adamic-Adar score from iGraph
                    ig_score = ig_aa_dict.get((node1, node2), 0)
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
