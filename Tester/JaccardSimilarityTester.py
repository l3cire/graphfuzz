import networkx as nx
from typing import Any, Callable
import random

from Tester.BaseTester import BaseTester, TestMetamorphism
from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter


class JaccardSimilarityTesterAlgorithms:
    @staticmethod
    def networkx(graph: nx.DiGraph):
        return list(nx.jaccard_coefficient(graph))

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph()

        if len(graph_ig.vs) == 0:
            vertex_id_map = {}
        else:
            vertex_id_map = {
                str(node): idx for idx, node in enumerate(graph_ig.vs["name"])
            }

        nodes = list(graph.nodes)
        ig_jaccard_results = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                u_id, v_id = vertex_id_map[str(u)], vertex_id_map[str(v)]
                ig_jaccard_score = graph_ig.similarity_jaccard(
                    pairs=[(u_id, v_id)], loops=False
                )[0]
                ig_jaccard_results.append((u, v, ig_jaccard_score))

        return ig_jaccard_results

    @staticmethod
    def find_coef(res: list[tuple[int, int, float]], u: int, v: int) -> float:
        for x, y, val in res:
            if (x == u and y == v) or (x == v and y == u):
                return val
        return 0


class JaccardSimilarityMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: int
    ) -> tuple[nx.Graph, Any, Callable[[int], bool]]:
        if len(graph.nodes) < 2:
            return graph, input, lambda _: True
        if random.choice([True, False]):
            return self.mutate_increase_similarity(graph, input, result)
        else:
            return self.mutate_decrease_similarity(graph, input, result)

    def mutate_increase_similarity(
        self, graph: nx.Graph, input: Any, result: list[tuple[int, int, float]]
    ) -> tuple[nx.Graph, Any, Callable[[list[tuple[int, int, float]]], bool]]:
        """
        add edge to create a common neighbour
        """
        for _ in range(100):
            u, v = random.sample(list(graph.nodes), 2)
            u_neighbours = set(graph.neighbors(u))
            v_neighbours = set(graph.neighbors(v))
            if u in v_neighbours:
                continue
            diff = v_neighbours.difference(u_neighbours.union(set([v])))
            if len(diff) == 0:
                continue
            new_graph = graph.copy()
            new_graph.add_edge(u, diff.pop())
            checker = lambda res: (
                JaccardSimilarityTesterAlgorithms.find_coef(res, u, v)
                > JaccardSimilarityTesterAlgorithms.find_coef(result, u, v)
            )
            return new_graph, input, checker
        return graph, input, lambda _: True

    def mutate_decrease_similarity(
        self, graph: nx.Graph, input: Any, result: list[tuple[int, int, float]]
    ) -> tuple[nx.Graph, Any, Callable[[list[tuple[int, int, float]]], bool]]:
        """
        add edge to create an exclusive neighbour
        """
        nodes = set(graph.nodes)
        for _ in range(100):
            u, v = random.sample(list(graph.nodes), 2)
            if JaccardSimilarityTesterAlgorithms.find_coef(result, u, v) == 0:
                continue
            u_neighbours = set(graph.neighbors(u))
            v_neighbours = set(graph.neighbors(v))
            remaining_nodes = nodes.difference(
                u_neighbours.union(v_neighbours).union(set([v]))
            )
            if len(remaining_nodes) == 0:
                continue
            new_graph = graph.copy()
            new_graph.add_edge(u, remaining_nodes.pop())
            checker = lambda res: (
                JaccardSimilarityTesterAlgorithms.find_coef(res, u, v)
                < JaccardSimilarityTesterAlgorithms.find_coef(result, u, v)
            )
            return new_graph, input, checker
        return graph, input, lambda _: True


class JaccardSimilarityTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="js_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms = {
            "networkx": JaccardSimilarityTesterAlgorithms.networkx,
            "igraph": JaccardSimilarityTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return JaccardSimilarityMetamorphism()

    def test_algorithms(self, G):
        """Test Jaccard similarity between networkx and igraph."""
        nx_jaccard = self.algorithms["networkx"](G)
        ig_jaccard = self.algorithms["igraph"](G)

        for u, v, nx_score in nx_jaccard:
            ig_score = JaccardSimilarityTesterAlgorithms.find_coef(ig_jaccard, u, v)
            if not self.approximately_equal(nx_score, ig_score):
                discrepancy_msg = (
                    f"Results of networkx and igraph are different for a graph!"
                )
                return discrepancy_msg, G

        return None, None

    @staticmethod
    def approximately_equal(a, b, tol=1e-6):
        """Check if two similarity values are approximately equal considering a tolerance."""
        return abs(a - b) <= tol
