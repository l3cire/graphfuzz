import networkx as nx

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter


class JaccardSimilarityTesterAlgorithms:
    @staticmethod
    def networkx(graph: nx.DiGraph):
        return list(nx.jaccard_coefficient(graph))

    @staticmethod
    def igraph(graph: nx.DiGraph, pairs):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph()

        if len(graph_ig.vs) == 0:
            vertex_id_map = {}
        else:
            vertex_id_map = {
                str(node): idx for idx, node in enumerate(graph_ig.vs["name"])
            }

        ig_jaccard_results = []
        for u, v in pairs:
            u_id, v_id = vertex_id_map[str(u)], vertex_id_map[str(v)]
            ig_jaccard_score = graph_ig.similarity_jaccard(
                pairs=[(u_id, v_id)], loops=False
            )[0]
            ig_jaccard_results.append((u, v, ig_jaccard_score))

        return ig_jaccard_results


class JaccardSimilarityTester(BaseTester):

    def __init__(self, corpus_path, discrepancy_filename="js_discrepancy"):
        super().__init__(corpus_path, discrepancy_filename)

    def test_algorithms(self, G):
        """Test Jaccard similarity between networkx and igraph."""
        nx_jaccard = list(nx.jaccard_coefficient(G))
        ig_jaccard = JaccardSimilarityTesterAlgorithms.igraph(
            G, [(u, v) for u, v, _ in nx_jaccard]
        )

        for (u, v, nx_score), (_, _, ig_score) in zip(nx_jaccard, ig_jaccard):
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
