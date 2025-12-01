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

    def __init__(self, corpus_filename="js_corpus.pkl"):
        super().__init__(corpus_filename)

    def test(self, G, timestamp):
        discrepancies = self.test_algorithms(G)
        if discrepancies:
            discrepancy_count = len(discrepancies)  # Count the discrepancies
            discrepancy_msg = (
                f"Results of NetworkX and iGraph are different for a graph!"
            )
            save_discrepancy(
                (discrepancy_msg, G, timestamp), f"js_discrepancy_{self.uuid}.pkl"
            )
            return discrepancy_msg, G, discrepancy_count
        return None, None, None

    def test_algorithms(self, G):
        """Test Jaccard similarity between networkx and igraph."""
        nx_jaccard = list(nx.jaccard_coefficient(G))
        ig_jaccard = JaccardSimilarityTesterAlgorithms.igraph(
            G, [(u, v) for u, v, _ in nx_jaccard]
        )

        discrepancies = []
        for (u, v, nx_score), (_, _, ig_score) in zip(nx_jaccard, ig_jaccard):
            if not self.approximately_equal(nx_score, ig_score):
                discrepancies.append((u, v, nx_score, ig_score))

        return discrepancies

    @staticmethod
    def approximately_equal(a, b, tol=1e-6):
        """Check if two similarity values are approximately equal considering a tolerance."""
        return abs(a - b) <= tol
