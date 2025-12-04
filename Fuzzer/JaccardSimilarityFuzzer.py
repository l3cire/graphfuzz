import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.JaccardSimilarityTester import JaccardSimilarityTester
from Utils.FileUtils import create_single_node_graph, save_graphs, load_graphs


class JaccardSimilarityFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "js_corpus"

    def executor(self, G):
        return list(nx.jaccard_coefficient(G))

    def get_tester(self):
        return JaccardSimilarityTester(
            self.corpus_path, test_method=self.test_method, algorithm=self.algorithm
        )

    def create_single_graph(self):
        return [create_single_node_graph()]

    def create_multiple_graphs(self):
        generator = SmokeGenerator(
            self.executor,
            n=30,
            m=2,
            directed=True,
            weighted=True,
            negative_weights=False,
            negative_cycle=False,
            parallel_edges=False,
        )
        generated_graphs = generator.generate_n_graphs(10)
        save_graphs(generated_graphs, "js_corpus")
        return load_graphs("js_corpus")


if __name__ == "__main__":
    js_fuzzer = JaccardSimilarityFuzzer(
        num_iterations=50, use_multiple_graphs=True, feedback_check_type="regular"
    )
    js_fuzzer.run()
