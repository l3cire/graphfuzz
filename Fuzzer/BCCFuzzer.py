import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.BCCTester import BCCTester
from Utils.FileUtils import create_single_node_graph, save_graphs, load_graphs


class BCCFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "bcc_corpus"

    def executor(self, G):
        return list(nx.biconnected_components(G))

    def get_tester(self):
        return BCCTester(
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
            negative_weights=True,
            negative_cycle=False,
            parallel_edges=False,
        )
        generated_graphs = generator.generate()
        save_graphs(generated_graphs, "bcc_corpus")
        return load_graphs("bcc_corpus")


if __name__ == "__main__":
    bcc_fuzzer = BCCFuzzer(
        num_iterations=50, use_multiple_graphs=True, feedback_check_type="regular"
    )
    bcc_fuzzer.run()
