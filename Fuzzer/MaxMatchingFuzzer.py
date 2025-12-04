import time

import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.CustomGenerator import CustomGenerator
from Tester.MaxMatchingTester import MaxMatchingTester
from Utils.FileUtils import save_graphs, load_graphs


class MaxMatchingFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "max_matching_corpus"

    def executor(self, G):
        return nx.algorithms.bipartite.matching.hopcroft_karp_matching(G)

    def get_tester(self):
        return MaxMatchingTester(
            self.corpus_path, test_method=self.test_method, algorithm=self.algorithm
        )

    def create_single_graph(self):
        generator = CustomGenerator(n=1, m=1, category="Bipartite")
        return generator.create_single_graph()

    def create_multiple_graphs(self):
        generator = CustomGenerator(n=30, m=5, category="Bipartite")
        generated_graphs = generator.create_graphs()
        save_graphs(generated_graphs, "max_matching_corpus")
        return load_graphs("max_matching_corpus")


if __name__ == "__main__":
    max_matching_fuzzer = MaxMatchingFuzzer(
        num_iterations=50, use_multiple_graphs=True, feedback_check_type="regular"
    )
    max_matching_fuzzer.run()
