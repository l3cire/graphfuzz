import time

import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.HarmonicCentralityTester import HarmonicCentralityTester
from Utils.FileUtils import create_single_node_graph, save_graphs, load_graphs


class HarmonicCentralityFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "hc_corpus"

    def executor(self, G):
        return nx.harmonic_centrality(G, distance="weight")

    def get_tester(self):
        return HarmonicCentralityTester(self.corpus_path)

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
            negative_cycle=True,
            parallel_edges=False,
        )
        generated_graphs = generator.generate()
        save_graphs(generated_graphs, "hc_corpus")
        return load_graphs("hc_corpus")


if __name__ == "__main__":
    hc_fuzzer = HarmonicCentralityFuzzer(
        num_iterations=60, use_multiple_graphs=False, feedback_check_type="regular"
    )

    # Optionally, set a user-defined interesting check function
    def custom_interesting_check(result):
        min_diff = 1e-5
        most_interesting_pair = None
        nodes = list(result.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                diff = abs(result[u] - result[v])
                if 0 < diff <= min_diff:
                    min_diff = diff
                    most_interesting_pair = (u, v)
        return min_diff

    hc_fuzzer.set_interesting_check(custom_interesting_check)
    hc_fuzzer.run()
