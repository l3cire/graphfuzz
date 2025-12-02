import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.SCCTester import SCCTester
from Utils.FileUtils import (
    create_single_node_digraph,
    save_graphs,
    load_graphs,
)


class SCCFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "scc_corpus"

    def executor(self, G):
        return list(nx.strongly_connected_components(G))

    def get_tester(self):
        return SCCTester(self.corpus_path)

    def create_single_graph(self):
        return [create_single_node_digraph()]

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
        save_graphs(generated_graphs, "scc_corpus")
        return load_graphs("scc_corpus")


if __name__ == "__main__":
    scc_fuzzer = SCCFuzzer(
        num_iterations=100, use_multiple_graphs=True, feedback_check_type="regular"
    )
    scc_fuzzer.run()
