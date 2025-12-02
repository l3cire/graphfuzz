import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.AdamicAdarTester import AdamicAdarTester
from Utils.FileUtils import create_single_node_graph, save_graphs, load_graphs


class AdamicAdarFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "aa_corpus"

    # def interesting_check(self, adamic_adar_scores):
    #     if not adamic_adar_scores:
    #         return None
    #     sorted_scores = sorted(adamic_adar_scores, key=lambda x: x[2], reverse=True)
    #     interesting_pair = sorted_scores[0] if sorted_scores else None
    #     return interesting_pair

    def executor(self, G):
        return list(nx.adamic_adar_index(G))

    def get_tester(self):
        return AdamicAdarTester(self.corpus_path)

    def create_single_graph(self):
        return [create_single_node_graph()]

    def create_multiple_graphs(self):
        generator = SmokeGenerator(
            self.executor,
            n=30,
            m=10,
            directed=True,
            weighted=True,
            negative_weights=True,
            negative_cycle=True,
            parallel_edges=False,
        )
        generated_graphs = generator.generate_n_graphs(10)
        save_graphs(generated_graphs, "aa_corpus")
        return load_graphs("aa_corpus")


if __name__ == "__main__":
    aa_fuzzer = AdamicAdarFuzzer(
        num_iterations=30, use_multiple_graphs=False, feedback_check_type="regular"
    )
    aa_fuzzer.run()
