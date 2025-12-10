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

    def executor_component_distribution(self, G):
        """Executor that returns hash of component size distribution.

        This tracks the pattern of component sizes rather than just the count.
        For example, [10, 1, 1] vs [4, 4, 4] both have 3 components but different
        distributions that exercise different code paths.
        """
        scc_list = list(nx.strongly_connected_components(G))
        if not scc_list:
            return 0
        # Sort component sizes in descending order for consistent hashing
        sizes = tuple(sorted([len(comp) for comp in scc_list], reverse=True))
        # Hash and bucket to reduce state space while maintaining diversity
        return hash(sizes) % 10000

    def executor_trivial_ratio(self, G):
        """Executor that returns the ratio of singleton components to total components.

        High ratio indicates sparse connectivity (many isolated nodes),
        low ratio indicates dense connectivity (fewer but larger components).
        This helps explore different connectivity patterns.
        """
        scc_list = list(nx.strongly_connected_components(G))
        if not scc_list:
            return 0
        singleton_count = sum(1 for comp in scc_list if len(comp) == 1)
        # Return ratio as percentage bucketed into ranges: 0-24%, 25-49%, 50-74%, 75-99%, 100%
        ratio = singleton_count / len(scc_list)
        return int(ratio * 100) // 25  # Returns 0, 1, 2, 3, or 4

    def component_distribution_interesting_check(self, result):
        """Check function for component distribution feedback.

        Result is already the hash of the distribution from executor_component_distribution.
        """
        return result

    def trivial_ratio_interesting_check(self, result):
        """Check function for trivial component ratio feedback.

        Result is already the bucketed ratio from executor_trivial_ratio.
        """
        return result

    def get_tester(self):
        return SCCTester(
            self.corpus_path, test_method=self.test_method, algorithm=self.algorithm
        )

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
