import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.STPLTester import STPLTester
from Utils.FileUtils import create_single_node_digraph, save_graphs, load_graphs


class STPLFuzzer(BaseFuzzer):
    def get_corpus_name(self):
        return "stpl_corpus"

    def executor(self, G):
        if len(G.nodes()) < 2:
            return float("inf")  # Return infinity for graphs with less than 2 nodes
        # Sort nodes by degree in descending order
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        # Select the node with the highest degree as the source
        source = sorted_nodes[0]
        # Select the node with the second highest degree as the target
        target = sorted_nodes[1]
        try:
            return nx.shortest_path_length(
                G, source=source, target=target, weight="weight", method="bellman-ford"
            )
        except nx.NetworkXNoPath:
            return float("inf")  # Return infinity if no path exists

    def executor_hop_count(self, G):
        """Executor that returns hop count instead of path weight."""
        if len(G.nodes()) < 2:
            return float("inf")
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        source = sorted_nodes[0]
        target = sorted_nodes[1]
        try:
            path = nx.shortest_path(
                G, source=source, target=target, weight="weight", method="bellman-ford"
            )
            return len(path) - 1  # Number of edges (hops)
        except nx.NetworkXNoPath:
            return float("inf")
        except (nx.NetworkXError, nx.NetworkXUnbounded):
            return float("-inf")

    def executor_negative_edges(self, G):
        """Executor that returns count of negative weight edges in shortest path."""
        if len(G.nodes()) < 2:
            return 0
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        source = sorted_nodes[0]
        target = sorted_nodes[1]
        try:
            path = nx.shortest_path(
                G, source=source, target=target, weight="weight", method="bellman-ford"
            )
            # Count negative edges in the path
            negative_count = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if G.has_edge(u, v):
                    weight = G[u][v].get("weight", 1)
                    if weight < 0:
                        negative_count += 1
            return negative_count
        except nx.NetworkXNoPath:
            return 0  # No path means no edges
        except (nx.NetworkXError, nx.NetworkXUnbounded):
            return -1  # Special marker for negative cycle

    def hop_count_interesting_check(self, result):
        """Returns the number of edges (hops) in the shortest path.

        This is orthogonal to path weight - a path with few heavy edges
        vs many light edges will have different hop counts.
        """
        # Result is already the hop count from executor_hop_count
        return result

    def negative_edge_interesting_check(self, result):
        """Returns the count of negative weight edges in the shortest path.

        Different counts indicate different handling of negative edges.
        """
        # Result is already the negative edge count from executor_negative_edges
        return result

    def get_tester(self):
        return STPLTester(
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
            negative_weights=True,
            negative_cycle=False,
            parallel_edges=False,
        )
        generated_graphs = generator.generate()
        save_graphs(generated_graphs, "stpl_corpus")
        return load_graphs("stpl_corpus")


if __name__ == "__main__":
    stpl_fuzzer = STPLFuzzer(
        num_iterations=100, use_multiple_graphs=True, feedback_check_type="regular"
    )
    stpl_fuzzer.run()
