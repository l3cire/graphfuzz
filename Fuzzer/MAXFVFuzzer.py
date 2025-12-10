import uuid

import networkx as nx

from Fuzzer.BaseFuzzer import BaseFuzzer
from Generator.SmokeGenerator import SmokeGenerator
from Tester.MAXFVTester import MAXFVTester
from Utils.FileUtils import (
    create_single_node_graph,
    save_graphs,
    load_graphs,
    save_discrepancy,
)


class MAXFVFuzzer(BaseFuzzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uuid = uuid.uuid4().hex[:8]

    def get_corpus_name(self):
        return "maxfv_corpus"

    def executor(self, G):
        if len(G.nodes()) < 2:
            return 0  # Return 0 for graphs with less than 2 nodes
        # Sort nodes by degree in descending order
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        # Select the node with the highest degree as the source
        s = sorted_nodes[0]
        # Select the node with the second highest degree as the target
        t = sorted_nodes[1]
        return nx.maximum_flow_value(G, s, t, capacity="weight")

    def executor_saturated_edges(self, G):
        """Executor that returns the count of saturated edges in the max flow.

        Saturated edges (at full capacity) indicate bottlenecks in the flow network.
        Different saturation patterns exercise different augmenting path selection strategies.
        """
        if len(G.nodes()) < 2:
            return 0

        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        s = sorted_nodes[0]
        t = sorted_nodes[1]

        try:
            # Get the actual flow dictionary, not just the value
            flow_value, flow_dict = nx.maximum_flow(G, s, t, capacity="weight")

            # Count saturated edges (where flow equals capacity)
            saturated_count = 0
            for u in flow_dict:
                for v, flow in flow_dict[u].items():
                    if G.has_edge(u, v):
                        capacity = G[u][v].get('weight', 0)
                        # Edge is saturated if flow equals capacity (within tolerance)
                        if abs(flow - capacity) < 1e-9 and flow > 0:
                            saturated_count += 1

            return saturated_count
        except:
            return 0

    def saturated_edges_interesting_check(self, result):
        """Check function for saturated edges feedback.

        Result is already the count of saturated edges from executor_saturated_edges.
        """
        return result

    def get_tester(self):
        return MAXFVTester(
            self.corpus_path,
            id=self.uuid,
            test_method=self.test_method,
            algorithm=self.algorithm,
        )

    def create_single_graph(self):
        return [create_single_node_graph()]

    def create_multiple_graphs(self):
        print("Creating multiple graphs")
        generator = SmokeGenerator(
            self.executor,
            n=30,
            m=2,
            directed=True,
            weighted=True,
            negative_weights=False,
        )
        generated_graphs = generator.generate_n_graphs(10)
        save_graphs(generated_graphs, "maxfv_corpus")
        return load_graphs("maxfv_corpus")


if __name__ == "__main__":
    maxfv_fuzzer = MAXFVFuzzer(
        num_iterations=50, use_multiple_graphs=True, feedback_check_type="regular"
    )
    maxfv_fuzzer.run()
