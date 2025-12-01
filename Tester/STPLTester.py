import random

import networkx as nx

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancies, save_discrepancy
from Utils.GraphConverter import GraphConverter


class STPLTesterAlgorithms:
    @staticmethod
    def _set_default_weights(graph: nx.Graph):
        for _, _, data in graph.edges(data=True):
            data.setdefault("weight", 1)

    @staticmethod
    def bellman_ford_path_length(graph: nx.Graph, source, target):
        STPLTesterAlgorithms._set_default_weights(graph)
        return nx.bellman_ford_path_length(
            graph, source=source, target=target, weight="weight"
        )

    @staticmethod
    def dijkstra_path_length(graph, source, target):
        STPLTesterAlgorithms._set_default_weights(graph)
        return nx.dijkstra_path_length(
            graph, source=source, target=target, weight="weight"
        )

    @staticmethod
    def goldberg_radzik(graph, source, target):
        STPLTesterAlgorithms._set_default_weights(graph)
        _, dist = nx.goldberg_radzik(graph, source, weight="weight")
        return dist.get(target, float("inf"))

    @staticmethod
    def igraph(graph, source, target):
        if graph.number_of_edges() == 0 or nx.negative_edge_cycle(
            graph, weight="weight"
        ):
            return float("inf")

        STPLTesterAlgorithms._set_default_weights(graph)
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph()
        source_ig = graph_ig.vs.find(name=str(source)).index
        target_ig = graph_ig.vs.find(name=str(target)).index

        shortest_paths = graph_ig.shortest_paths(
            source=source_ig, target=target_ig, weights="weight"
        )
        return shortest_paths[0][0] if shortest_paths else float("inf")


class STPLTester(BaseTester):

    def __init__(self, corpus_filename="stpl_corpus.pkl"):
        super().__init__(corpus_filename)

    def test(self, G, timestamp, num_pairs=5):
        total_discrepancies = []

        if len(G) < 2:
            return []

        for _ in range(num_pairs):
            # # Get the degrees of all nodes and sort them in descending order
            # sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

            # # Select the top two nodes with the highest degree as source and target
            # source, target = sorted_nodes[:2]
            source, target = random.sample(G.nodes(), 2)

            discrepancy_msg, discrepancy_graph = self.test_algorithms(G, source, target)

            if discrepancy_msg and len(G.nodes()) < 20:
                save_discrepancy(
                    (discrepancy_msg, discrepancy_graph, timestamp),
                    f"stpl_discrepancy_{self.uuid}.pkl",
                )
                total_discrepancies.append((discrepancy_msg, discrepancy_graph))

        return total_discrepancies

    def test_algorithms(self, G, source, target, exception_result=float("inf")):
        has_negative_weight = any(
            data.get("weight", 0) < 0 for _, _, data in G.edges(data=True)
        )
        self.algorithms = {
            "bellman_ford_path_length": STPLTesterAlgorithms.bellman_ford_path_length,
            "goldberg_radzik": STPLTesterAlgorithms.goldberg_radzik,
            "igraph": STPLTesterAlgorithms.igraph,
        }
        # Include Dijkstra's algorithm if there are no negative weights
        if not has_negative_weight:
            self.algorithms["dijkstra_path_length"] = (
                STPLTesterAlgorithms.dijkstra_path_length
            )

        return super().test_algorithms(
            G, source, target, exception_result=exception_result
        )

    def run(self):
        """Test shortest path length algorithms on every graph in the corpus."""
        discrepancy_data = []
        discrepancy_counts = {}
        count = 0
        for G in self.corpus:
            # Ensure the graph has at least two nodes
            if len(G) < 2:
                print(f"Graph number {count + 1} has less than two nodes. Skipping...")
                continue

            # Sort nodes by degree in descending order
            # sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

            # Select the node with the highest degree as the source
            # source = sorted_nodes[0]

            # Select the node with the second highest degree as the target
            # target = sorted_nodes[1]
            # source, target = random.sample(G.nodes(), 2)

            for _ in range(10):
                if len(G) < 2:
                    break  # Skip if the graph has less than 2 nodes

                source, target = random.sample(G.nodes(), 2)

                test_result, discrepancy_msg = self.test_stpl_algorithms_updated(
                    G, source, target
                )
                if not test_result:
                    discrepancy_counts[discrepancy_msg] = (
                        discrepancy_counts.get(discrepancy_msg, 0) + 1
                    )

                    if discrepancy_counts[discrepancy_msg] <= 5:
                        discrepancy_data.append((discrepancy_msg, G))

            count += 1

        if discrepancy_data:
            # Save the discrepancy graphs and messages to a file
            save_discrepancies(discrepancy_data, "stpl_discrepancy.pkl")
            print(f"There are {len(discrepancy_data)} graphs with discrepancies saved.")

            # Print all the discrepancy messages and their counts
        for msg, count in discrepancy_counts.items():
            print(f'Discrepancy message: "{msg}" occurred {count} times.')

        print("End of STPL testing.")
        return discrepancy_counts
