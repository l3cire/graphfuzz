import random
from typing import Callable, Any

import networkx as nx
from networkx.algorithms.flow import (
    edmonds_karp,
    shortest_augmenting_path,
    dinitz,
    boykov_kolmogorov,
    preflow_push,
)

from Tester.BaseTester import BaseTester
from Utils.FileUtils import save_discrepancies
from Utils.GraphConverter import GraphConverter


class MAXFVTesterAlgorithms:
    @staticmethod
    def edmonds_karp(graph: nx.Graph, source, target):
        return nx.maximum_flow_value(
            graph, source, target, flow_func=edmonds_karp, capacity="weight"
        )

    @staticmethod
    def shortest_augmenting_path(graph: nx.Graph, source, target):
        return nx.maximum_flow_value(
            graph, source, target, flow_func=shortest_augmenting_path, capacity="weight"
        )

    @staticmethod
    def dinitz(graph: nx.Graph, source, target):
        return nx.maximum_flow_value(
            graph, source, target, flow_func=dinitz, capacity="weight"
        )

    @staticmethod
    def boykov_kolmogorov(graph: nx.Graph, source, target):
        return nx.maximum_flow_value(
            graph, source, target, flow_func=boykov_kolmogorov, capacity="weight"
        )

    @staticmethod
    def preflow_push(graph: nx.Graph, source, target):
        return nx.maximum_flow_value(
            graph, source, target, flow_func=preflow_push, capacity="weight"
        )

    @staticmethod
    def igraph(graph: nx.Graph, source, target):
        converter = GraphConverter(graph)
        G_ig = converter.to_igraph()
        if "weight" not in G_ig.es.attribute_names():
            G_ig.es["weight"] = 1
        # Find iGraph indices for source and target
        source_ig = G_ig.vs.find(name=str(source)).index
        target_ig = G_ig.vs.find(name=str(target)).index

        return G_ig.maxflow(source_ig, target_ig, capacity="weight").value


class MAXFVTester(BaseTester):

    def __init__(self, discrepancy_filename="maxfv_corpus.pkl", id=None):
        super().__init__(discrepancy_filename)
        self.algorithms: dict[str, Callable[[nx.DiGraph, int, int], Any]] = {
            "edmonds-karp": MAXFVTesterAlgorithms.edmonds_karp,
            "shortest_augmenting_path": MAXFVTesterAlgorithms.shortest_augmenting_path,
            "dinitz": MAXFVTesterAlgorithms.dinitz,
            "boykov_kolmogorov": MAXFVTesterAlgorithms.boykov_kolmogorov,
            "preflow_push": MAXFVTesterAlgorithms.preflow_push,
            "igraph": MAXFVTesterAlgorithms.igraph,
        }

    def test(self, G):
        discrepancies = self.run_maxfv_tests_multiple_times(G)
        if discrepancies:
            return discrepancies
        return None

    def run_maxfv_tests_multiple_times(self, G, num_runs=5):
        """Run the test_maxfv_algorithms function multiple times with different source-target pairs."""
        discrepancies = {}

        nodes = list(G.nodes)
        if len(nodes) < 2:
            return discrepancies

        for _ in range(num_runs):
            # Randomly select source and target nodes
            source = random.choice(nodes)
            target = random.choice(nodes)
            while target == source:  # Ensure source and target are different
                target = random.choice(nodes)

            # Call the provided test function
            discrepancy, graph = self.test_algorithms(G, source, target)

            # If a discrepancy is found, add it to the dictionary
            if discrepancy is not None:
                # discrepancy_message = f"Source: {source}, Target: {target}, Discrepancy: {discrepancy}"
                discrepancy_message = f"Discrepancy: {discrepancy}"
                discrepancies[discrepancy_message] = graph

        return discrepancies

    def run(self):
        """Test maximum flow value algorithms on every graph in the corpus."""
        discrepancy_data = []
        discrepancy_counts = {}
        count = 0

        for G in self.corpus:
            # Ensure the graph has at least two nodes
            if len(G) < 2:
                print(f"Graph number {count + 1} has less than two nodes. Skipping...")
                continue

            # Sort nodes by degree in descending order
            sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

            # Consider top 10 and bottom 10 nodes
            top_10_nodes = sorted_nodes[:3] if len(sorted_nodes) >= 3 else sorted_nodes
            bottom_10_nodes = (
                sorted_nodes[-3:] if len(sorted_nodes) >= 3 else sorted_nodes
            )

            for source in top_10_nodes:
                for target in bottom_10_nodes:
                    if source != target:
                        count += 1
                        test_result, discrepancy_msg = self.test_maxfv_algorithms(
                            G, source, target
                        )
                        if not test_result:
                            discrepancy_counts[discrepancy_msg] = (
                                discrepancy_counts.get(discrepancy_msg, 0) + 1
                            )

                            if discrepancy_counts[discrepancy_msg] <= 5:
                                discrepancy_data.append((discrepancy_msg, G))

        if discrepancy_data:
            # Save the discrepancy graphs and messages to a file
            save_discrepancies(discrepancy_data, self.discrepancy_filename)
            print(f"There are {len(discrepancy_data)} graphs with discrepancies saved.")

            # Print all the discrepancy messages and their counts
        for msg, count in discrepancy_counts.items():
            print(f'Discrepancy message: "{msg}" occurred {count} times.')

        print("End of MAXFV testing.")
        return discrepancy_counts
