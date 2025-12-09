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

from Tester.BaseTester import BaseTester, TestMetamorphism
from Utils.FileUtils import save_discrepancies, save_discrepancy
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


class MAXFVTestMetramorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: tuple[int, int], result: int
    ) -> tuple[nx.Graph, tuple[int, int], Callable[[int], bool]]:
        if len(graph.nodes) <= 1:
            return graph, input, (lambda _: True)
        graph_mutated, source, sink, exp_res = self.compose_methods(
            graph, input[0], input[1], result
        )
        checker = lambda res: (res == exp_res)
        return graph_mutated, (source, sink), checker

    def compose_methods(
        self, graph: nx.Graph, source: int, sink: int, result: int, max_compositions=4
    ):
        n_compositions = random.randint(1, max_compositions)
        all_methods = [
            self.add_source_sink_link,
            self.add_endpoint_node,
            self.swap_source_sink,
        ]
        new_graph, new_source, new_sink, new_result = graph, source, sink, result
        for _ in range(n_compositions):
            method = random.choice(all_methods)
            new_graph, new_source, new_sink, new_result = method(
                new_graph, new_source, new_sink, new_result
            )
        return new_graph, new_source, new_sink, new_result

    def add_source_sink_link(
        self, graph: nx.Graph, source: int, sink: int, result: int, max_weight=10
    ):
        if graph.has_edge(source, sink):
            return graph, source, sink, result

        weight = random.randint(1, max_weight)
        new_graph = graph.copy()
        new_graph.add_edge(source, sink, weight=weight)
        return new_graph, source, sink, result + weight

    def add_endpoint_node(self, graph: nx.Graph, source: int, sink: int, result: int):
        new_node = max(list(graph.nodes)) + 1
        new_graph = graph.copy()
        new_graph.add_node(new_node)
        weight = random.randint(1, result + 1)
        if random.choice([True, False]):
            new_graph.add_edge(new_node, source, weight=weight)
            return new_graph, new_node, sink, min(result, weight)
        else:
            new_graph.add_edge(sink, new_node, weight=weight)
            return new_graph, source, new_node, min(result, weight)

    def swap_source_sink(self, graph: nx.Graph, source: int, sink: int, result: int):
        if graph.is_directed():
            return nx.reverse(graph), sink, source, result
        return graph, sink, source, result


class MAXFVTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="maxfv_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph, int, int], Any]] = {
            "edmonds-karp": MAXFVTesterAlgorithms.edmonds_karp,
            "shortest_augmenting_path": MAXFVTesterAlgorithms.shortest_augmenting_path,
            "dinitz": MAXFVTesterAlgorithms.dinitz,
            "boykov_kolmogorov": MAXFVTesterAlgorithms.boykov_kolmogorov,
            "preflow_push": MAXFVTesterAlgorithms.preflow_push,
            "igraph": MAXFVTesterAlgorithms.igraph,
        }

    def get_test_metamorphism(self):
        return MAXFVTestMetramorphism()

    def test(self, G, timestamp):
        return self.run_maxfv_tests_multiple_times(G, timestamp)

    def run_maxfv_tests_multiple_times(self, G, timestamp, num_runs=5):
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

            discrepancies = super().test(G, timestamp, source, target)
            if len(discrepancies) > 0:
                return discrepancies

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
