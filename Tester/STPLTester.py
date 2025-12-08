import random
from typing import Any

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


class STPLTestMetamorphism:
    """Metamorphism implementations for shortest-path-length testing.

    Each mutation returns (mutated_graph, new_args, checker) where
    `new_args` are the (source, target) tuple and `checker` validates
    the algorithm output on the mutated graph.
    """

    def _source_distances(self, graph: nx.Graph, source):
        try:
            # prefer Dijkstra when no negative weights
            has_negative = any(
                data.get("weight", 0) < 0 for _, _, data in graph.edges(data=True)
            )
            if has_negative:
                # may raise if negative cycle present
                return nx.single_source_bellman_ford_path_length(graph, source, weight="weight")
            else:
                return nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        except Exception:
            return None

    def mutate(self, graph: nx.Graph, input: Any, result: float):
        print("start of mutate")
        # input is tuple (source, target)
        if not input:
            return graph, input, (lambda _: True)
        source, target = input

        if len(graph) < 1:
            return graph, input, (lambda _: True)

        # compute source distances (dict) or bail out
        dist = self._source_distances(graph, source)
        if dist is None:
            return graph, input, (lambda _: True)

        methods = [
            self._add_edge_distance_based,
            self._split_edge,
            self._scale_weights,
            self._add_long_new_path,
            self._add_short_new_path,
        ]

        method = random.choice(methods)
        try:
            print("mutating using", method.__name__)
            return method(graph, source, target, dist, result)
        except Exception:
            # On any error, return no-op mutation
            return graph, input, (lambda _: True)

    def _add_edge_distance_based(self, graph, source, target, dist, orig_result):
        # add random edge (u,v,w) where w = d(v)-d(u) or +1
        graph_mut = graph.copy()
        nodes = [n for n in graph.nodes if n in dist and dist[n] != float("inf")]
        if len(nodes) < 2:
            return graph, (source, target), (lambda new_res: orig_result == new_res)

        for _ in range(50):
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u == v:
                continue
            du = dist.get(u, float("inf"))
            dv = dist.get(v, float("inf"))
            w_base = dv - du
            w = w_base if random.random() < 0.5 else (w_base + 1)
            graph_mut.add_edge(u, v, weight=w)
            checker = lambda new_res: new_res == orig_result
            return graph_mut, (source, target), checker

        return graph, (source, target), (lambda new_res: orig_result == new_res)

    def _split_edge(self, graph, source, target, dist, orig_result):
        # pick existing edge (u,v,w) with weight >=1 and replace with u->a (1) and a->v (w-1)
        graph_mut = graph.copy()
        edges = list(graph.edges(data=True))
        random.shuffle(edges)
        for u, v, data in edges:
            w = data.get("weight", 1)
            if w >= 1:
                new_node = max(graph.nodes) + 1
                graph_mut.remove_edge(u, v)
                graph_mut.add_node(new_node)
                graph_mut.add_edge(u, new_node, weight=1)
                graph_mut.add_edge(new_node, v, weight=w - 1)
                checker = lambda new_res: new_res == orig_result
                return graph_mut, (source, target), checker

        return graph, (source, target), (lambda _: True)

    def _scale_weights(self, graph, source, target, dist, orig_result):
        # multiply all weights by positive constant k
        graph_mut = graph.copy()
        # pick k from reasonable set
        ks = [2, 3, 4]
        k = random.choice(ks)
        for u, v, data in graph_mut.edges(data=True):
            data["weight"] = data.get("weight", 1) * k

        def checker(new_res):
            if orig_result in (float("inf"), float("-inf"), float("nan")):
                return new_res == orig_result
            return new_res == k * orig_result

        return graph_mut, (source, target), checker

    def _add_long_new_path(self, graph, source, target, dist, orig_result):
        # Add path source -> ... -> target composed of new nodes whose total length > orig_result
        graph_mut = graph.copy()
        # create new nodes
        new_node = max(graph.nodes) + 1
        graph_mut.add_node(new_node)
        graph_mut.add_edge(source, new_node, 1)
        graph_mut.add_edge(new_node, target, dist+10)
        checker = lambda new_res: new_res == orig_result
        return graph_mut, (source, target), checker

    def _add_short_new_path(self, graph, source, target, dist, orig_result):
        # Add path composed of new nodes with total length smaller than orig_result
        graph_mut = graph.copy()
        # create new nodes
        new_node = max(graph.nodes) + 1
        graph_mut.add_node(new_node)
        graph_mut.add_edge(source, new_node, 1)
        graph_mut.add_edge(new_node, target, dist / 2)
        checker = lambda new_res: new_res == dist / 2
        return graph_mut, (source, target), checker



class STPLTester(BaseTester):
    def __init__(
        self, coprus_path, discrepancy_filename="stpl_discrepancy", *args, **kwargs
    ):
        super().__init__(coprus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms = {
            "bellman_ford_path_length": STPLTesterAlgorithms.bellman_ford_path_length,
            "goldberg_radzik": STPLTesterAlgorithms.goldberg_radzik,
            "dijkstra_path_length": STPLTesterAlgorithms.dijkstra_path_length,
            "igraph": STPLTesterAlgorithms.igraph,
        }

    def test(self, G, timestamp, num_pairs=10):
        total_discrepancies = {}

        if len(G) < 2:
            return {}

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
                total_discrepancies[discrepancy_msg] = discrepancy_graph

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

    @staticmethod
    def get_test_metamorphism() -> Any:
        return STPLTestMetamorphism()
