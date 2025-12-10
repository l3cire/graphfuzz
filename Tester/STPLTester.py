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
        try:
            return nx.bellman_ford_path_length(
                graph, source=source, target=target, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return float("inf")
        except (nx.NetworkXUnbounded):
            # Negative cycle exists
            return float("-inf")

    @staticmethod
    def dijkstra_path_length(graph, source, target):
        STPLTesterAlgorithms._set_default_weights(graph)
        try:
            return nx.dijkstra_path_length(
                graph, source=source, target=target, weight="weight"
            )
        except nx.NetworkXNoPath:
            return float("inf")

    @staticmethod
    def goldberg_radzik(graph, source, target):
        STPLTesterAlgorithms._set_default_weights(graph)
        try:
            _, dist = nx.goldberg_radzik(graph, source, weight="weight")
            return dist.get(target, float("inf"))
        except nx.NetworkXError:
            return float("inf")
        except nx.NetworkXUnbounded:
            # Negative cycle exists
            return float("-inf")

    @staticmethod
    def igraph(graph, source, target):
        # Check for negative cycle
        try:
            if nx.negative_edge_cycle(graph, weight="weight"):
                return float("-inf")
        except (nx.NetworkXError, nx.NetworkXUnbounded):
            return float("-inf")

        if graph.number_of_edges() == 0:
            return float("inf")

        STPLTesterAlgorithms._set_default_weights(graph)
        try:
            converter = GraphConverter(graph)
            graph_ig = converter.to_igraph()
            source_ig = graph_ig.vs.find(name=str(source)).index
            target_ig = graph_ig.vs.find(name=str(target)).index
            # Sanitize edge weights to avoid passing NaN or non-numeric
            # values into igraph C code (which aborts on NaN).
            try:
                raw_weights = graph_ig.es['weight'] if 'weight' in graph_ig.es.attribute_names() else None
            except Exception:
                raw_weights = None

            weights = None
            if raw_weights is not None:
                clean = []
                import math
                for w in raw_weights:
                    # If weight is a list (from consolidated multiedges), pick the minimum
                    if isinstance(w, (list, tuple)) and len(w) > 0:
                        try:
                            nums = [float(x) for x in w]
                            val = min(nums)
                        except Exception:
                            val = 1.0
                    else:
                        try:
                            val = float(w)
                        except Exception:
                            val = 1.0
                    # Replace NaN with a large finite weight (treat as effectively absent)
                    if math.isnan(val):
                        val = float('1e300')
                    clean.append(val)
                weights = clean

            shortest_paths = graph_ig.shortest_paths(
                source=source_ig, target=target_ig, weights=weights if weights is not None else "weight"
            )
            result = shortest_paths[0][0] if shortest_paths else float("inf")
            # igraph returns inf for no path
            return result if result != float("inf") else float("inf")
        except Exception:
            # If any error occurs (e.g., node not found), return inf
            return float("inf")


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

        # Compute source distances (dict) or None if negative cycle
        dist = self._source_distances(graph, source)

        # For negative cycle cases or when distances can't be computed,
        # we can still apply mutations that don't depend on distances
        if dist is None or result == float("-inf"):
            # Use only mutations that don't require distance information
            methods = [
                self._split_edge,
                self._scale_weights,
            ]
        else:
            # Use all mutations when distances are available
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
        except Exception as e:
            # On any error, return no-op mutation
            print(f"Error in mutation: {e}")
            return graph, input, (lambda _: True)

    def _add_edge_distance_based(self, graph, source, target, dist, orig_result):
        # add random edge (u,v,w) where w = d(v)-d(u) or +1
        # This requires valid distance information
        if dist is None:
            return graph, (source, target), (lambda _: True)

        graph_mut = graph.copy()
        nodes = [n for n in graph.nodes if n in dist and dist[n] != float("inf")]
        if len(nodes) < 2:
            return graph, (source, target), (lambda new_res: new_res == orig_result)

        for _ in range(50):
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u == v:
                continue
            du = dist.get(u, float("inf"))
            dv = dist.get(v, float("inf"))
            if du == float("inf") or dv == float("inf"):
                continue
            w_base = dv - du
            if not graph.is_directed():
                w_base = abs(w_base)
            w = w_base if random.random() < 0.5 else (w_base + 1)
            graph_mut.add_edge(u, v, weight=w)
            # The checker needs to handle inf and -inf properly
            def checker(new_res):
                # Both should be the same (including inf cases)
                return new_res == orig_result
            return graph_mut, (source, target), checker

        return graph, (source, target), (lambda new_res: new_res == orig_result)

    def _split_edge(self, graph, source, target, dist, orig_result):
        # pick existing edge (u,v,w) with weight >=1 and replace with u->a (1) and a->v (w-1)
        # This metamorphism works regardless of negative cycles
        graph_mut = graph.copy()
        edges = list(graph.edges(data=True))
        random.shuffle(edges)
        for u, v, data in edges:
            w = data.get("weight", 1)
            if w >= 1:
                new_node = max(graph_mut.nodes) + 1
                graph_mut.remove_edge(u, v)
                graph_mut.add_node(new_node)
                graph_mut.add_edge(u, new_node, weight=1)
                graph_mut.add_edge(new_node, v, weight=w - 1)
                def checker(new_res):
                    # Distance should remain the same (works for -inf too)
                    return new_res == orig_result
                return graph_mut, (source, target), checker

        return graph, (source, target), (lambda _: True)

    def _scale_weights(self, graph, source, target, dist, orig_result):
        # multiply all weights by positive constant k
        graph_mut = graph.copy()
        # pick k from reasonable set
        ks = [2, 3, 4]
        k = random.choice(ks)
        for _, _, data in graph_mut.edges(data=True):
            data["weight"] = data.get("weight", 1) * k

        def checker(new_res):
            # Special values (inf, -inf, nan) should remain unchanged
            if orig_result == float("inf"):
                return new_res == float("inf")
            if orig_result == float("-inf"):
                return new_res == float("-inf")
            # Check if both are nan (nan != nan, so we need special handling)
            if orig_result != orig_result:  # is nan
                return new_res != new_res  # should also be nan
            # For finite values, result should scale
            return abs(new_res - k * orig_result) < 1e-9

        return graph_mut, (source, target), checker

    def _add_long_new_path(self, graph, source, target, dist, orig_result):
        # Add path source -> ... -> target composed of new nodes whose total length > orig_result
        # This requires valid distance information
        if dist is None:
            return graph, (source, target), (lambda _: True)

        graph_mut = graph.copy()

        # If no path exists (orig_result is inf), this might create one
        # If path exists, we add a longer alternative path - distance should not change
        new_node = max(graph_mut.nodes) + 1
        graph_mut.add_node(new_node)

        # Calculate long path weight
        if orig_result == float("inf"):
            # Currently no path - adding any path will change the result
            # So we skip this metamorphism for inf case
            return graph, (source, target), (lambda _: True)
        elif orig_result == float("-inf"):
            # Negative cycle - adding a longer path won't change -inf result
            graph_mut.add_edge(source, new_node, weight=1)
            graph_mut.add_edge(new_node, target, weight=10)
        else:
            # Add a longer alternative path
            long_weight = abs(orig_result) + 10
            graph_mut.add_edge(source, new_node, weight=1)
            graph_mut.add_edge(new_node, target, weight=long_weight)

        def checker(new_res):
            # Shortest path should remain unchanged
            return new_res == orig_result

        return graph_mut, (source, target), checker

    def _add_short_new_path(self, graph, source, target, dist, orig_result):
        # Add path composed of new nodes with total length smaller than orig_result
        # This requires valid distance information
        if dist is None:
            return graph, (source, target), (lambda _: True)

        graph_mut = graph.copy()

        # Skip if no valid shorter path can be created
        if orig_result == float("inf") or orig_result == float("-inf") or orig_result <= 2:
            return graph, (source, target), (lambda _: True)

        # Create a shorter alternative path
        new_node = max(graph_mut.nodes) + 1
        graph_mut.add_node(new_node)

        # The new path should be shorter than the original
        short_weight = orig_result / 2
        graph_mut.add_edge(source, new_node, weight=1)
        graph_mut.add_edge(new_node, target, weight=short_weight - 1)

        expected_new_distance = short_weight

        def checker(new_res):
            # The new shortest path should use our shorter path
            return abs(new_res - expected_new_distance) < 1e-9

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
            source, target = random.sample(list(G.nodes()), 2)

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

                source, target = random.sample(list(G.nodes()), 2)

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
