from typing import Callable, Any
import random

import networkx as nx

from Utils.FileUtils import save_discrepancy
from Utils.GraphConverter import GraphConverter
from Tester.BaseTester import BaseTester, TestMetamorphism


class SCCTesterAlgorithms:
    @staticmethod
    def _wrapper(result):
        return set(map(frozenset, list(result)))

    @staticmethod
    def default(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(nx.strongly_connected_components(graph))

    @staticmethod
    def recursive(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(
            nx.strongly_connected_components_recursive(graph)
        )

    @staticmethod
    def kosaraju(graph: nx.DiGraph):
        return SCCTesterAlgorithms._wrapper(
            nx.kosaraju_strongly_connected_components(graph)
        )

    @staticmethod
    def igraph(graph: nx.DiGraph):
        converter = GraphConverter(graph)
        graph_ig = converter.to_igraph_default()
        return set(
            frozenset(map(int, graph_ig.vs[component]["_nx_name"]))
            for component in graph_ig.components(mode="STRONG")
        )


class SCCTestMetamorphism(TestMetamorphism):
    def mutate(
        self, graph: nx.Graph, input: Any, result: set[frozenset[int]]
    ) -> tuple[nx.Graph, Any, Callable[[set[frozenset[int]]], bool]]:
        if len(graph.nodes) == 0:
            return graph, input, (lambda _: True)

        graph_mutated, expected_result = self.compose_methods(graph, result)
        checker = lambda res: (len(res) == len(expected_result))
        return graph_mutated, input, checker

    def compose_methods(
        self, graph: nx.DiGraph, result: set[frozenset[int]], max_compositions=4
    ):
        n_compositions = random.randint(1, max_compositions)
        all_methods = [
            self.add_edge_inside_component,
            self.remove_edge_between_components,
            self.add_path_inside_component,
            self.add_cycle_component,
            self.add_isolated_node,
        ]
        new_graph, new_result = graph, result
        for _ in range(n_compositions):
            method = random.choice(all_methods)
            new_graph, new_result = method(new_graph, new_result)
        return new_graph, new_result

    def add_edge_inside_component(self, graph: nx.DiGraph, result: set[frozenset[int]]):
        component = list(random.choice(list(result)))
        start, end = (random.choice(component), random.choice(component))
        graph_mutated = graph.copy()
        graph_mutated.add_edge(start, end)
        return graph_mutated, result

    def remove_edge_between_components(
        self, graph: nx.DiGraph, result: set[frozenset[int]]
    ):
        graph_mutated = graph.copy()
        for _ in range(100):
            out_component = random.choice(list(result))
            start_node = random.choice(list(out_component))
            if len(graph.edges(start_node)) == 0:
                continue
            edge = random.choice(list(graph.edges(start_node)))
            if edge[1] in out_component:
                continue
            graph_mutated.remove_edge(edge[0], edge[1])
            return graph_mutated, result
        return graph_mutated, result

    def add_path_inside_component(
        self, graph: nx.DiGraph, result: set[frozenset[int]], max_vertices=5
    ):
        graph_mutated = graph.copy()
        component = random.choice(list(result))
        start, end = (random.choice(list(component)), random.choice(list(component)))

        n_new_nodes = random.randint(1, max_vertices)
        new_nodes = [max(graph.nodes) + 1 + i for i in range(n_new_nodes)]
        graph_mutated.add_nodes_from(new_nodes)

        path = [start] + new_nodes + [end]
        graph_mutated.add_edges_from(
            [(path[i - 1], path[i]) for i in range(1, len(path))]
        )

        new_result = result.copy()
        new_result.remove(component)
        new_result.add(component.union(new_nodes))
        return graph_mutated, new_result

    def add_isolated_node(self, graph: nx.DiGraph, result: set[frozenset[int]]):
        graph_mutated = graph.copy()

        # create a single new node id after current max
        new_node = max(graph.nodes) + 1
        graph_mutated.add_node(new_node)

        new_result = result.copy()
        # an isolated node is its own SCC
        new_result.add(frozenset({new_node}))
        return graph_mutated, new_result

    def add_cycle_component(
        self, graph: nx.DiGraph, result: set[frozenset[int]], max_vertices=5
    ):
        graph_mutated = graph.copy()

        n_new_nodes = random.randint(1, max_vertices)
        new_nodes = [max(graph.nodes) + 1 + i for i in range(n_new_nodes)]
        graph_mutated.add_nodes_from(new_nodes)
        graph_mutated.add_edges_from(
            [(new_nodes[i - 1], new_nodes[i]) for i in range(len(new_nodes))]
        )

        edge_prob = random.random()
        edge_dir = random.randint(0, 1)
        for node in graph.nodes:
            if random.random() >= edge_prob:
                continue
            if edge_dir == 0:
                graph_mutated.add_edge(node, random.choice(new_nodes))
            else:
                graph_mutated.add_edge(random.choice(new_nodes), node)

        new_result = result.copy()
        new_result.add(frozenset(new_nodes))
        return graph_mutated, new_result


class SCCTester(BaseTester):

    def __init__(
        self, corpus_path, discrepancy_filename="scc_discrepancy", *args, **kwargs
    ):
        super().__init__(corpus_path, discrepancy_filename, *args, **kwargs)
        self.algorithms: dict[str, Callable[[nx.DiGraph], Any]] = {
            "default": SCCTesterAlgorithms.default,
            "recursive": SCCTesterAlgorithms.recursive,
            "kosaraju": SCCTesterAlgorithms.kosaraju,
            "igraph": SCCTesterAlgorithms.igraph,
        }

    @staticmethod
    def get_test_metamorphism():
        return SCCTestMetamorphism()
