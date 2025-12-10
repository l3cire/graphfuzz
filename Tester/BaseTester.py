from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Any, Callable
import uuid

import networkx as nx

from Utils.FileUtils import save_discrepancies, save_discrepancy

T = TypeVar("T")


class TestMetamorphism(ABC):
    @abstractmethod
    def mutate(
        self, graph: nx.Graph, input: Any, result: T
    ) -> tuple[nx.Graph, Any, Callable[[T], bool]]:
        pass


class BaseTester(ABC):
    def __init__(
        self,
        corpus_path: str,
        discrepancy_filename: str,
        id=None,
        test_method: str = "differential",
        algorithm: str = None,
    ):
        self.corpus = []
        self.corpus_path = corpus_path
        self.discrepancy_filename = discrepancy_filename
        self.uuid = uuid.uuid4().hex[:8] if not id else id
        self.test_method = test_method
        self.algorithm = algorithm
        self.algorithms: dict[str, Callable] = {}
        print(f"Bug file id: {self.uuid}")

    @staticmethod
    @abstractmethod
    def get_test_metamorphism() -> TestMetamorphism:
        pass

    def test(
        self, graph: nx.Graph, timestamp: float, *args, **kwargs
    ) -> dict[str, nx.Graph]:
        if self.test_method == "differential":
            discrepancy_msg, discrepancy_graph = self.test_algorithms(graph)
        elif self.test_method == "metamorphic":
            alg = self.algorithms.get(self.algorithm, None)
            if alg is None:
                message = f"Incorrect algorithm name provided: {self.algorithm}"
                return {message: graph}
            discrepancy_msg, discrepancy_graph = self.test_metamorphic(
                graph, alg, *args
            )

        if discrepancy_msg:
            save_discrepancy(
                (discrepancy_msg, discrepancy_graph, timestamp),
                f"{self.discrepancy_filename}_{self.uuid}.pkl",
            )
            return {discrepancy_msg: discrepancy_graph}
        return {}

    def test_metamorphic(
        self,
        graph: nx.Graph,
        alg: Callable,
        *args,
        n_tries=10,
    ) -> tuple[Optional[str], Optional[nx.Graph]]:
        orig_result = alg(graph, *args)
        mutator: TestMetamorphism = self.get_test_metamorphism()
        for _ in range(n_tries):
            new_graph, new_args, checker = mutator.mutate(graph, args, orig_result)
            new_result = alg(new_graph, *new_args)
            if not checker(new_result):
                discrepancy_msg = (
                    f"Results for a graph and its mutation are inconsistent!"
                )
                return (discrepancy_msg, (graph, args, new_graph, new_args))
        return None, None

    def test_algorithms(
        self, graph: nx.Graph, *args, exception_result=None
    ) -> tuple[Optional[str], Optional[nx.Graph]]:
        results = {}
        for algo_name, algo_func in self.algorithms.items():
            try:
                results[algo_name] = algo_func(graph, *args)
            except Exception:
                results[algo_name] = exception_result

        discrepancy_messages = []
        algo_names = list(self.algorithms.keys())
        for i in range(len(algo_names)):
            for j in range(i + 1, len(algo_names)):
                algo_name1 = algo_names[i]
                algo_name2 = algo_names[j]
                result1 = results[algo_name1]
                result2 = results[algo_name2]
                if result1 != result2:
                    discrepancy_msg = f"Results of {algo_name1} and {algo_name2} are different for a graph!"
                    discrepancy_messages.append(discrepancy_msg)

        if discrepancy_messages:
            return (
                "--".join(discrepancy_messages),
                graph,
            )  # Return the concatenated discrepancy messages and the graph
        else:
            return None, None  # Return None if no discrepancy

    def run(self):
        discrepancy_data = []
        discrepancy_counts = {}
        for G in self.corpus:
            discrepancy_msg, discrepancy_graph = self.test(G)
            if discrepancy_msg:
                # Increment the discrepancy message count
                discrepancy_counts[discrepancy_msg] = (
                    discrepancy_counts.get(discrepancy_msg, 0) + 1
                )
                # Check the count to decide whether to save the graph
                if discrepancy_counts[discrepancy_msg] <= 5:
                    discrepancy_data.append((discrepancy_msg, discrepancy_graph))

        if discrepancy_data:
            # Save the discrepancy graphs and messages to a file
            save_discrepancies(discrepancy_data, self.discrepancy_filename)
            print(f"There are {len(discrepancy_data)} graphs with discrepancies saved.")

        # Now print all the discrepancy messages and their counts
        for msg, count in discrepancy_counts.items():
            print(f'Discrepancy message: "{msg}" occurred {count} times.')

        print("End of SCC testing.")
        return discrepancy_counts
