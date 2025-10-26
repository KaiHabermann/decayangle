from decayangle.backend import jax_backend, numpy_backend
from types import ModuleType
from typing import Tuple, List, Union, Optional
import multiprocessing


class _cfg:
    __state = {
        "backend": "numpy",
        "sorting": "value",
        "numerical_safety_checks": True,
        "gamma_tolerance": 1e-10,
        "shift_precision": 1e-10,
        "parallel_cores": None,
        "parallel_chunk_size": None,
    }
    backend_map = {
        "jax": jax_backend,
        "numpy": numpy_backend,
    }

    @property
    def backend(self) -> ModuleType:
        """The backend to use for the calculations

        Returns:
            module : The backend module
        """
        if self.__state["backend"] not in self.backend_map:
            raise ValueError(f"Backend {self.__state['backend']} not found")
        return self.backend_map[self.__state["backend"]]

    @backend.setter
    def backend(self, value: str):
        """
        Set the backend for the calculations

        Args:
            value (str): The backend to use
        """
        if value not in self.backend_map:
            raise ValueError(f"Backend {value} not found")
        self.__state["backend"] = value

    @property
    def sorting(self) -> str:
        """
        The sorting setting for the nodes

        Returns:
            str: The sorting setting
        """

        return self.__state["sorting"]

    @sorting.setter
    def sorting(self, value: str):
        """
        Set the sorting setting for the nodes

        Args:
            value (str): The sorting setting
        """

        if value not in ["off", "value"]:
            raise ValueError(
                f"Node sorting {value} not found"
                "Only 'value' and 'off' are allowed for the time being"
            )
        self.__state["sorting"] = value

    @property
    def numerical_safety_checks(self) -> bool:
        """
        The numerical safety checks setting indicating if the code should raise exceptions if numerical problems are detected

        Returns:
            bool: The numerical safety checks setting
        """

        return self.__state["numerical_safety_checks"]

    @numerical_safety_checks.setter
    def numerical_safety_checks(self, value: bool):
        """
        Set the numerical safety checks setting

        Args:
            value (bool): The numerical safety checks setting
        """

        if not isinstance(value, bool):
            raise ValueError(
                f"Value {value} or type {type(value)} not understood for numerical_safety_checks"
            )
        self.__state["numerical_safety_checks"] = value

    @property
    def gamma_tolerance(self) -> float:
        """
        The tolerance for the gamma factor to be considered as 1
        Used in cases where momenta in a rest frame are expected

        Returns:
            float: The tolerance for the gamma factor to be considered as 1
        """

        return self.__state["gamma_tolerance"]

    @gamma_tolerance.setter
    def gamma_tolerance(self, new_value: float):
        """
        Set the tolerance for the gamma factor to be considered as 1

        Args:
            new_value (float): The tolerance for the gamma factor to be considered as 1
        """
        self.__state["gamma_tolerance"] = new_value

    @property
    def shift_precision(self) -> float:
        """
        The precision at which the 2 pi flip is applied
        I.e. if Lambda(SU2)_rec as reconstructed from the angles decoded from the O(3,1) representation is within +- shift_precision of Lambda(SU2) which is obtained by applying all boosts and rotations, the 2 pi flip is not applied
        The 2 pi flip is applied when Lambda(SU2)_rec is within +- shift_precision of -Lambda(SU2)
        If none of both are true an exception is raised given numerical_safety_checks is set to True

        Returns:
            float: The precision at which the 2 pi flip is applied
        """

        return self.__state["shift_precision"]

    @shift_precision.setter
    def shift_precision(self, new_value):
        """
        Set the precision at which the 2 pi flip is applied

        Args:
            new_value (float): The precision at which the 2 pi flip is applied
        """
        self.__state["shift_precision"] = new_value

    @property
    def parallel_cores(self) -> Optional[Union[int, str]]:
        """
        The number of cores to use for parallel computation.
        Can be an integer, "auto" for automatic detection, or None to disable.

        Returns:
            Optional[Union[int, str]]: The parallel cores setting
        """
        return self.__state["parallel_cores"]

    @parallel_cores.setter
    def parallel_cores(self, value: Optional[Union[int, str]]):
        """
        Set the number of cores for parallel computation.

        Args:
            value (Optional[Union[int, str]]): The parallel cores setting.
                - int: Number of cores to use
                - "auto": Use available cores - 1
                - None: Disable parallel computation
        """
        if value is not None and value != "auto" and not isinstance(value, int):
            raise ValueError(
                f"parallel_cores must be an integer, 'auto', or None, got {type(value)}"
            )
        if isinstance(value, int) and value <= 0:
            raise ValueError("parallel_cores must be a positive integer")
        self.__state["parallel_cores"] = value

    @property
    def parallel_chunk_size(self) -> Optional[Union[int, str]]:
        """
        The chunk size for parallel computation over arrays.
        Can be an integer, "auto" for automatic detection, or None to disable chunking.

        Returns:
            Optional[Union[int, str]]: The parallel chunk size setting
        """
        return self.__state["parallel_chunk_size"]

    @parallel_chunk_size.setter
    def parallel_chunk_size(self, value: Optional[Union[int, str]]):
        """
        Set the chunk size for parallel computation.

        Args:
            value (Optional[Union[int, str]]): The parallel chunk size setting.
                - int: Chunk size to use
                - "auto": Use 50,000 as default chunk size
                - None: Disable chunking
        """
        if value is not None and value != "auto" and not isinstance(value, int):
            raise ValueError(
                f"parallel_chunk_size must be an integer, 'auto', or None, got {type(value)}"
            )
        if isinstance(value, int) and value <= 0:
            raise ValueError("parallel_chunk_size must be a positive integer")
        self.__state["parallel_chunk_size"] = value

    def get_parallel_cores(
        self, value: Optional[Union[int, str]] = None
    ) -> Optional[int]:
        """
        Get the effective number of cores for parallel computation.
        Resolves "auto" to actual number of cores - 1.

        Args:
            value: Optional override value. If provided, uses this instead of config value.

        Returns:
            Optional[int]: The effective number of cores, or None if disabled
        """
        cores_value = value if value is not None else self.parallel_cores
        if cores_value is None:
            return None
        if cores_value == "auto":
            return max(1, multiprocessing.cpu_count() - 1)
        return cores_value

    def get_parallel_chunk_size(
        self, value: Optional[Union[int, str]] = None
    ) -> Optional[int]:
        """
        Get the effective chunk size for parallel computation.
        Resolves "auto" to 50,000.

        Args:
            value: Optional override value. If provided, uses this instead of config value.

        Returns:
            Optional[int]: The effective chunk size, or None if disabled
        """
        chunk_value = value if value is not None else self.parallel_chunk_size
        if chunk_value is None:
            return None
        if chunk_value == "auto":
            return 50_000
        return chunk_value

    def __value_sorting_fun(
        self, value: Union[int, tuple, list]
    ) -> Union[int, tuple, list]:
        """Sort the value by length of the tuple first and then by absolute value of the integers
        Two tuples of the same length are sorted by the first element

        Returns:
            the sorted value
        """

        def key(value):
            if isinstance(value, tuple):
                # this is a hack to make sure, that the order of the daughters is consistent
                # it will fail, if there are more than 10000 particles in the final state
                # but this is not realistic for the time being
                return -len(value) * 10000 + value[0]
            if isinstance(value, int):
                return abs(value)

            raise ValueError(
                f"Value {value} of type {type(value)} not understood for sorting"
            )

        if isinstance(value, int):
            return value

        if isinstance(value, tuple):
            return tuple(sorted(value, key=key))

        if isinstance(value, list):
            return sorted(value, key=key)

        raise ValueError(
            f"Value {value} of type {type(value)} not understood for sorting"
        )

    def ordering_function(
        self, value: Union[int, tuple, list]
    ) -> Union[int, tuple, list]:
        """
        The ordering function for the nodes in the topologies
        This function is used to handle the default cases of the sorting.
        For custom sorting functions, the ordering_function can be passed to the TopologyCollection or Topology constructor.
        See the documentation of the TopologyCollection or Topology for more information.

        Args:
            value (Union[int, tuple, list]): The value to sort

        """
        if self.sorting == "value":
            return self.__value_sorting_fun(value)
        if self.sorting == "off":
            return value

        raise ValueError(f"Node sorting {self.sorting} not found")

    def raise_if_safety_on(self, exception: Exception):
        """
        Raise the exception if numerical_safety_checks is set to True

        Args:
            exception (Exception): The exception to raise
        """
        if self.numerical_safety_checks:
            raise exception


config = _cfg()
