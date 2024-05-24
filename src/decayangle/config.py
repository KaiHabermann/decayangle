from decayangle.backend import jax_backend, numpy_backend
from types import ModuleType
from typing import Tuple, List, Union


class _cfg:
    __state = {
        "backend": "numpy",
        "sorting": "value",
        "numerical_safety_checks": True,
        "gamma_tolerance": 1e-10,
        "shift_precision": 1e-10,
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
    def backend(self, value) -> str:
        if value not in self.backend_map:
            raise ValueError(f"Backend {value} not found")
        old_value = self.__state["backend"]
        self.__state["backend"] = value
        return old_value

    @property
    def sorting(self) -> str:
        return self.__state["sorting"]

    @sorting.setter
    def sorting(self, value) -> str:
        if value not in ["off", "value"]:
            raise ValueError(
                f"Node sorting {value} not found"
                "Only 'value' and 'off' are allowed for the time being"
            )
        old_value = self.__state["sorting"]
        self.__state["sorting"] = value
        return old_value

    @property
    def numerical_safety_checks(self) -> bool:
        return self.__state["numerical_safety_checks"]

    @numerical_safety_checks.setter
    def numerical_safety_checks(self, value: bool) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"Value {value} or type {type(value)} not understood for numerical_safety_checks"
            )
        old_value = self.__state["numerical_safety_checks"]
        self.__state["numerical_safety_checks"] = value
        return old_value

    @property
    def gamma_tolerance(self) -> float:
        return self.__state["gamma_tolerance"]

    @gamma_tolerance.setter
    def gamma_tolerance(self, new_value: float) -> float:
        old_value = self.gamma_tolerance
        self.__state["gamma_tolerance"] = new_value
        return old_value

    @property
    def shift_precision(self) -> float:
        return self.__state["shift_precision"]

    @shift_precision.setter
    def shift_precision(self, new_value) -> float:
        old_value = self.shift_precision
        self.__state["shift_precision"] = new_value
        return old_value

    def __value_sorting_fun(
        self, value: Union[int, tuple, list]
    ) -> Union[int, tuple, list]:
        """Sort the value by lenght of the tuple first and then by absolute value of the integers
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

    def ordering_function(self, value):
        if self.sorting == "value":
            return self.__value_sorting_fun(value)
        if self.sorting == "off":
            return value

        raise ValueError(f"Node sorting {self.sorting} not found")

    def raise_if_safety_on(self, exception: Exception):
        if self.numerical_safety_checks:
            raise exception


config = _cfg()
