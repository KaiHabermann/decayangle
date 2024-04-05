from decayangle.backend import jax_backend, numpy_backend


class _cfg:
    state = {
        "backend": "numpy",
        "node_sorting": "off",
    }
    backend_map = {
        "jax": jax_backend,
        "numpy": numpy_backend,
    }

    @property
    def backend(self):
        """The backend to use for the calculations

        Returns:
            module : The backend module
        """
        if self.state["backend"] not in self.backend_map:
            raise ValueError(f"Backend {self.state['backend']} not found")
        return self.backend_map[self.state["backend"]]

    @backend.setter
    def backend(self, value):
        if value not in self.backend_map:
            raise ValueError(f"Backend {value} not found")
        self.state["backend"] = value

    @property
    def node_sorting(self):
        return self.state["node_sorting"]

    @node_sorting.setter
    def node_sorting(self, value):
        if value not in ["off", "value", "process_plane"]:
            raise ValueError(
                f"Node sorting {value} not found"
                "Only 'value' is allowed for the time being"
            )
        self.state["node_sorting"] = value

    def __value_sorting_fun(self, value):
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
        
        raise ValueError(f"Value {value} of type {type(value)} not understood for sorting")

    def sorting_fun(self, value):
        if self.node_sorting == "value":
            return self.__value_sorting_fun(value)
        if self.node_sorting == "off":
            return value

        raise ValueError(f"Node sorting {self.node_sorting} not found")


config = _cfg()
