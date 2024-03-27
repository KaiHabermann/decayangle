from decayangle.backend import jax_backend, numpy_backend


class _cfg:
    state = {
        "backend": "numpy",
        "node_sorting": "value",
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
        if value not in ["value", "process_plane"]:
            raise ValueError(f"Node sorting {value} not found"
                             "Only 'value' and 'process_plane' are allowed"
                             "Default is 'value'")
        self.state["node_sorting"] = value
    
    def __value_sorting_key(self, value):
        """Get the sorting key of the node. 
        This is used to sort the daughters and make sure, that the order of the daughters is consistent.

        Returns:
            int: the sorting key
        """
        if isinstance(value, tuple):
            # this is a hack to make sure, that the order of the daughters is consistent
            # it will fail, if there are more than 10000 particles in the final state
            # but this is not realistic for the time being
            return -len(value) * 10000 + value[0]
        if isinstance(value, int):
            return abs(value)
        raise ValueError(f"Value {value} not understood for sorting")

    def __value_process_plane_sorting_key(self, value):
        raise NotImplementedError("Not implemented")
    
    def sorting_key(self, value):
        if self.node_sorting == "value":
            return self.__value_sorting_key(value)

        if self.node_sorting == "process_plane":
            return self.__value_process_plane_sorting_key(value)
        
        raise ValueError(f"Node sorting {self.node_sorting} not found")

config = _cfg()
