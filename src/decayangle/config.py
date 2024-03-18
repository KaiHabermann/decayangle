from decayangle.backend import jax_backend, numpy_backend

class _cfg:
    state = {
        "backend": "numpy",
    }
    backend_map = {
        "jax": jax_backend,
        "numpy": numpy_backend,
    }

    @property
    def backend(self):
        if self.state["backend"] not in self.backend_map:
            raise ValueError(f"Backend {self.state['backend']} not found")
        return self.backend_map[self.state["backend"]]
    
    @backend.setter
    def backend(self, value):
        if value not in self.backend_map:
            raise ValueError(f"Backend {value} not found")
        self.state["backend"] = value
    
config = _cfg()