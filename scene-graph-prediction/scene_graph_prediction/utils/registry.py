# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import MutableMapping, TypeVar, Any, Callable

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class Registry(dict, MutableMapping[_KT, _VT]):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Warning: if absolute imports are done from within scene_graph_prediction and the path is not absolute,
             then some modules may be registered twice.
                 Example structure:
                 scene_graph_prediction:
                    engine
                    modeling
                If in scene_graph_prediction.engine, the modeling submodule is imported as modeling
                rather than scene_graph_prediction.modeling, then modules will be registered twice...

    Example: creating a registry:
        some_registry = Registry({"default": default_module})

    There are two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def register(self, module_name: str, module: Any = None) -> Callable[[Callable], Callable] | None:
        # used as function call
        if module is not None:
            assert module_name not in self
            self[module_name] = module
            return

        # used as decorator
        def register_fn(fn: Callable) -> Callable:
            assert module_name not in self
            self[module_name] = fn
            return fn

        return register_fn
