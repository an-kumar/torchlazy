import torch
import functools
from typing import Callable, TypeVar, List, Any
from .create import BuildCreator, InvokeCreator, push_creator, pop_creator

class BuiltModule(torch.nn.Module):
    def __init__(self, fn: Callable, created_list: List[Any]):
        super().__init__()
        self._fn = fn
        self.mods = torch.nn.ModuleList(created_list)
        self._creator = InvokeCreator(created_list)

    def forward(self, *args, **kwargs):
        push_creator(self._creator)

        r = self._fn(*args, **kwargs)
        self._creator.restart()

        pop_creator()
        return r

def build(fn: Callable, *args, **kwargs) -> torch.nn.Module:
    creator = BuildCreator()

    push_creator(creator)
    r = fn(*args, **kwargs)
    pop_creator()

    bound_fn = functools.partial(fn, **kwargs)


    return BuiltModule(bound_fn, creator.get_created())
