from typing import Callable, TypeVar, List, Any, Protocol, Optional

T = TypeVar('T')

class Creator(Protocol):
    def create(self, fn: Callable[..., Any], *args, **kwargs):
        ...

class BuildCreator:
    def __init__(self):
        self._created: List[Any] = []

    def create(self, fn: Callable[..., Any], *args, **kwargs):
        r = fn(*args, **kwargs)
        self._created.append(r)
        return r

    def get_created(self):
        return self._created

class InvokeCreator:
    def __init__(self, created_list: List[Any]):
        self._created_list = created_list
        self._curr = 0

    def create(self, fn: Callable[..., Any], *args, **kwargs):
        if self._curr >= len(self._created_list):
            raise RuntimeError(
                    "Requesting more created values than were created during `build`."
                    )

        r = self._created_list[self._curr]
        self._curr += 1
        return r

    def restart(self):
        if self._curr != len(self._created_list):
            raise RuntimeError(
                    "Resetting `CreationList` before returning all created values. Got to: {}. Total length: {}".format(self._curr, len(self._created_list))
                    )
        self._curr = 0

_g_creator_stack: List[Creator] = []

def get_creator():
    global _g_creator_stack
    return _g_creator_stack[-1]

def push_creator(c: Creator):
    global _g_creator_stack
    _g_creator_stack.append(c)

def pop_creator():
    global _g_creator_stack
    _g_creator_stack.pop()

def create(fn: Callable[..., T], *args, **kwargs) -> T:
    """
    Lazily create a value, such as a torch.nn.Module.

    Args:
        fn: A function that returns anything, though usually this would be a `torch.nn.Module`.
        args, kwargs: Arguments to pass to `fn`.

    Returns:
        The returned value of `fn`. When called during `build`, this will be created from scratch
        and stored, to be retrieved when subsequently called.
    """
    creator = get_creator()
    if creator is None:
        raise RuntimeError(
                "Calling `torchlazy.create` outside of the scope of `a torchlazy.BuiltModule`."
                )

    return creator.create(fn, *args, **kwargs)
