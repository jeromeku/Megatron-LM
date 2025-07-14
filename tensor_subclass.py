from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import TorchDispatchMode, transform_subclass, is_traceable_wrapper_subclass, return_and_correct_aliasing

import torch, functools
import contextlib

@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

def _make(cls, data: torch.Tensor):
    return torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)  # helper

class First(torch.Tensor):
    @staticmethod
    def __new__(cls, data): return _make(cls, data)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print("First handles", func.__name__)
        # fall through to the next candidate
        with no_dispatch():
            return func(*args, **(kwargs or {}))

class Second(torch.Tensor):
    @staticmethod
    def __new__(cls, data): return _make(cls, data)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print("Second handles", func.__name__)
        with no_dispatch():
            return func(*args, **(kwargs or {}))

a = First(torch.eye(2))
b = Second(torch.eye(2))

print("a @ b  -->")
torch.matmul(a, b)       # First is left-most → First runs first

print("\nb @ a  -->")
torch.matmul(b, a)       # Second is left-most → Second runs first
