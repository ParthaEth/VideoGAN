import sys
sys.path.append('../eg3d')
from torch_utils import misc, persistence
import dataclasses
@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class Foo():
    i:int = 1

    def __post_init__(self):
        super().__init__()


print('Before __init__')
foo = Foo(5)
print('After __init__')
print(foo.i)