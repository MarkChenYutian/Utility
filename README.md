# Utility


## Color Pallet

A generated RGB 8-bit color pallet using the `husl` color pallet of seaborn package along with the color card.

## Class Hierarchy

Having too many classes in Python Project? Want to switch submoduels in system without changing any code? Try use the config-driven method to instantiate submodules in your system by subclassing the `RegisterClassRoot`.

```python
from Utility.ClassHierarchy import RegisterClassRoot

class A(RegisterClassRoot):
    def __init__(self, a) -> None:
        print("a", a)
class B(RegisterClassRoot):
    def __init__(self, b) -> None:
        print("b", b)
class C(B):
    def __init__(self, b) -> None:
        super().__init__(b)
        print("^^from C!")

RegisterClassRoot.Instantiate("/A", a=100)
RegisterClassRoot.Instantiate("/B/C", b=42)
RegisterClassRoot.show_hierarchy()
```

Output

```
a 100
b 42
^^from C!
RegisterClassRoot
| A
| B
| | C
```
