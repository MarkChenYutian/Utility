# Utility


<details markdown=1>
<summary markdown=1>
<h2>ClassHierarchy</h2>

Having too many classes in Python Project? Want to switch submoduels in system without changing any code? Try use the config-driven method to instantiate submodules in your system by subclassing the `RegisterClassRoot`.
</summary>

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
</details>


<details markdown=1>
<summary markdown=1>
<h2>SharedTensorPipe</h2>

**30%+ Throughput Improvement** on PyTorch tensor sharing compare to naive `mp.Pipe`.

Efficient serialization and sharing of torch.Tensor between processes via shared memory coalescing and reusing.
</summary>

Benchmarking and example code see `SharedTensorPipe/test.py`

```bash
# Reference: directly passing tensors through multiprocessing.Pipe
python -m SharedTensorPipe.test ref

# throughput: 3639msg [00:09, 364.60msg/s]
```

```bash
# Experiment: use SharedTensorPipe instead of mp.Pipe directly
python -m SharedTensorPipe.test exp

# throughput: 7250msg [00:14, 510.36msg/s]
```

</details>
