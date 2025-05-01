from typing import Any, Tuple

__version__ = "1.0.1"

class TopK():
    """
    Classe permettant de gérer le top K de valeurs en gardant les informations associées
    """
    def __init__(self, k: int=9) -> None:
        self.k = k
        self.top = []
        self.data = []

    def append(self, value: float, data: Any) -> None:
        if len(self.top) < self.k:
            self.top.append(value)
            self.data.append(data)
        else:
            if self.top[-1] < value:
                self.top.pop()
                self.data.pop()
                self.top.append(value)
                self.data.append(data)
            else:
                return
        self.top, self.data = [list(t) for t in zip(*sorted(zip(self.top, self.data), reverse=True))]

    def __len__(self) -> int:
        return len(self.top)
    
    def __getitem__(self, idx: int) -> Tuple[float, Any]:
        idx += len(self.top)
        assert 0 <= idx and idx < len(self.top), f"idx should be in [-{len(self.top)}; {len(self.top)}["
        return self.top[idx], self.data[idx]

    def __str__(self) -> str:
        return str(list(zip(self.top, self.data)))  

    def __eq__(self, value: list) -> bool:
        return list(zip(topk.top, topk.data)) == value

    
if __name__ == "__main__":
    import numpy as np

    l = np.arange(99)
    np.random.shuffle(l)

    topk = TopK(9)

    for v in l:
        topk.append(v, f"{v}-{v}")
    expected = [(98, '98-98'), (97, '97-97'), (96, '96-96'), (95, '95-95'), (94, '94-94'), (93, '93-93'), (92, '92-92'), (91, '91-91'), (90, '90-90')]
    assert topk == expected, f"Should be {expected}"
    print("-- Tests of TopK passed")
     
    print("== All tests passed =======")