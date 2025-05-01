from typing import Any, Tuple, TypeVar

__version__ = "1.1.1"


ClassTopK = TypeVar("ClassTopK", bound="TopK")
class TopK():
    """
    Classe permettant de gérer le top K de valeurs en gardant les informations associées
    """
    def __init__(self, k: int=9) -> None:
        self.k = k
        self.top = []
        self.data = []

    def append_one(self, value: float, data: Any) -> None:
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

    def append(self, values: list[float], data: list[Any]) -> None:
        if type(values) != list:
            self.append_one(values, data)
            return
        
        assert type(data) == list, f"data should be a list, but got {type(data)}"
        assert len(values) == len(data), f"values and data should have the same length, but got {len(values)} and {len(data)}"
        for value, d in zip(values, data):
            self.append_one(value, d)

    def __len__(self) -> int:
        return len(self.top)
    
    def __getitem__(self, idx: int) -> Tuple[float, Any]:
        if idx < 0:
            idx += len(self.top)
        assert 0 <= idx and idx < len(self.top), f"idx {idx} should be in [-{len(self.top)}; {len(self.top)}["
        return self.top[idx], self.data[idx]

    def __str__(self) -> str:
        return str(list(zip(self.top, self.data)))  

    def __eq__(self, value: list|ClassTopK) -> bool:
        if isinstance(value, TopK):
            value = list(zip(value.top, value.data))
        return list(zip(topk.top, topk.data)) == value

    
if __name__ == "__main__":
    import numpy as np

    l = [(v, f"{v}-{v}") for v in np.arange(90, 99)]
    np.random.shuffle(l)
    print("list l:", l)
    expected_topk_with_l = [(98, '98-98'), (97, '97-97'), (96, '96-96'), (95, '95-95'), (94, '94-94'), (93, '93-93'), (92, '92-92'), (91, '91-91'), (90, '90-90')]
    print("Expected topk with l:", expected_topk_with_l)

    # Test of TopK.__str__()
    topk = TopK(9)
    topk.append_one(1, "1")
    topk.append_one(2, "2")
    topk.append_one(3, "3")
    expected = [(3, "3"), (2, "2"), (1, "1")]
    assert str(topk) == str(expected), f"Should be {expected} instead of {str(topk)}"
    print("-- Tests of TopK.__str__() passed")

    # Test of TopK.__len__()
    topk = TopK(9)
    assert len(topk) == 0, f"Should be 0"
    for v, d in l:
        topk.append_one(v, d)
    expected = len(l)
    assert len(topk) == expected, f"Should be {expected} instead of {len(topk)}"
    print("-- Tests of TopK.__len__() passed")

    # Test of TopK.__getitem__()
    topk = TopK(9)
    topk.append_one(1, "1")
    topk.append_one(2, "2")
    topk.append_one(3, "3")
    tests_idx = [0, 1, 2, -1, -2]
    expecteds = [(3, "3"), (2, "2"), (1, "1"), (1, "1"), (2, "2")]
    for idx, expected in zip(tests_idx, expecteds):
        assert topk[idx] == expected, f"Should be {expected} instead of {topk[idx]} for idx {idx}"
    print("-- Tests of TopK.__getitem__() passed")

    # Test of TopK.__eq__()
    topk = TopK(9)
    for v, d in l:
        topk.append_one(v, d)
    expected = expected_topk_with_l
    assert topk == expected, f"Should be equal"
    
    topk2 = TopK(9)
    for v, d in reversed(l):
        topk2.append_one(v, f"{v}-{v}")
    assert topk == topk2, f"Should be equal"

    # Test of TopK.append_one()
    topk = TopK(9)
    for v, d in l:
        topk.append_one(v, d)
    expected = expected_topk_with_l
    assert topk == expected, f"Should be {expected}"
    print("-- Tests of TopK.append_one() passed")
     
    # Test of TopK.append()
    topk = TopK(9)
    topk.append([v for v, d in l], [d for v, d in l])
    expected = expected_topk_with_l
    assert topk == expected, f"Should be {expected}"

    topk = TopK(9)
    for v, d in l:
        topk.append(v, d)
    expected = expected_topk_with_l
    assert topk == expected, f"Should be {expected}"
    print("-- Tests of TopK.append() passed")

    print("== All tests passed =======")