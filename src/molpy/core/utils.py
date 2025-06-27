from collections import defaultdict
from typing import List, Dict, Any


def to_dict_of_list(list_of_dict: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    result = defaultdict(list)
    for item in list_of_dict:
        for key, value in item.items():
            result[key].append(value)
    return dict(result)


def to_list_of_dict(list_of_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(list_of_dict.keys())
    length = len(next(iter(list_of_dict.values())))
    return [{key: list_of_dict[key][i] for key in keys} for i in range(length)]
