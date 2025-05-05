
from typing import Any


class UniqueCounter:
    """
    A class to count unique items.
    """

    def __init__(self):
        self.counter = 0
        self.unique_items = {}

    def add(self, item):
        """
        Add an item to the counter.
        """
        if item not in self.unique_items:
            self.counter += 1
            self.unique_items[item] = self.counter
        return self.unique_items[item]

    def __getitem__(self, item):
        return self.unique_items[item]
    
    def __contains__(self, item):
        return item in self.unique_items

class TagApplyer:
    """
    A class to apply tags to a list of items.
    """

    def __init__(self):
        self.dollar_mapping = {}

    def apply_tags(self, items: list[dict[str, Any]]):
        """
        Apply tags to the items.
        """
        for item in items:
            for key, value in list(item.items()):
                if key.startswith("$"):
                    item[key[1:]] = self.resolve_dollar(key[1:], value)
    
    def resolve_dollar(self, key, value):
        """
        Resolve the dollar sign in the value.
        """
        if key not in self.dollar_mapping:
            self.dollar_mapping[key] = UniqueCounter()

        if value in self.dollar_mapping[key]:
            return self.dollar_mapping[key][value]
        
        return self.dollar_mapping[key].add(value)
            

