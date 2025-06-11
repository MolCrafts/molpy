
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
    
class IncrementalCounter:

    def __init__(self):
        self.value = 0

    def update(self):
        """
        Update the counter.
        """
        self.value += 1
        return self.value

class TagApplyer:
    """
    A class to apply tags to a list of items.
    """

    def __init__(self):
        self.dollar_mapping = {}

    def update_dollar_counter(self):
        """
        Update the dollar counter.
        """
        for value in self.dollar_mapping.values():
            value.update()

    def apply_tags(self, items: list[dict[str, Any]]):
        """
        Apply tags to the items.
        """
        for item in items:
            for key, value in item.items():
                if isinstance(value, str) and value.startswith("$"):
                    item[key] = self.resolve_dollar(value[1:])
    
    def resolve_dollar(self, value):
        """
        Resolve the dollar sign in the value.
        """
        if value not in self.dollar_mapping:
            self.dollar_mapping[value] = IncrementalCounter()

        return self.dollar_mapping[value].value

