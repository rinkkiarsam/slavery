from typing import Generic, TypeVar
from AVL import AVLTree


KT = TypeVar('KT')  # Key type
VT = TypeVar('VT')  # Value type


class AssocArray(Generic[KT, VT]):
    """
    A minimal AVL-based associative array implementation.

    [ОБЪЯСНЕНИЯ ЧТО, ПОЧЕМУ и ЗАЧЕМ:]
    As per the problem's specifications, the data structure to be implemented has to have the 'necessary and sufficient'
    operations. Those were taken to be the insert, delete, and get operations.
    It was made sure that the provided operations satisfy necessary properties stated at:
        https://en.wikipedia.org/wiki/Associative_array#Operations

    Choice of the tree-based implementation of associative array was primarily dictated by the fact that the author
    didn't see much use in making a hash table implementation, since they are already present in Python, and are called
    dicts (read as: that's plain boring guys idk).

    Tree implementation provides significantly better worst-case performance of O(log n) as opposed to hash table
    implementation's O(n) and is used in e.g. C++'s maps.
    """

    def __init__(self):
        """Initialize an empty AVL-based associative array."""

        self._storage = AVLTree()

    def __str__(self) -> str:
        """Return a human-readable string representation of the associative array."""

        items: list = []
        for node in self._storage:
            items.append(f"{repr(node.key)}: {repr(node.value)}")
        return "{" + ", ".join(items) + "}"

    def insert(self, key: KT, value: VT) -> None:
        """Insert a key-value pair into the associative array, or update the value if the key exists."""
        self._storage.insert(key, value)

    def delete(self, key: KT) -> None:
        """Remove the key-value pair associated with the given key from the associative array."""

        self._storage.delete(key)

    def get(self, key: KT) -> VT:
        """Retrieve the value associated with the given key from the associative array."""

        return self._storage.get(key)


if __name__ == '__main__':
    # Demo
    ass_array = AssocArray()
    ass_array.insert("name", "biba")
    ass_array.insert("surname", "shurshunchikov")
    ass_array.insert("age", "unknown")
    ass_array.insert("key smash", "afsdvwfsrbsbdng")

    # {'age': 'unknown', 'key smash': 'afsdvwfsrbsbdng', 'name': 'biba', 'surname': 'shurshunchikov'}
    print(ass_array)

    ass_array.insert("age", "12")
    ass_array.delete("key smash")

    # {'age': '12', 'name': 'biba', 'surname': 'shurshunchikov'}
    print(ass_array)

    # 12
    print(ass_array.get("age"))

    ass_array.insert("сало", "ваше")
    ass_array.insert("мусала", "ваши же")
