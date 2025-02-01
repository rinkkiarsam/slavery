from typing import Generic, TypeVar, Iterator, Any
from collections import deque


# For code readability purposes
KT = TypeVar('KT')  # Key type (must support __lt__, __gt__, __eq__ operators)
VT = TypeVar('VT')  # Value type (any)


class AVLNode(Generic[KT, VT]):
    """Represents a node in an AVL tree"""

    def __init__(self, key: KT, value: VT = None) -> None:
        self.key: KT = key
        self.value: VT = value or key
        self.left: AVLNode[KT, VT] | None = None
        self.right: AVLNode[KT, VT] | None = None
        self.height: int = 1

    def update(self, key: KT, value: VT):
        self.key = key
        self.value = value

    def __str__(self) -> str:
        return f'AVLNode({self.key}: {self.value})'

    def __repr__(self):
        return str(self)


class AVLTree(Generic[KT, VT]):
    """
    Implements an AVL tree: a self-balancing binary search tree.
    see: http://en.wikipedia.org/wiki/AVL_tree

    Key type KT assumed to be compatible with <, ==, > operators
    """

    def __init__(self) -> None:
        """
        Instantiates AVLTree object.
        Tree is represented by its root node, initially None.
        """

        self.root: AVLNode[KT, VT] | None = None
        self.size: int = 0

    def insert(self, key: KT, value: VT = None) -> None:
        """
        Inserts a new node into the AVL tree.
        Note that duplicates are not supported. If the key being inserted is already present in the tree,
        it will simply overwrite current value with a new one.
        """

        self.root = self._insert(self.root, key, value)

    def _insert(self, node: AVLNode[KT, VT] | None, key: KT, value: VT) -> AVLNode[KT, VT]:
        """Recursive insertion helper with post-operation balancing."""

        if node is None:
            self.size += 1
            return AVLNode(key, value)

        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:  # Key already exists -- update value
            node.update(key, value)
            return node  # No balancing required for updates

        # Update height and rebalance tree after insertion
        self._update_height(node)
        return self._balance(node)

    def delete(self, key: KT) -> None:
        """
        Deletes a node with the specified key, if it exists.
        """

        self.root = self._delete(self.root, key)

    def _delete(self, node: AVLNode[KT, VT] | None, key: KT) -> AVLNode[KT, VT] | None:
        """Recursive deletion helper with post-operation balancing."""
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:  # Element found
            # check number of children
            if node.left is None and node.right is None:
                # No children
                return None
            elif node.left is None:
                # Right children only
                node = node.right
            elif node.right is None:
                # Left children only
                node = node.left
            else:
                # Two children
                aux = self._get_min(node.right)
                node.update(aux.key, aux.value)
                node.right = self._delete(node.right, aux.key)

        self.size -= 1
        if node is not None:
            self._update_height(node)
            self._balance(node)
        return node

    @staticmethod
    def _get_min(node: AVLNode[KT, VT]) -> AVLNode[KT, VT]:
        """Find minimum node in a (sub)tree and returns it."""
        while node.left is not None:
            node = node.left
        return node

    def get(self, key: KT, default: Any = None, raise_exception: bool = False) -> VT:
        """
        Search a given key in the tree. Returns value associated with the key.
        The .default parameter is responsible for the value returned in case given key is not found in the AVL tree.
        The .raise_exception parameter raises a KeyError is set to True, else doesn't.
        """

        node = self._get(self.root, key)
        if node is None:
            if raise_exception:
                raise KeyError(f"Key '{key}' not found")
            return default
        else:
            return node.value

    def _get(self, node: AVLNode[KT, VT] | None, key: KT) -> AVLNode[KT, VT] | None:
        """Recursive search helper."""

        if node is None:
            return None

        if key < node.key:
            return self._get(node.left, key)
        elif key > node.key:
            return self._get(node.right, key)
        else:
            return node

    def height(self) -> int:
        """Returns height of the tree -- length of the longest path from root to one of the leaves"""

        return self._height(self.root)

    @staticmethod
    def _height(node: AVLNode[KT, VT] | None) -> int:
        """Return height of the node."""

        return node.height if node is not None else 0

    def _update_height(self, node: AVLNode[KT, VT]) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: AVLNode[KT, VT]) -> int:
        """Balance factor (difference of subtree heights)."""

        return self._height(node.left) - self._height(node.right)

    def _rotate_left(self, z: AVLNode[KT, VT]) -> AVLNode[KT, VT]:
        """
        Rotates around z to rebalance subtree.

        Makes z the left child of y.
        The left child of y becomes the left child of x.
        """

        y = z.right
        z.right = y.left
        y.left = z

        self._update_height(z)
        self._update_height(y)

        return y

    def _rotate_right(self, z: AVLNode[KT, VT]) -> AVLNode[KT, VT]:
        """
        Rotates around z to rebalance subtree.

        Makes z the right child of y.
        The parent of z becomes the parent of y.
        """
        y = z.left
        z.left = y.right
        y.right = z

        self._update_height(z)
        self._update_height(y)

        return y

    def _balance(self, node: AVLNode[KT, VT]) -> AVLNode[KT, VT] | None:
        """
        Determines orientation of imbalanced nodes and calls indicated balancing methods.

        Calls _rotate_right or _rotate_left as determined by orientation of unbalanced nodes.
        """

        if node is None:
            return None

        self._update_height(node)
        _balance_factor = self._balance_factor(node)

        if _balance_factor > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
            else:
                return self._rotate_right(node)
        elif _balance_factor < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
            else:
                return self._rotate_left(node)

        return node

    def in_order(self) -> Iterator[AVLNode[KT, VT]]:
        """In-order traversal iterator (LNR)."""

        stack: list[AVLNode[KT, VT]] = []
        current: AVLNode[KT, VT] = self.root

        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            yield current
            current = current.right

    def pre_order(self) -> Iterator[AVLNode[KT, VT]]:
        """Pre-order traversal iterator (NLR)."""

        stack: list[AVLNode[KT, VT]] = [self.root] if self.root is not None else []

        while stack:
            node: AVLNode[KT, VT] = stack.pop()
            yield node
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    def post_order(self) -> Iterator[AVLNode[KT, VT]]:
        """Post-order traversal iterator (LRN)."""

        stack: list[AVLNode[KT, VT]] = []
        last_visited: AVLNode[KT, VT] | None = None
        current: AVLNode[KT, VT] | None = self.root

        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek: AVLNode[KT, VT] = stack[-1]
                if peek.right and last_visited.key != peek.right.key:
                    current = peek.right
                else:
                    yield peek
                    last_visited = stack.pop()

    def _level_order(self) -> list[list[AVLNode[KT, VT]]]:
        """Breadth-first traversal iterator."""

        queue: deque[AVLNode[KT, VT]] = deque([self.root]) if self.root is not None else deque()

        while queue:
            node = queue.popleft()
            yield node

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

    def __iter__(self) -> Iterator[AVLNode[KT, VT]]:
        """Default iterator (in-order traversal)."""
        return self.in_order()

    def merge(self, other: 'AVLTree[KT, VT]') -> 'AVLTree[KT, VT]':
        """
        Merges two AVL trees into a new balanced AVL tree.

        For duplicate keys, values from the 'other' tree take precedence.
        Original trees remain unmodified.
        """

        self_nodes = [x for x in self.in_order()]
        other_nodes = [x for x in other.in_order()]

        merged_nodes = self._merge_sorted_node_lists(self_nodes, other_nodes)

        new_tree = AVLTree[KT, VT]()
        new_tree.root = self._build(merged_nodes)
        return new_tree

    @staticmethod
    def _merge_sorted_node_lists(this: list[AVLNode[KT, VT]], other: list[AVLNode[KT, VT]]) -> list[AVLNode[KT, VT]]:
        """Merges two sorted node lists, prioritizing other's nodes on key conflicts."""

        merged: list[AVLNode[KT, VT]] = []
        i = j = 0

        while i < len(this) and j < len(other):
            if this[i].key < other[j].key:
                merged.append(this[i])
                i += 1
            elif this[i].key > other[j].key:
                merged.append(other[j])
                j += 1
            else:  # Key conflict: take from other and advance both
                merged.append(other[j])
                i += 1
                j += 1

        merged.extend(this[i:])
        merged.extend(other[j:])
        return merged

    def _build(self, nodes: list[AVLNode[KT, VT]]) -> AVLNode[KT, VT] | None:
        """Constructs a perfectly balanced AVL tree from sorted list of nodes."""

        return self._build_tree(nodes, 0, len(nodes) - 1)

    def _build_tree(self, nodes: list[AVLNode[KT, VT]], start: int, end: int) -> AVLNode[KT, VT] | None:
        if start > end:
            return None

        mid = (start + end) // 2
        original_node = nodes[mid]

        # Create new node to avoid reference to original trees
        new_node = AVLNode[KT, VT](original_node.key, original_node.value)
        new_node.left = self._build_tree(nodes, start, mid - 1)
        new_node.right = self._build_tree(nodes, mid + 1, end)

        # Calculate height based on children
        left_height = AVLTree._height(new_node.left)
        right_height = AVLTree._height(new_node.right)
        new_node.height = 1 + max(left_height, right_height)

        return new_node

    def split_at(self, key: KT) -> tuple['AVLTree[KT, VT]', 'AVLTree[KT, VT]']:
        """
        Splits the tree into two AVL trees based on the key.

        Returns:
            (AVLTree[KT, VT], AVLTree[KT, Kv]):
            - Left tree contains keys < key
            - Right tree contains keys >= key
        """

        nodes: list[AVLNode[KT, VT]] = [x for x in self.in_order()]
        keys: list[KT] = [node.key for node in nodes]
        split_index = keys.index(key)

        left_tree, right_tree = AVLTree[KT, VT](), AVLTree[KT, VT]()
        left_tree.root = self._build(nodes[:split_index])
        right_tree.root = self._build(nodes[split_index:])

        return left_tree, right_tree

    def _vis(self, node: AVLNode[KT, VT]) -> list[str]:
        """
        A recursive visualization helper shamelessly stolen from one of my labs.
        Can't even call it stealing -- how can I steal from myself?
        """
        if node is None:
            if self.root is None:
                return []

        strings: list[str] = []
        if node.right is not None:
            for right_string in self._vis(node.right):
                strings.append(5 * ' ' + right_string.replace('->', '/-', 1))

        strings.append(f'-> ({repr(node.key)})')

        if node.left is not None:
            for left_string in self._vis(node.left):
                strings.append(5 * ' ' + left_string.replace('->', '\\-', 1))

        return strings

    def __str__(self) -> str:
        """Return a human-readable string representation of the AVL tree."""
        return '\n'.join(self._vis(self.root))

    def __len__(self):
        """Return the number of key-value pairs stored in the AVL tree."""
        return self.size

    def __getitem__(self, key: KT) -> VT:
        """
        Dunder method for retrieving the value associated with the given key.
        Essentially a wrapper over .get() method.
        """

        return self.get(key)

    def __setitem__(self, key: KT, value: VT) -> None:
        """
        Insert a new key-value pair into the AVL tree or update the value if the key exists.
        Essentially a wrapper over .insert() method.
        """

        self.insert(key, value)

    def __delitem__(self, key: KT) -> None:
        """
        Remove the key-value pair associated with the given key from the AVL tree.
        Essentially a wrapper over .delete() method.
        """

        self.delete(key)

    def __contains__(self, key: KT):
        """Check if the given key exists in the AVL tree."""

        return self.get(key) is not None

    def __add__(self, other: 'AVLTree[KT, VT]') -> 'AVLTree[KT, VT]':
        """
        Merged two AVL trees into a new AVL tree, which is returned.
        Essentially a wrapper over .merge() method.
        """

        return self.merge(other)


if __name__ == '__main__':
    # Demonstration of the AVL tree functionality
    list1 = [1, 2, 3, 4]
    list2 = [3, 4, 5, 6]

    tree1 = AVLTree()
    for x in list1:
        tree1.insert(x)
    tree2 = AVLTree()
    for x in list2:
        tree2.insert(x)
    # tree1:              tree2:
    #         2                   4
    #        / \                 / \
    #       1   3               3   5
    #            \                   \
    #             4                   6
    print(tree1, end='\n' + 8 * '-' + '\n')
    print(tree2, end='\n' + 8 * '-' + '\n')

    # Merging trees
    tree3 = tree1 + tree2
    # tree3:
    #         3
    #       /   \
    #      1     5
    #       \   / \
    #        2  4  6
    print(tree3, end='\n' + 8 * '-' + '\n')

    # Splitting tree
    tree1, tree2 = tree3.split_at(4)
    # tree1:          tree2:
    #         2               5
    #        / \             / \
    #       1   3           4   6
    print(tree1, end='\n' + 8 * '-' + '\n')
    print(tree2, end='\n' + 8 * '-' + '\n')

    # Traversal
    print([x for x in tree1.in_order()])
    print([x for x in tree1.pre_order()])
    print([x for x in tree1.post_order()])
    # [AVLNode(1: 1), AVLNode(2: 2), AVLNode(3: 3)]
    # [AVLNode(2: 2), AVLNode(1: 1), AVLNode(3: 3)]
    # [AVLNode(1: 1), AVLNode(3: 3), AVLNode(2: 2)]
