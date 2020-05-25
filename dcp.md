Question 1

Given a list of numbers, return whether any two sums to k.
For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.

Bonus: Can you do this in one pass?

Answer 1





```

def check_sums(array, k):
    potential_solutions = set()
    for num in array:
        if num in potential_solutions:
            return True
        potential_solutions.add(k - num)
    
    return False


assert not check_sums([], 17)
assert check_sums([10, 15, 3, 7], 17)
assert not check_sums([10, 15, 3, 4], 17)

```



Question 2

This problem was asked by Uber.

Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.

For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].

Follow-up: what if you can't use division?

Answer 2





```

def get_factors(array):
    cumulative_product = 1
    right_prod_array = list()
    for num in array:
        cumulative_product *= num
        right_prod_array.append(cumulative_product)

    cumulative_product = 1
    left_prod_array = list()
    for num in array[::-1]:
        cumulative_product *= num
        left_prod_array.append(cumulative_product)
    left_prod_array = left_prod_array[::-1]

    output_array = list()
    for i in range(len(array)):
        num = None
        if i == 0:
            num = left_prod_array[i + 1]
        elif i == len(array) - 1:
            num = right_prod_array[i - 1]
        else:
            num = right_prod_array[i - 1] * left_prod_array[i + 1]
        output_array.append(num)
    
    return output_array


assert get_factors([1, 2, 3, 4, 5]) == [120, 60, 40, 30, 24]
assert get_factors([3, 2, 1]) == [2, 3, 6]

```



Question 3

This problem was asked by Google.

Given the root to a binary tree, implement serialize(root), which serializes the tree into a string, and deserialize(s), which deserializes the string back into the tree.

Answer 3





```

import json

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    
    def __repr__(self):
        return str(self.data)

def serialize(root):
    if not root:
        return None

    serialized_tree_map = dict()
    serialized_left = serialize(root.left)
    serialized_right = serialize(root.right)

    serialized_tree_map['data'] = root.data
    if serialized_left:
        serialized_tree_map['left'] = serialized_left
    if serialized_right:
        serialized_tree_map['right'] = serialized_right

    return json.dumps(serialized_tree_map)


def deserialize(s):
    serialized_tree_map = json.loads(s)

    node = Node(serialized_tree_map['data'])
    if 'left' in serialized_tree_map:
        node.left = deserialize(serialized_tree_map['left'])
    if 'right' in serialized_tree_map:
        node.right = deserialize(serialized_tree_map['right'])

    return node


node_a = Node('a')
node_b = Node('b')
node_c = Node('c')
node_d = Node('d')
node_e = Node('e')
node_f = Node('f')
node_g = Node('g')
node_a.left = node_b
node_a.right = node_c
node_b.left = node_d
node_b.right = node_e
node_c.left = node_f
node_c.right = node_g

serialized_a = serialize(node_a)
print(serialized_a)

deserialized_a = deserialize(serialized_a)
assert str(deserialized_a) == "a"

```



Question 4

This problem was asked by Stripe.

Given an array of integers, find the first missing positive integer in linear time and constant space. In other words, find the lowest positive integer that does not exist in the array. The array can contain duplicates and negative numbers as well.

For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.

You can modify the input array in-place.

Answer 4





```

def get_positive_subset(array):
    i = 0
    j = len(array) - 1

    while i < j:
        if array[i] > 0 and array[j] <= 0:
            array[i], array[j] = array[j], array[i]
            i += 1
            j -= 1
        elif array[i] > 0:
            j -= 1
        else:
            i += 1

    # print("i: {}, j: {}".format(i, j))
    # print("partitioned_array:", array)
    pivot = i if array[i] > 0 else i + 1
    return array[pivot:]


def get_missing_number(array):
    if not array:
        return 1

    array = get_positive_subset(array)
    array_len = len(array)
    # print("array: {}".format(array))

    if not array:
        return 1

    if max(array) == len(array):
        return max(array) + 1

    for num in array:
        current_num = abs(num)
        if (current_num - 1) < array_len:
            array[current_num - 1] *= -1
    # print("mutated_array: {}".format(array))

    for i, num in enumerate(array):
        if num > 0:
            return i + 1


assert get_missing_number([3, 4, -1, 1]) == 2
assert get_missing_number([1, 2, 0]) == 3
assert get_missing_number([1, 2, 5]) == 3
assert get_missing_number([1]) == 2
assert get_missing_number([-1, -2]) == 1
assert get_missing_number([]) == 1

```



Question 5

This problem was asked by Jane Street.

cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair. For example, car(cons(3, 4)) returns 3, and cdr(cons(3, 4)) returns 4.

Given this implementation of cons:
```python
def cons(a, b):
    return lambda f : f(a, b)
```
Implement car and cdr.

Answer 5





```

def cons(a, b):
    return lambda f: f(a, b)

def car(f):
    z = lambda x, y: x
    return f(z)

def cdr(f):
    z = lambda x, y: y
    return f(z)


assert car(cons(3, 4)) == 3
assert cdr(cons(3, 4)) == 4

```



Question 6

This problem was asked by Google.

An XOR linked list is a more memory efficient doubly linked list. Instead of each node holding next and prev fields, it holds a field named both, which is an XOR of the next node and the previous node. Implement an XOR linked list; it has an add(element) which adds the element to the end, and a get(index) which returns the node at index.

If using a language that has no pointers (such as Python), you can assume you have access to get_pointer and dereference_pointer functions that converts between nodes and memory addresses.

Answer 6





```

class Node:
    def __init__(self, data):
        self.data = data
        self.both = id(data)

    def __repr__(self):
        return str(self.data)

a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")
e = Node("e")

# id_map simulates object pointer values
id_map = dict()
id_map[id("a")] = a
id_map[id("b")] = b
id_map[id("c")] = c
id_map[id("d")] = d
id_map[id("e")] = e


class LinkedList:

    def __init__(self, node):
        self.head = node
        self.tail = node
        self.head.both = 0
        self.tail.both = 0

    def add(self, element):
        self.tail.both ^= id(element.data)
        element.both = id(self.tail.data)
        self.tail = element

    def get(self, index):
        prev_node_address = 0
        result_node = self.head
        for i in range(index):
            next_node_address = prev_node_address ^ result_node.both
            prev_node_address = id(result_node.data)
            result_node = id_map[next_node_address]
        return result_node.data


llist = LinkedList(c)
llist.add(d)
llist.add(e)
llist.add(a)

assert llist.get(0) == "c"
assert llist.get(1) == "d"
assert llist.get(2) == "e"
assert llist.get(3) == "a"

```



Question 7

This problem was asked by Facebook.

Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example, '001' is not allowed.

Answer 7





```

def is_char(code):
    return 0 if code > 26 or code < 1 else 1

def get_message_count(code):
    code_str = str(code)
    if len(code_str) == 1:
        count = 1
    elif len(code_str) == 2:
        count = 1 + is_char(code)
    else:
        count = get_message_count(int(code_str[1:]))
        if is_char(int(code_str[:2])):
            count += get_message_count(int(code_str[2:]))

    return count


assert get_message_count(81) == 1
assert get_message_count(11) == 2
assert get_message_count(111) == 3
assert get_message_count(1111) == 5
assert get_message_count(1311) == 4

```



Question 8

This problem was asked by Google.

A unival tree (which stands for "universal value") is a tree where all nodes under it have the same value.

Given the root to a binary tree, count the number of unival subtrees.

For example, the following tree has 5 unival subtrees:

```
   0
  / \
 1   0
    / \
   1   0
  / \
 1   1
```

Answer 8





```

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    
    def __repr__(self):
        return str(self.data)

def count_unival_trees(root):
    if not root:
        return 0
    elif not root.left and not root.right:
        return 1
    elif not root.left and root.data == root.right.data:
        return 1 + count_unival_trees(root.right)
    elif not root.right and root.data == root.left.data:
        return 1 + count_unival_trees(root.left)
    
    child_counts = count_unival_trees(root.left) + count_unival_trees(root.right)
    current_node_count = 0
    if root.data == root.left.data and root.data == root.left.data:
        current_node_count = 1

    return current_node_count + child_counts


node_a = Node('0')
node_b = Node('1')
node_c = Node('0')
node_d = Node('1')
node_e = Node('0')
node_f = Node('1')
node_g = Node('1')
node_a.left = node_b
node_a.right = node_c
node_c.left = node_d
node_c.right = node_e
node_d.left = node_f
node_d.right = node_g

assert count_unival_trees(None) == 0
assert count_unival_trees(node_a) == 5
assert count_unival_trees(node_c) == 4
assert count_unival_trees(node_g) == 1
assert count_unival_trees(node_d) == 3

```



Question 9

This problem was asked by Airbnb.

Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

For example, [2, 4, 6, 8] should return 12, since we pick 4 and 8. [5, 1, 1, 5] should return 10, since we pick 5 and 5.

Answer 9





```

import sys

def get_largest_non_adj_sum(array):
    previous, largest = 0, 0
    for amount in array:
        print("amount: {}; previous: {}; largest: {}".format(amount, previous, largest))
        previous, largest = largest, max(largest, previous + amount)
        print("new_previous: {}; new_largest: {}".format(previous, largest))
    return largest

print(get_largest_non_adj_sum([2, 4, 6, 8]))
print(get_largest_non_adj_sum([5, 1, 1, 5]))

```



Question 10

This problem was asked by Apple.

Implement a job scheduler which takes in a function f and an integer n, and calls f after n milliseconds.

Answer 10





```

from time import sleep

def sample_function():
    print("hello")

def schedule_function(f, delay_in_ms):
    delay_in_s = delay_in_ms / 1000
    sleep(delay_in_s)
    f()

schedule_function(sample_function, 10000)

```



Question 11

This problem was asked by Twitter.

Implement an autocomplete system. That is, given a query string s and a set of all possible query strings, return all strings in the set that have s as a prefix.

For example, given the query string de and the set of strings [dog, deer, deal], return [deer, deal].

Hint: Try preprocessing the dictionary into a more efficient data structure to speed up queries.

Answer 11





```

def add_to_trie(s, trie):
    if not s:
        return trie

    character = s[0]
    if character not in trie:
        trie[character] = dict()
    
    trie[character] = add_to_trie(s[1:], trie[character])

    return trie

def get_dictionary_trie(dictionary):
    trie = dict()
    for word in dictionary:
        trie = add_to_trie(word, trie)

    return trie

def get_possible_completions(trie):
    possible_completions = list()
    for character in trie:
        if trie[character]:
            child_completions = get_possible_completions(trie[character])
            for child_completion in child_completions:
                possible_completions.append(character + child_completion)
        else:
            possible_completions.append(character)
        
    return possible_completions

def get_autocomplete_suggestions(s, dictionary):
    trie = get_dictionary_trie(dictionary)

    current_trie = trie
    for character in s:
        if character not in current_trie:
            return []
        current_trie = current_trie[character]

    completions = get_possible_completions(current_trie)
    completions = [s + x for x in completions]

    return completions 

assert get_autocomplete_suggestions("de", ["dog", "deer", "deal"]) == ["deer", "deal"]
assert get_autocomplete_suggestions("ca", ["cat", "car", "cer"]) == ["cat", "car"]
assert get_autocomplete_suggestions("ae", ["cat", "car", "cer"]) == []
assert get_autocomplete_suggestions("ae", []) == []

```



Question 12

This problem was asked by Amazon.

There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function that returns the number of unique ways you can climb the staircase. The order of the steps matters.

For example, if N is 4, then there are 5 unique ways:

```
1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2
```

What if, instead of being able to climb 1 or 2 steps at a time, you could climb any number from a set of positive integers X? For example, if X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time.

Answer 12





```

def get_step_combos(num_steps, step_sizes):
    combos = list()
    
    if num_steps < min(step_sizes):
        return combos
    
    for step_size in step_sizes:
        if num_steps == step_size:
            combos.append([step_size])
        elif num_steps > step_size:
            child_combos = get_step_combos(num_steps - step_size, step_sizes)
            for child_combo in child_combos:
                combos.append([step_size] + child_combo)
    return combos


assert get_step_combos(4, [1, 2]) == \
    [[1, 1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2]]
assert get_step_combos(4, [1, 2, 3]) == \
    [[1, 1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 3], [2, 1, 1], [2, 2], [3, 1]]

```



Question 13

This problem was asked by Amazon.

Given an integer k and a string s, find the length of the longest substring that contains at most k distinct characters.

For example, given s = "abcba" and k = 2, the longest substring with k distinct characters is "bcb".

Answer 13





```

def get_longest_sub_with_k_dist(s, k):
    if not s:
        return ""
    elif len(s) <= k:
        return s
    elif k == 1:
        return s[0]

    distinct_char_count = 0
    seen_chars = set()
    candidate = None
    remaining_string = None

    # to keep track of where the second character occurred
    first_char = s[0]
    second_char_index = 0
    while s[second_char_index] == first_char:
        second_char_index += 1

    candidate = s
    for index, char in enumerate(s):
        if char not in seen_chars:
            seen_chars.add(char)
            distinct_char_count += 1

        if distinct_char_count > k:
            candidate = s[:index]
            remaining_string = s[second_char_index:]
            break
            
    longest_remaining = get_longest_sub_with_k_dist(remaining_string, k)
    
    longest_substring = None
    if len(candidate) < len(longest_remaining):
        longest_substring = longest_remaining
    else:
        longest_substring = candidate
    return longest_substring


assert get_longest_sub_with_k_dist("abcba", 2) == "bcb"
assert get_longest_sub_with_k_dist("abccbba", 2) == "bccbb"
assert get_longest_sub_with_k_dist("abcbbbabbcbbadd", 2) == "bbbabb"
assert get_longest_sub_with_k_dist("abcbbbaaaaaaaaaabbcbbadd", 1) == "a"
assert get_longest_sub_with_k_dist("abccbba", 3) == "abccbba"

```



Question 14

This problem was asked by Google.

The area of a circle is defined as r^2. Estimate \pi to 3 decimal places using a Monte Carlo method.

Hint: The basic equation of a circle is x^2 + y^2 = r^2.

Answer 14





```

from random import random
radius = 2


def estimate_pi(num_random_tests):
    pi_counter = 0
    rsquared = radius ** 2
    for _ in range(num_random_tests):
        x_rand = random() * radius
        y_rand = random() * radius
        if (x_rand ** 2) + (y_rand ** 2) < rsquared:
            pi_counter += 1

    return 4 * pi_counter / num_random_tests


assert round(estimate_pi(100000000), 3) == 3.141

```



Question 15

This problem was asked by Facebook.

Given a stream of elements too large to store in memory, pick a random element from the stream with uniform probability.

Answer 15





```

from random import random

count_so_far = 0
result = None


def pick_random_element(x):
    global count_so_far, result
    count_so_far += 1

    print(count_so_far)

    if count_so_far == 1:
        result = x
    else:
        random_value = int(count_so_far * random())
        if random_value == count_so_far - 1:
            result = x

    return result


sample_stream = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for index, element in enumerate(sample_stream):
    random_element = pick_random_element(element)
    print("Random element of the first {} is {}".format(index + 1, random_element))

```



Question 16

This problem was asked by Twitter.

You run an e-commerce website and want to record the last N order ids in a log. Implement a data structure to accomplish this, with the following API:

record(order_id): adds the order_id to the log
get_last(i): gets the ith last element from the log. i is guaranteed to be smaller than or equal to N.
You should be as efficient with time and space as possible.

Answer 16





```

class OrderLog:
    def __init__(self, size):
        self.log = list()
        self.size = size

    def __repr__(self):
        return str(self.log)

    def record(self, order_id):
        self.log.append(order_id)
        if len(self.log) > self.size:
            self.log = self.log[1:]

    def get_last(self, i):
        return self.log[-i]


log = OrderLog(5)
log.record(1)
log.record(2)
assert log.log == [1, 2]
log.record(3)
log.record(4)
log.record(5)
assert log.log == [1, 2, 3, 4, 5]
log.record(6)
log.record(7)
log.record(8)
assert log.log == [4, 5, 6, 7, 8]
assert log.get_last(4) == 5
assert log.get_last(1) == 8

```



Question 17

This problem was asked by Google.

Suppose we represent our file system by a string in the following manner:

The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

```
dir
    subdir1
    subdir2
        file.ext
```

The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.

The string "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" represents:

```
dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
```

The directory dir contains two sub-directories subdir1 and subdir2. subdir1 contains a file file1.ext and an empty second-level sub-directory subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing a file file2.ext.

We are interested in finding the longest (number of characters) absolute path to a file within our file system. For example, in the second example above, the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its length is 32 (not including the double quotes).

Given a string representing the file system in the above format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.

Answer 17





```

class Node:
    def __init__(self, name, pathtype):
        self.name = name
        self.type = pathtype
        self.children = list()
        self.length = len(name)
        self.max_child_length = 0

    def __repr__(self):
        return "(name={}, type={}, len={}, max_child_len={})".format(
            self.name, self.type, self.length, self.max_child_length)


def create_graph(node_list):
    if not node_list:
        return None

    parent_node = node_list[0][1]
    level = node_list[0][0]

    for index, (node_level, _) in enumerate(node_list[1:]):
        if node_level <= level:
            break
        if node_level == level + 1:
            child_node = create_graph(node_list[index + 1:])
            parent_node.children.append(child_node)
            if child_node.children or child_node.type == 'file':
                if child_node.max_child_length + child_node.length > parent_node.max_child_length:
                    parent_node.max_child_length = child_node.max_child_length + child_node.length

    # print("current_parent: {}".format(parent_node))
    # print("it's children: {}".format(parent_node.children))

    return parent_node


def get_path_type(name):
    return 'file' if '.' in name else 'directory'


def get_longest_path(s):
    if not s:
        return 0

    individual_lines = s.split('\n')
    split_lines = [x.split('\t') for x in individual_lines]
    annotated_lines = [
        (len(x) - 1, Node(name=x[-1], pathtype=get_path_type(x[-1])))
        for x in split_lines]

    graph = create_graph(annotated_lines)

    return graph.max_child_length + graph.length if graph.max_child_length > 0 else 0


assert get_longest_path("dir\n\tsubdir1\n\tsubdir2") == 0
assert get_longest_path("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext") == 18
assert get_longest_path(
    "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\t" +
    "subsubdir2\n\t\t\tfile2.ext") == 29

```



Question 18

This problem was asked by Google.

Given an array of integers and a number k, where 1 <= k <= length of the array, compute the maximum values of each subarray of length k.

For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10, 7, 8, 8], since:

```
10 = max(10, 5, 2)
7 = max(5, 2, 7)
8 = max(2, 7, 8)
8 = max(7, 8, 7)
```

Do this in O(n) time and O(k) space. You can modify the input array in-place and you do not need to store the results. You can simply print them out as you compute them.

Answer 18





```

from collections import deque


def get_sliding_max(a, k):

    window_max_elements = list()

    if not a:
        return None
    if len(a) <= k:
        return max(a)

    dq = deque()

    for i in range(k):
        while dq and a[dq[-1]] < a[i]:
            dq.pop()
        dq.append(i)
    window_max_elements.append(a[dq[0]])

    for i in range(k, len(a)):
        while dq and dq[0] <= i - k:
            dq.popleft()

        while dq and a[dq[-1]] < a[i]:
            dq.pop()
        dq.append(i)

        window_max_elements.append(a[dq[0]])

    return window_max_elements


assert get_sliding_max([10, 5, 2, 7, 8, 7], 3) == [10, 7, 8, 8]
assert get_sliding_max([5, 2, 1], 2) == [5, 2]

```



Question 19

This problem was asked by Facebook.

A builder is looking to build a row of N houses that can be of K different colors. He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.

Given an N by K matrix where the nth row and kth column represents the cost to build the nth house with kth color, return the minimum cost which achieves this goal.

Answer 19





```

import sys


def get_minimum_painting_cost(cost_matrix, num_houses, num_colors):
    if not cost_matrix:
        return 0

    prev_house_min = 0
    prev_house_min_index = -1
    prev_house_second_min = 0

    for i in range(num_houses):
        curr_house_min = sys.maxsize
        curr_house_second_min = sys.maxsize
        curr_house_min_index = 0

        for j in range(num_colors):
            if prev_house_min_index == j:
                cost_matrix[i][j] += prev_house_second_min
            else:
                cost_matrix[i][j] += prev_house_min

            if curr_house_min > cost_matrix[i][j]:
                curr_house_second_min = curr_house_min
                curr_house_min = cost_matrix[i][j]
                curr_house_min_index = j
            elif curr_house_second_min > cost_matrix[i][j]:
                curr_house_second_min = cost_matrix[i][j]

        prev_house_min = curr_house_min
        prev_house_second_min = curr_house_second_min
        prev_house_min_index = curr_house_min_index

    return min(cost_matrix[num_houses - 1])


cost_matrix = \
    [[7, 3, 8, 6, 1, 2],
     [5, 6, 7, 2, 4, 3],
     [10, 1, 4, 9, 7, 6]]
assert get_minimum_painting_cost(cost_matrix,
                                 len(cost_matrix), len(cost_matrix[0])) == 4

cost_matrix = \
    [[7, 3, 8, 6, 1, 2],
     [5, 6, 7, 2, 4, 3],
     [10, 1, 4, 9, 7, 6],
     [10, 1, 4, 9, 7, 6]]
assert get_minimum_painting_cost(cost_matrix,
                                 len(cost_matrix), len(cost_matrix[0])) == 8

```



Question 20

This problem was asked by Google.

Given two singly linked lists that intersect at some point, find the intersecting node. The lists are non-cyclical.

For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return the node with value 8.

In this example, assume nodes with the same value are the exact same node objects.

Do this in O(M + N) time (where M and N are the lengths of the lists) and constant space.

Answer 20





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def get_intersection_node(list_a, list_b):

    def get_list_length(linked_list):
        length = 0
        node = linked_list
        while node:
            length += 1
            node = node.next

        return length

    len_a, len_b = get_list_length(list_a), get_list_length(list_b)
    min_len = min(len_a, len_b)

    for _ in range(len_a - min_len):
        list_a = list_a.next
    for _ in range(len_b - min_len):
        list_b = list_b.next

    node_a = list_a
    node_b = list_b
    for _ in range(min_len):
        if node_a.val == node_b.val:
            return node_a
        node_a = node_a.next
        node_b = node_b.next

    return None


assert not get_intersection_node(
    get_nodes([]), get_nodes([]))
assert get_intersection_node(
    get_nodes([0, 3, 7, 8, 10]), get_nodes([99, 1, 8, 10])).val == 8
assert get_intersection_node(
    get_nodes([7, 8, 10]), get_nodes([99, 1, 8, 10])).val == 8

```



Question 21

This problem was asked by Snapchat.

Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of rooms required.

For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.

Answer 21





```

def get_num_classrooms(timing_tuples):
    if not timing_tuples:
        return 0

    start_times = dict()
    end_times = dict()
    for start, end in timing_tuples:
        if start not in start_times:
            start_times[start] = 0
        start_times[start] += 1

        if end not in end_times:
            end_times[end] = 0
        end_times[end] += 1

    global_start, global_end = min(start_times), max(end_times)

    max_class_count = 0
    current_class_count = 0
    for i in range(global_start, global_end):
        if i in start_times:
            current_class_count += start_times[i]
            if current_class_count > max_class_count:
                max_class_count = current_class_count
        if i in end_times:
            current_class_count -= end_times[i]

    return max_class_count


assert get_num_classrooms([]) == 0
assert get_num_classrooms([(30, 75), (0, 50), (60, 150)]) == 2
assert get_num_classrooms([(30, 75), (0, 50), (10, 60), (60, 150)]) == 3
assert get_num_classrooms([(60, 150)]) == 1
assert get_num_classrooms([(60, 150), (150, 170)]) == 2
assert get_num_classrooms([(60, 150), (60, 150), (150, 170)]) == 3

```



Question 22

This problem was asked by Microsoft.

Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return null.

For example, given the set of words 'quick', 'brown', 'the', 'fox', and the string "thequickbrownfox", you should return ['the', 'quick', 'brown', 'fox'].

Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the string "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or ['bedbath', 'and', 'beyond'].

Answer 22





```

def get_sentence_split(s, words):
    if not s or not words:
        return []

    word_set = set(words)
    sentence_words = list()
    for i in range(len(s)):
        if s[0:i + 1] in word_set:
            sentence_words.append(s[0:i + 1])
            word_set.remove(s[0:i + 1])
            sentence_words += get_sentence_split(s[i + 1:], word_set)
            break

    return sentence_words


assert get_sentence_split("thequickbrownfox", ['quick', 'brown', 'the', 'fox']) == [
    'the', 'quick', 'brown', 'fox']
assert get_sentence_split("bedbathandbeyond", [
                          'bed', 'bath', 'bedbath', 'and', 'beyond']) == ['bed', 'bath', 'and', 'beyond']

```



Question 23

This problem was asked by Google.

You are given an M by N matrix consisting of booleans that represents a board. Each True boolean represents a wall. Each False boolean represents a tile you can walk on.

Given this matrix, a start coordinate, and an end coordinate, return the minimum number of steps required to reach the end coordinate from the start. If there is no possible path, then return null. You can move up, left, down, and right. You cannot move through walls. You cannot wrap around the edges of the board.

For example, given the following board:

```
[[f, f, f, f],
 [t, t, f, t],
 [f, f, f, f],
 [f, f, f, f]]
```

and start = (3, 0) (bottom left) and end = (0, 0) (top left), the minimum number of steps required to reach the end is 7, since we would need to go through (1, 2) because there is a wall everywhere else on the second row.

Answer 23





```

from copy import deepcopy


def add_to_cache(coordinate, cache):
    new_cache = deepcopy(cache)
    new_cache.add("{}-{}".format(coordinate[0], coordinate[1]))
    return new_cache


def is_visited(coordinate, cache):
    return "{}-{}".format(coordinate[0], coordinate[1]) in cache


def find_path(matrix, rows, cols, start, end, cache):
    if start == end:
        return 0

    cache = add_to_cache(start, cache)

    def explore_neighbour(coordinate):
        if not is_visited(coordinate, cache) and \
                matrix[coordinate[0]][coordinate[1]] != "t":
            path_length = find_path(matrix, rows, cols, coordinate, end, cache)
            if path_length != None:
                path_lengths.append(path_length)

    path_lengths = list()
    if start[0] != 0:
        coordinate = (start[0] - 1, start[1])
        explore_neighbour(coordinate)
    if start[0] != rows - 1:
        coordinate = (start[0] + 1, start[1])
        explore_neighbour(coordinate)
    if start[1] != 0:
        coordinate = (start[0], start[1] - 1)
        explore_neighbour(coordinate)
    if start[1] != cols - 1:
        coordinate = (start[0], start[1] + 1)
        explore_neighbour(coordinate)

    return min(path_lengths) + 1 if path_lengths else None


matrix = [["f", "f", "f", "f"],
          ["t", "t", "f", "t"],
          ["f", "f", "f", "f"],
          ["f", "f", "f", "f"]]
assert find_path(matrix, len(matrix), len(
    matrix[0]), (0, 0), (0, 0), set()) == 0
assert find_path(matrix, len(matrix), len(
    matrix[0]), (1, 0), (0, 0), set()) == 1
assert find_path(matrix, len(matrix), len(
    matrix[0]), (3, 0), (0, 0), set()) == 7
assert find_path(matrix, len(matrix), len(
    matrix[0]), (3, 0), (0, 3), set()) == 6

matrix = [["f", "f", "f", "f"],
          ["t", "t", "t", "f"],
          ["f", "f", "f", "f"],
          ["f", "f", "f", "f"]]
assert find_path(matrix, len(matrix), len(
    matrix[0]), (0, 0), (0, 0), set()) == 0
assert find_path(matrix, len(matrix), len(
    matrix[0]), (1, 0), (0, 0), set()) == 1
assert find_path(matrix, len(matrix), len(
    matrix[0]), (3, 0), (0, 0), set()) == 9
assert find_path(matrix, len(matrix), len(
    matrix[0]), (3, 0), (0, 3), set()) == 6
assert find_path(matrix, len(matrix), len(
    matrix[0]), (2, 0), (3, 3), set()) == 4

```



Question 24

This problem was asked by Google.

Implement locking in a binary tree. A binary tree node can be locked or unlocked only if all of its descendants or ancestors are not locked.

Design a binary tree node class with the following methods:

is_locked, which returns whether the node is locked
lock, which attempts to lock the node. If it cannot be locked, then it should return false. Otherwise, it should lock it and return true.
unlock, which unlocks the node. If it cannot be unlocked, then it should return false. Otherwise, it should unlock it and return true.
You may augment the node to add parent pointers or any other property you would like. You may assume the class is used in a single-threaded program, so there is no need for actual locks or mutexes. Each method should run in O(h), where h is the height of the tree.

Answer 24





```

def is_parent_locked(node):
    if not node.parent:
        return False
    elif node.parent.locked:
        return True
    return is_parent_locked(node.parent)


def update_parent(node, enable_locks):
    increment = 1 if enable_locks else -1
    if node.parent:
        node.parent.locked_descendants += increment
        update_parent(node.parent, enable_locks)


class Node:
    def __init__(self, val, parent):
        self.val = val
        self.parent = parent
        self.left = None
        self.right = None
        self.locked = False
        self.locked_descendants = 0

    def __str__(self):
        return "val={}; locked={}; locked_descendants={}".format(
            self.val, self.locked, self.locked_descendants)

    def lock(self):
        if is_parent_locked(self) or self.locked_descendants:
            return False
        else:
            self.locked = True
            update_parent(node=self, enable_locks=True)
            return True

    def unlock(self):
        if is_parent_locked(self) or self.locked_descendants:
            return False
        else:
            self.locked = False
            update_parent(node=self, enable_locks=False)
            return True

    def is_locked(self):
        return self.locked


a = Node("a", None)
b = Node("b", a)
c = Node("c", a)
d = Node("d", b)
e = Node("e", b)
f = Node("f", c)
g = Node("g", c)

assert b.lock()
assert b.is_locked()
assert c.lock()
assert b.unlock()
assert not b.is_locked()
assert d.lock()

assert not g.lock()
assert c.unlock()
assert g.lock()

assert f.lock()
assert e.lock()
assert a.locked_descendants == 4
assert b.locked_descendants == 2
assert c.locked_descendants == 2

```



Question 25

This problem was asked by Facebook.

Implement regular expression matching with the following special characters:

* . (period) which matches any single character
* \* (asterisk) which matches zero or more of the preceding element
That is, implement a function that takes in a string and a valid regular expression and returns whether or not the string matches the regular expression.

For example, given the regular expression "ra." and the string "ray", your function should return true. The same regular expression on the string "raymond" should return false.

Given the regular expression `".*at"` and the string "chat", your function should return true. The same regular expression on the string "chats" should return false.

Answer 25





```

def is_match(regex, string):

    # if no pattern and no text, return True
    if not regex:
        return not string

    # first character will not be a Kleene star
    first_match = bool(string) and regex[0] in {string[0], '.'}

    if len(regex) >= 2 and regex[1] == '*':
        # regex[0] consumes no characters or
        # regex[0] consumes one character
        return is_match(regex[2:], string) or \
            (first_match and is_match(regex, string[1:]))
    else:
        # regex[0] consumes one character
        return first_match and is_match(regex[1:], string[1:])


assert is_match("ra.", "ray")
assert not is_match("ra.", "raymond")
assert is_match(".*at", "chat")
assert not is_match(".*at", "chats")

```



Question 26

This problem was asked by Google.

Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller than the length of the list.

The list is very long, so making more than one pass is prohibitively expensive.

Do this in constant space and in one pass.

Answer 26





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def remove_kth_last(head, k):
    if not head or not k:
        return head

    dummy = Node(None)
    dummy.next = head
    runner = head

    for _ in range(k):
        runner = runner.next

    current_node = dummy
    while runner:
        runner = runner.next
        current_node = current_node.next

    current_node.next = current_node.next.next

    return dummy.next


assert get_list(remove_kth_last(
    get_nodes([]), 1)) == []
assert get_list(remove_kth_last(
    get_nodes([0, 3, 7, 8, 10]), 2)) == [0, 3, 7, 10]
assert get_list(remove_kth_last(
    get_nodes([7, 8, 10]), 3)) == [8, 10]
assert get_list(remove_kth_last(
    get_nodes([7, 8, 10]), 1)) == [7, 8]

```



Question 27

This problem was asked by Facebook.

Given a string of round, curly, and square open and closing brackets, return whether the brackets are balanced (well-formed).

For example, given the string "([])[]({})", you should return true.

Given the string "([)]" or "((()", you should return false.

Answer 27





```

brace_map = {
    ")": "(",
    "}": "{",
    "]": "["
}


def is_balanced(s):
    stack = list()
    for char in s:
        if stack and char in brace_map and stack[-1] == brace_map[char]:
            stack.pop()
        else:
            stack.append(char)
    return not stack


assert is_balanced("")
assert is_balanced("{}")
assert is_balanced("([])")
assert is_balanced("([])[]({})")
assert not is_balanced("(")
assert not is_balanced("]")
assert not is_balanced("((()")
assert not is_balanced("([)]")

```



Question 28

This problem was asked by Palantir.

Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings which represents each line, fully justified.

More specifically, you should have as many words as possible in each line. There should be at least one space between each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as equally as possible, with the extra spaces, if any, distributed starting from the left.

If you can only fit one word on a line, then you should pad the right-hand side with spaces.

Each word is guaranteed not to be longer than k.

For example, given the list of words ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] and k = 16, you should return the following:

["the  quick brown", # 1 extra space on the left
"fox  jumps  over", # 2 extra spaces distributed evenly
"the   lazy   dog"] # 4 extra spaces distributed evenly

Answer 28





```

def justify_text(words, max_line_length):
    lines = list()

    cumulative_length = -1
    current_words = list()
    line_lengths = list()
    for word in words:
        if cumulative_length + (len(word) + 1) > max_line_length:
            lines.append(current_words)
            line_lengths.append(cumulative_length)
            cumulative_length = -1
            current_words = list()
        cumulative_length += (len(word) + 1)
        current_words.append(word)
        # print(current_words)
        # print(cumulative_length)
    if current_words:
        lines.append(current_words)
        line_lengths.append(cumulative_length)

    # print(lines)
    # print(line_lengths)

    justified_lines = list()
    for words, length in zip(lines, line_lengths):
        spaces_to_add = max_line_length - length
        guaranteed_spaces = 1 + (spaces_to_add // (len(words) - 1))
        bonus_space_recipients = spaces_to_add % (len(words) - 1)
        # print("spaces_to_add: {}".format(spaces_to_add))
        line = ""
        for (index, word) in enumerate(words[:-1]):
            line += word
            line += guaranteed_spaces * " "
            if index < bonus_space_recipients:
                line += " "
        line += words[-1]
        # print(line)
        justified_lines.append(line)

    # print(justified_lines)
    return justified_lines


assert justify_text(["the", "quick", "brown", "fox", "jumps",
                     "over", "the", "lazy", "dog"], 16) == \
    ['the  quick brown', 'fox  jumps  over', 'the   lazy   dog']
assert justify_text(["the", "quick", "brown", "fox", "jumps", "over"], 16) == \
    ['the  quick brown', 'fox  jumps  over']
assert justify_text(["the", "quick"], 16) == ['the        quick']

```



Question 29


This problem was asked by Amazon.

Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string "AAAABBBCCDAA" would be encoded as "4A3B2C1D2A".

Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.

Answer 29





```

def encode_string(s):
    encoded_chars = list()

    count = 0
    prev_char = None
    for char in s:
        if char == prev_char or not prev_char:
            count += 1
        else:
            encoded_chars.append(str(count))
            encoded_chars.append(prev_char)
            count = 1
        prev_char = char

    if count:
        encoded_chars.append(str(count))
        encoded_chars.append(prev_char)

    return "".join(encoded_chars)


def decode_string(s):

    decoded_chars = list()
    index = 0

    while index < len(s):
        decoded_chars.append(int(s[index]) * s[index + 1])
        index += 2

    return "".join(decoded_chars)


assert encode_string("") == ""
assert encode_string("AAA") == "3A"
assert encode_string("AAAABBBCCDAA") == "4A3B2C1D2A"

assert decode_string("") == ""
assert decode_string("3A") == "AAA"
assert decode_string("4A3B2C1D2A") == "AAAABBBCCDAA"

```



Question 30

You are given an array of non-negative integers that represents a two-dimensional elevation map where each element is unit-width wall and the integer is the height. Suppose it will rain and all spots between two walls get filled up.

Compute how many units of water remain trapped on the map in O(N) time and O(1) space.

For example, given the input [2, 1, 2], we can hold 1 unit of water in the middle.

Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2 in the second, and 3 in the fourth index (we cannot hold 5 since it would run off to the left), so we can trap 8 units of water.

Answer 30





```

def calculate_trapped_water(walls):
    if len(walls) < 3:
        return 0

    total_water_volume = 0

    left = 0
    right = len(walls) - 1
    left_max = 0
    right_max = 0

    while left <= right:
        if walls[left] < walls[right]:
            if walls[left] > left_max:
                left_max = walls[left]
            else:
                total_water_volume += left_max - walls[left]
            left += 1
        else:
            if walls[right] > right_max:
                right_max = walls[right]
            else:
                total_water_volume += right_max - walls[right]
            right -= 1

    return total_water_volume


assert calculate_trapped_water([1]) == 0
assert calculate_trapped_water([2, 1]) == 0
assert calculate_trapped_water([2, 1, 2]) == 1
assert calculate_trapped_water([4, 1, 2]) == 1
assert calculate_trapped_water([4, 1, 2, 3]) == 3
assert calculate_trapped_water([3, 0, 1, 3, 0, 5]) == 8
assert calculate_trapped_water([10, 9, 1, 1, 6]) == 10

```



Question 31

This problem was asked by Google.

The edit distance between two strings refers to the minimum number of character insertions, deletions, and substitutions required to change one string to the other. For example, the edit distance between "kitten" and "sitting" is three: substitute the "k" for "s", substitute the "e" for "i", and append a "g".

Given two strings, compute the edit distance between them.

Answer 31





```

def get_edit_distance(s1, s2):
    if s1 == s2:
        return 0
    elif not s1:
        return len(s2)
    elif not s2:
        return len(s1)

    if s1[0] == s2[0]:
        return get_edit_distance(s1[1:], s2[1:])

    return 1 + min(
        get_edit_distance(s1[1:], s2),      # deletion from s1
        get_edit_distance(s1, s2[1:]),      # addition to s1
        get_edit_distance(s1[1:], s2[1:]))  # modification to s1


assert get_edit_distance("", "") == 0
assert get_edit_distance("a", "b") == 1
assert get_edit_distance("abc", "") == 3
assert get_edit_distance("abc", "abc") == 0
assert get_edit_distance("kitten", "sitting") == 3

```



Question 32

This problem was asked by Jane Street.

Suppose you are given a table of currency exchange rates, represented as a 2D array. Determine whether there is a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount A of any currency, so that you can end up with some amount greater than A of that currency.

There are no transaction costs and you can trade fractional quantities.

Answer 32





```

from math import log


def arbitrage(table):
    transformed_graph = [[-log(edge) for edge in row] for row in table]

    # Pick any source vertex -- we can run Bellman-Ford from any vertex and
    # get the right result
    source = 0
    n = len(transformed_graph)
    min_dist = [float('inf')] * n

    min_dist[source] = 0

    # Relax edges |V - 1| times
    for _ in range(n - 1):
        for v in range(n):
            for w in range(n):
                if min_dist[w] > min_dist[v] + transformed_graph[v][w]:
                    min_dist[w] = min_dist[v] + transformed_graph[v][w]

    # If we can still relax edges, then we have a negative cycle
    for v in range(n):
        for w in range(n):
            if min_dist[w] > min_dist[v] + transformed_graph[v][w]:
                return True

    return False


assert arbitrage([[1, 2], [2, 1]])
assert not arbitrage([[1, 1], [1, 1]])

```



Question 33

This problem was asked by Microsoft.

Compute the running median of a sequence of numbers. That is, given a stream of numbers, print out the median of the list so far on each new element.

Recall that the median of an even-numbered list is the average of the two middle numbers.

For example, given the sequence [2, 1, 5, 7, 2, 0, 5], your algorithm should print out:

```
2
1.5
2
3.5
2
2
2
```

Answer 33





```

import heapq as hq


def get_running_medians(arr):
    if not arr:
        return None

    min_heap = list()
    max_heap = list()
    medians = list()

    for x in arr:
        hq.heappush(min_heap, x)
        if len(min_heap) > len(max_heap) + 1:
            smallest_large_element = hq.heappop(min_heap)
            hq.heappush(max_heap, -smallest_large_element)

        if len(min_heap) == len(max_heap):
            median = (min_heap[0] - max_heap[0]) / 2
        else:
            median = min_heap[0]
        medians.append(median)

    return medians


assert not get_running_medians(None)
assert not get_running_medians([])
assert get_running_medians([2, 5]) == [2, 3.5]
assert get_running_medians([3, 3, 3, 3]) == [3, 3, 3, 3]
assert get_running_medians([2, 1, 5, 7, 2, 0, 5]) == [2, 1.5, 2, 3.5, 2, 2, 2]

```



Question 34

This problem was asked by Quora.

Given a string, find the palindrome that can be made by inserting the fewest number of characters as possible anywhere in the word. If there is more than one palindrome of minimum length that can be made, return the lexicographically earliest one (the first one alphabetically).

For example, given the string "race", you should return "ecarace", since we can add three letters to it (which is the smallest amount to make a palindrome). There are seven other palindromes that can be made from "race" by adding three letters, but "ecarace" comes first alphabetically.

As another example, given the string "google", you should return "elgoogle".

Answer 34





```

def is_palindrome(s):
    return s[::-1] == s


def get_nearest_palindrome(s):

    if is_palindrome(s):
        return s

    if s[0] == s[-1]:
        return s[0] + get_nearest_palindrome(s[1:-1]) + s[-1]
    else:
        pal_1 = s[0] + get_nearest_palindrome(s[1:]) + s[0]
        pal_2 = s[-1] + get_nearest_palindrome(s[:-1]) + s[-1]

        if len(pal_1) > len(pal_2):
            return pal_2
        elif len(pal_1) < len(pal_2):
            return pal_1
        return pal_1 if pal_1 < pal_2 else pal_2


assert get_nearest_palindrome("racecar") == "racecar"
assert get_nearest_palindrome("google") == "elgoogle"
assert get_nearest_palindrome("egoogle") == "elgoogle"
assert get_nearest_palindrome("elgoog") == "elgoogle"
assert get_nearest_palindrome("race") == "ecarace"

```



Question 35

This problem was asked by Google.

Given an array of strictly the characters 'R', 'G', and 'B', segregate the values of the array so that all the Rs come first, the Gs come second, and the Bs come last. You can only swap elements of the array.

Do this in linear time and in-place.

For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].

Answer 35





```

def swap_indices(arr, i, j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


def pull_elements_to_front(arr, start_index, end_index, letter):
    i = start_index
    j = end_index
    last_letter_index = -1

    while i < j:
        if arr[i] == letter:
            last_letter_index = i
            i += 1
        elif arr[j] != letter:
            j -= 1
        else:
            last_letter_index = i
            swap_indices(arr, i, j)

    return last_letter_index


def reorder_array(arr):
    last_index = pull_elements_to_front(arr, 0, len(arr) - 1, "R")
    pull_elements_to_front(arr, last_index + 1, len(arr) - 1, "G")

    return arr


assert reorder_array(['G', 'R']) == ['R', 'G']
assert reorder_array(['G', 'B', 'R']) == ['R', 'G', 'B']
assert reorder_array(['B', 'G', 'R']) == ['R', 'G', 'B']
assert reorder_array(['G', 'B', 'R', 'R', 'B', 'R', 'G']) == [
    'R', 'R', 'R', 'G', 'G', 'B', 'B']

```



Question 36

This problem was asked by Dropbox.

Given the root to a binary search tree, find the second largest node in the tree.

Answer 36





```

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.data)


def find_largest_and_parent(node):
    parent = None
    while node.right:
        parent = node
        node = node.right

    return node, parent


def find_second_largest(root):
    if not root:
        return None

    second_largest = None
    if root.left and not root.right:
        second_largest, _ = find_largest_and_parent(root.left)
    else:
        _, second_largest = find_largest_and_parent(root)
    print("second_largest", second_largest)

    return second_largest


def test_0():
    node_a = Node(5)

    assert not find_second_largest(node_a)


def test_1():
    node_a = Node(5)
    node_b = Node(3)
    node_c = Node(8)
    node_d = Node(2)
    node_e = Node(4)
    node_f = Node(7)
    node_g = Node(9)
    node_a.left = node_b
    node_a.right = node_c
    node_b.left = node_d
    node_b.right = node_e
    node_c.left = node_f
    node_c.right = node_g

    assert find_second_largest(node_a).data == 8


def test_2():
    node_a = Node(5)
    node_b = Node(3)
    node_d = Node(2)
    node_e = Node(4)
    node_a.left = node_b
    node_b.left = node_d
    node_b.right = node_e

    assert find_second_largest(node_a).data == 4


test_0()
test_1()
test_2()

```



Question 37

This problem was asked by Google.

The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.

For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.

You may also use a list or array to represent a set.

Answer 37





```

def get_power_set(numbers):
    if len(numbers) == 0:
        return [set([])]

    power_set = list()

    current_number = numbers[0]
    child_power_set = get_power_set(numbers[1:])
    power_set.extend(child_power_set)

    for child_set in child_power_set:
        new_set = child_set.copy()
        new_set.add(current_number)
        power_set.append(new_set)

    return power_set


assert get_power_set([]) == [set()]
assert get_power_set([1]) == [set(), {1}]
assert get_power_set([1, 2]) == [set(), {2}, {1}, {1, 2}]
assert get_power_set([1, 2, 3]) == [
    set(), {3}, {2}, {2, 3}, {1}, {1, 3}, {1, 2}, {1, 2, 3}]
assert get_power_set([1, 2, 3, 4]) == [
    set(), {4}, {3}, {3, 4}, {2}, {2, 4}, {2, 3}, {2, 3, 4}, {1}, {1, 4},
    {1, 3}, {1, 3, 4}, {1, 2}, {1, 2, 4}, {1, 2, 3}, {1, 2, 3, 4}]

```



Question 38

This problem was asked by Microsoft.

You have an N by N board. Write a function that, given N, returns the number of possible arrangements of the board where N queens can be placed on the board without threatening each other, i.e. no two queens share the same row, column, or diagonal.

Answer 38





```

def is_valid(board, row):
    if row in board:
        return False

    column = len(board)
    for occupied_column, occupied_row in enumerate(board):
        if abs(occupied_row - row) == abs(occupied_column - column):
            return False

    return True


def get_queen_configurations(board, n):
    if n == len(board):
        return 1

    count = 0
    for row in range(n):
        if is_valid(board, row):
            count += get_queen_configurations(board + [row], n)

    return count


assert not is_valid([0, 2], 0)
assert not is_valid([0, 2], 2)
assert is_valid([0, 8], 3)
assert not is_valid([1, 3], 2)
assert is_valid([], 1)

assert get_queen_configurations([], 2) == 0
assert get_queen_configurations([], 4) == 2
assert get_queen_configurations([], 5) == 10
assert get_queen_configurations([], 8) == 92

```



Question 39

This problem was asked by Dropbox.

Conway's Game of Life takes place on an infinite two-dimensional board of square cells. Each cell is either dead or alive, and at each tick, the following rules apply:

Any live cell with less than two live neighbours dies.
Any live cell with two or three live neighbours remains living.
Any live cell with more than three live neighbours dies.
Any dead cell with exactly three live neighbours becomes a live cell.
A cell neighbours another cell if it is horizontally, vertically, or diagonally adjacent.

Implement Conway's Game of Life. It should be able to be initialized with a starting list of live cell coordinates and the number of steps it should run for. Once initialized, it should print out the board state at each step. Since it's an infinite board, print out only the relevant coordinates, i.e. from the top-leftmost live cell to bottom-rightmost live cell.

You can represent a live cell with an asterisk `*` and a dead cell with a dot `.`.

Answer 39





```

import sys


class Coordinate():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return "C(x={};y={})".format(self.x, self.y)

    def get_adjacent_coordinates(self):
        adjacent_coordinates = list()
        adjacent_coordinates.append(Coordinate(self.x, self.y - 1))
        adjacent_coordinates.append(Coordinate(self.x, self.y + 1))
        adjacent_coordinates.append(Coordinate(self.x - 1, self.y))
        adjacent_coordinates.append(Coordinate(self.x - 1, self.y - 1))
        adjacent_coordinates.append(Coordinate(self.x - 1, self.y + 1))
        adjacent_coordinates.append(Coordinate(self.x + 1, self.y))
        adjacent_coordinates.append(Coordinate(self.x + 1, self.y - 1))
        adjacent_coordinates.append(Coordinate(self.x + 1, self.y + 1))
        return adjacent_coordinates


def get_live_coordinates(all_live_coordinates, coordinates_to_check):
    count = 0
    for coordinate in coordinates_to_check:
        if coordinate in all_live_coordinates:
            count += 1

    return count


def update_liveness_neighbourhood(liveness_neighbourhood, adjacent_coordinates):
    for coordinate in adjacent_coordinates:
        if coordinate not in liveness_neighbourhood:
            liveness_neighbourhood[coordinate] = 0
        liveness_neighbourhood[coordinate] += 1


def play_iteration(board):
    coordinate_set = set(board)
    dead_coordinates = set()
    liveness_neighbourhood = dict()

    for coordinate in board:
        adjacent_coordinates = coordinate.get_adjacent_coordinates()
        live_adjacent_coordinate_count = get_live_coordinates(
            coordinate_set, adjacent_coordinates)
        update_liveness_neighbourhood(
            liveness_neighbourhood, adjacent_coordinates)

        if live_adjacent_coordinate_count < 2 or live_adjacent_coordinate_count > 3:
            dead_coordinates.add(coordinate)

    for coordinate in liveness_neighbourhood:
        if liveness_neighbourhood[coordinate] == 3:
            coordinate_set.add(coordinate)

    new_coordinate_set = coordinate_set - dead_coordinates

    return list(new_coordinate_set)


def print_board(coordinates):
    min_x, min_y, max_x, max_y = sys.maxsize, sys.maxsize, -sys.maxsize, -sys.maxsize

    if not coordinates:
        print(".")
        return

    for coordinate in coordinates:
        if coordinate.x < min_x:
            min_x = coordinate.x
        if coordinate.x > max_x:
            max_x = coordinate.x
        if coordinate.y < min_y:
            min_y = coordinate.y
        if coordinate.y > max_y:
            max_y = coordinate.y

    board = []
    for _ in range(max_x - min_x + 1):
        board.append(["."] * (max_y - min_y + 1))

    for coordinate in coordinates:
        board[coordinate.x - min_x][coordinate.y - min_y] = "*"

    for row in board:
        print(" ".join(row))


def play_game(initial_board, steps):
    print("\nPlaying Game of Life with the intial board:")
    print_board(initial_board)
    current_board = initial_board
    for step in range(steps):
        current_board = play_iteration(current_board)
        print("Iteration: {}".format(step))
        print_board(current_board)


# board is a list of Coordinates
board_0 = [
    Coordinate(0, 0), Coordinate(1, 0), Coordinate(1, 1), Coordinate(1, 5)]
play_game(board_0, 3)


board_1 = [
    Coordinate(0, 0), Coordinate(1, 0), Coordinate(1, 1), Coordinate(1, 5),
    Coordinate(2, 5), Coordinate(2, 6)]
play_game(board_1, 4)


board_2 = [
    Coordinate(0, 0), Coordinate(1, 0), Coordinate(1, 1),
    Coordinate(2, 5), Coordinate(2, 6), Coordinate(3, 9),
    Coordinate(4, 8), Coordinate(5, 10)]
play_game(board_2, 4)

```



Question 40

This problem was asked by Google.

Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find and return the non-duplicated integer.

For example, given `[6, 1, 3, 3, 3, 6, 6]`, return `1`. Given `[13, 19, 13, 13]`, return `19`.

Do this in $O(N)$ time and $O(1)$ space.

Answer 40





```

WORD_SIZE = 32


def get_non_duplicated_number(arr):
    non_duplicate = 0

    for i in range(0, WORD_SIZE):
        sum_i_position_bits = 0
        x = 1 << i
        for j in range(len(arr)):
            if arr[j] & x:
                sum_i_position_bits += 1

        if sum_i_position_bits % 3:
            non_duplicate |= x

    return non_duplicate


assert get_non_duplicated_number([6, 1, 3, 3, 3, 6, 6]) == 1
assert get_non_duplicated_number([13, 19, 13, 13]) == 19

```



Question 41

This problem was asked by Facebook.

Given an unordered list of flights taken by someone, each represented as (origin, destination) pairs, and a starting airport, compute the person's itinerary. If no such itinerary exists, return null. If there are multiple possible itineraries, return the lexicographically smallest one. All flights must be used in the itinerary.

For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')] and starting airport 'YUL', you should return the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].

Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting airport 'COM', you should return null.

Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')] and starting airport 'A', you should return the list ['A', 'B', 'C', 'A', 'C'] even though ['A', 'C', 'A', 'B', 'C'] is also a valid itinerary. However, the first one is lexicographically smaller.

Answer 41





```

def get_itinerary(flights, starting_point, current_itinerary):
    # print("flights", flights)
    # print("starting_point", starting_point)
    # print("current_itinerary", current_itinerary)

    if not flights:
        return current_itinerary + [starting_point]

    updated_itinerary = None
    for index, (city_1, city_2) in enumerate(flights):
        if starting_point == city_1:
            child_itinerary = get_itinerary(
                flights[:index] + flights[index + 1:], city_2, current_itinerary + [city_1])
            if child_itinerary:
                if not updated_itinerary or "".join(child_itinerary) < "".join(updated_itinerary):
                    updated_itinerary = child_itinerary

    # print(updated_itinerary)

    return updated_itinerary


assert get_itinerary([('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'),
                      ('HKO', 'ORD')], "YUL", []) == ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD']
assert not get_itinerary([('SFO', 'COM'), ('COM', 'YYZ')], "YUL", [])
assert get_itinerary([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')], "A", []) == [
    'A', 'B', 'C', 'A', 'C']

```



Question 42

This problem was asked by Google.

Given a list of integers S and a target number k, write a function that returns a subset of S that adds up to k. If such a subset cannot be made, then return null.

Integers can appear more than once in the list. You may assume all numbers in the list are positive.

For example, given `S = [12, 1, 61, 5, 9, 2]` and `k = 24`, return [12, 9, 2, 1] since it sums up to 24.

Answer 42





```

def get_subset_for_sum(arr, k):

    if len(arr) == 0:
        return None

    if arr[0] == k:
        return [arr[0]]

    with_first = get_subset_for_sum(arr[1:], k - arr[0])
    if with_first:
        return [arr[0]] + with_first
    else:
        return get_subset_for_sum(arr[1:], k)


assert not get_subset_for_sum([], 1)
assert get_subset_for_sum([12, 1, 61, 5, 9, 2], 24) == [12, 1, 9, 2]
assert get_subset_for_sum([12, 1, 61, 5, 9, 2], 61) == [61]
assert get_subset_for_sum([12, 1, 61, 5, -108, 2], -106) == [-108, 2]

```



Question 43

This problem was asked by Amazon.

Implement a stack that has the following methods:

* `push(val)`, which pushes an element onto the stack
* `pop()`, which pops off and returns the topmost element of the stack. If there are no elements in the stack, then it should throw an error or return null.
* `max()`, which returns the maximum value in the stack currently. If there are no elements in the stack, then it should throw an error or return null.

Each method should run in constant time.

Answer 43





```

class Stack:
    def __init__(self):
        self.stack = []
        self.max_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.max_stack or val > self.stack[self.max_stack[-1]]:
            self.max_stack.append(len(self.stack) - 1)

    def pop(self):
        if not self.stack:
            return None
        if len(self.stack) - 1 == self.max_stack[-1]:
            self.max_stack.pop()

        return self.stack.pop()

    def max(self):
        if not self.stack:
            return None
        return self.stack[self.max_stack[-1]]


s = Stack()
s.push(1)
s.push(3)
s.push(2)
s.push(5)
assert s.max() == 5
s.pop()
assert s.max() == 3
s.pop()
assert s.max() == 3
s.pop()
assert s.max() == 1
s.pop()
assert not s.max()

s = Stack()
s.push(10)
s.push(3)
s.push(2)
s.push(5)
assert s.max() == 10
s.pop()
assert s.max() == 10
s.pop()
assert s.max() == 10
s.pop()
assert s.max() == 10
s.pop()
assert not s.max()

```



Question 44

This problem was asked by Google.

We can determine how "out of order" an array A is by counting the number of inversions it has. Two elements A[i] and A[j] form an inversion if A[i] > A[j] but i < j. That is, a smaller element appears after a larger element.

Given an array, count the number of inversions it has. Do this faster than O(N^2) time.

You may assume each element in the array is distinct.

For example, a sorted list has zero inversions. The array [2, 4, 1, 3, 5] has three inversions: (2, 1), (4, 1), and (4, 3). The array [5, 4, 3, 2, 1] has ten inversions: every distinct pair forms an inversion.

Answer 44





```

def merge(a_with_inv, b_with_inv):
    i, k = 0, 0
    merged = list()
    a, a_inv = a_with_inv
    b, b_inv = b_with_inv
    inversions = a_inv + b_inv

    while i < len(a) and k < len(b):
        if a[i] < b[k]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[k])
            inversions += len(a[i:])
            k += 1

    while i < len(a):
        merged.append(a[i])
        i += 1
    while k < len(b):
        merged.append(b[k])
        k += 1

    return merged, inversions


def merge_sort(arr):
    if not arr or len(arr) == 1:
        return arr, 0

    mid = len(arr) // 2
    merged_array, inversions = merge(
        merge_sort(arr[:mid]), merge_sort(arr[mid:]))

    return merged_array, inversions


def count_inversions(arr):
    _, inversions = merge_sort(arr)
    return inversions


assert count_inversions([1, 2, 3, 4, 5]) == 0
assert count_inversions([2, 1, 3, 4, 5]) == 1
assert count_inversions([2, 4, 1, 3, 5]) == 3
assert count_inversions([2, 6, 1, 3, 7]) == 3
assert count_inversions([5, 4, 3, 2, 1]) == 10

```



Question 45

This problem was asked by Two Sigma.

Using a function rand5() that returns an integer from 1 to 5 (inclusive) with uniform probability, implement a function rand7() that returns an integer from 1 to 7 (inclusive).

Answer 45





```

from random import randint


def rand5():
    return randint(1, 5)


def rand7():
    i = 5*rand5() + rand5() - 5  # uniformly samples between 1-25
    if i < 22:
        return i % 7 + 1
    return rand7()


num_experiments = 100000
result_dict = dict()
for _ in range(num_experiments):
    number = rand7()
    if number not in result_dict:
        result_dict[number] = 0
    result_dict[number] += 1

desired_probability = 1 / 7
for number in result_dict:
    result_dict[number] = result_dict[number] / num_experiments
    assert round(desired_probability, 2) == round(result_dict[number], 2)

```



Question 46

This problem was asked by Amazon.

Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length, return any one.

For example, the longest palindromic substring of "aabcdcb" is "bcdcb". The longest palindromic substring of "bananas" is "anana".

Answer 46





```

def is_palindrome(s1):
    return s1 == s1[::-1]


def get_longest_palindrome_substring(s):
    if not s or is_palindrome(s):
        return s

    s1 = get_longest_palindrome_substring(s[1:])
    s2 = get_longest_palindrome_substring(s[:-1])

    return s1 if len(s1) >= len(s2) else s2


assert get_longest_palindrome_substring("aabcdcb") == "bcdcb"
assert get_longest_palindrome_substring("bananas") == "anana"

```



Question 47

This problem was asked by Facebook.

Given a array of numbers representing the stock prices of a company in chronological order, write a function that calculates the maximum profit you could have made from buying and selling that stock once. You must buy before you can sell it.

For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could buy the stock at 5 dollars and sell it at 10 dollars.

Answer 47





```

import sys


def get_stock_profit(prices):
    if not prices or len(prices) < 2:
        return

    min_price = prices[0]
    max_diff = -sys.maxsize
    for price in prices[1:]:
        if price - min_price > max_diff:
            max_diff = price - min_price
        if price < min_price:
            min_price = price

    return max_diff


assert get_stock_profit([9]) == None
assert get_stock_profit([9, 11, 8, 5, 7, 10]) == 5
assert get_stock_profit([1, 2, 3, 4, 5]) == 4
assert get_stock_profit([1, 1, 1, 1, 1]) == 0
assert get_stock_profit([1, 1, 1, 2, 1]) == 1
assert get_stock_profit([5, 4]) == -1

```



Question 48

This problem was asked by Google.

Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.

For example, given the following preorder traversal:

```
[a, b, d, e, c, f, g]
```

And the following inorder traversal:

```
[d, b, e, a, f, c, g]
```

You should return the following tree:

```
    a
   / \
  b   c
 / \ / \
d  e f  g
```

Answer 48





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def get_tree(preorder, inorder):
    if not preorder or not inorder:
        return None

    root_char = preorder[0]
    if len(preorder) == 1:
        return Node(root_char)

    root_node = Node(root_char)
    for i, char in enumerate(inorder):
        if char == root_char:
            root_node.left = get_tree(
                preorder=preorder[1:i+1], inorder=inorder[:i])
            root_node.right = get_tree(
                preorder=preorder[i+1:], inorder=inorder[i+1:])

    return root_node


tree = get_tree(preorder=['a', 'b', 'd', 'e', 'c', 'f', 'g'],
                inorder=['d', 'b', 'e', 'a', 'f', 'c', 'g'])
assert tree.val == 'a'
assert tree.left.val == 'b'
assert tree.left.left.val == 'd'
assert tree.left.right.val == 'e'
assert tree.right.val == 'c'
assert tree.right.left.val == 'f'
assert tree.right.right.val == 'g'

tree = get_tree(preorder=['a', 'b', 'd', 'e', 'c', 'g'],
                inorder=['d', 'b', 'e', 'a', 'c', 'g'])
assert tree.val == 'a'
assert tree.left.val == 'b'
assert tree.left.left.val == 'd'
assert tree.left.right.val == 'e'
assert tree.right.val == 'c'
assert tree.right.right.val == 'g'

```



Question 49

This problem was asked by Amazon.

Given an array of numbers, find the maximum sum of any contiguous subarray of the array.

For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would be 137, since we would take elements 42, 14, -5, and 86.

Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would not take any elements.

Do this in O(N) time.

Answer 49





```

def get_max_subarray(arr):
    if not arr or max(arr) < 0:
        return 0

    current_max_sum = arr[0]
    overall_max_sum = arr[0]

    for num in arr[1:]:
        current_max_sum = max(num, current_max_sum + num)
        overall_max_sum = max(overall_max_sum, current_max_sum)

    return overall_max_sum


assert get_max_subarray([34, -50, 42, 14, -5, 86]) == 137
assert get_max_subarray([-5, -1, -8, -9]) == 0
assert get_max_subarray([44, -5, 42, 14, -150, 86]) == 95

```



Question 50

This problem was asked by Microsoft.

Suppose an arithmetic expression is given as a binary tree. Each leaf is an integer and each internal node is one of '+', '−', '∗', or '/'.

Given the root to such a tree, write a function to evaluate it.

For example, given the following tree:

```
    *
   / \
  +    +
 / \  / \
3  2  4  5
```

You should return 45, as it is (3 + 2) * (4 + 5).

Answer 50





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def solve_graph(root):
    if root.val.isnumeric():
        return float(root.val)

    return eval("{} {} {}".format(solve_graph(root.left), root.val, solve_graph(root.right)))


d = Node("3")
e = Node("2")
f = Node("4")
g = Node("5")

b = Node("+")
b.left = d
b.right = e

c = Node("+")
c.left = f
c.right = g

a = Node("*")
a.left = b
a.right = c


assert solve_graph(a) == 45
assert solve_graph(c) == 9
assert solve_graph(b) == 5
assert solve_graph(d) == 3

```



Question 51

This problem was asked by Facebook.

Given a function that generates perfectly random numbers between 1 and k (inclusive), where k is an input, write a function that shuffles a deck of cards represented as an array using only swaps.

It should run in O(N) time.

Hint: Make sure each one of the 52! permutations of the deck is equally likely.

Answer 51





```

from random import randint

NUM_CARDS = 52


def get_random_number(k):
    return randint(0, k)


def shuffle_card_deck():
    cards = [x for x in range(NUM_CARDS)]

    for old_pos in cards:
        new_pos = old_pos + get_random_number(NUM_CARDS - old_pos - 1)
        temp = cards[new_pos]
        cards[new_pos] = cards[old_pos]
        cards[old_pos] = temp

    return cards


for _ in range(10):
    assert all(x in shuffle_card_deck() for x in range(NUM_CARDS))

```



Question 52

This problem was asked by Google.

Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:

set(key, value): sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least recently used item.
get(key): gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.

Answer 52





```

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRU:
    def __init__(self, cache_limit):
        self.cache_limit = cache_limit
        self.cache_contents = dict()
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node):
        prev_node = self.tail.prev
        node.next = self.tail
        node.prev = prev_node
        prev_node.next = node
        self.tail.prev = node

    def set_value(self, key, value):
        if key in self.cache_contents:
            node = self.cache_contents[key]
            self._remove(node)
        node = Node(key, value)
        self._add(node)
        self.cache_contents[key] = node
        if len(self.cache_contents) > self.cache_limit:
            node_to_delete = self.head.next
            self._remove(node_to_delete)
            del self.cache_contents[node_to_delete.key]

    def get_value(self, key):
        if key in self.cache_contents:
            node = self.cache_contents[key]
            self._remove(node)
            self._add(node)
            return node.value

        return None


lru = LRU(cache_limit=3)

assert not lru.get_value("a")
lru.set_value("a", 1)
assert lru.get_value("a") == 1
lru.set_value("b", 2)
lru.set_value("c", 3)
lru.set_value("d", 4)
lru.set_value("e", 5)
lru.set_value("a", 1)
assert lru.get_value("a") == 1
assert not lru.get_value("b")
assert lru.get_value("e") == 5
assert not lru.get_value("c")

```



Question 53

This problem was asked by Apple.

Implement a queue using two stacks. Recall that a queue is a FIFO (first-in, first-out) data structure with the following methods: enqueue, which inserts an element into the queue, and dequeue, which removes it.

Answer 53





```

class Queue:
    def __init__(self):
        self.main_stack = list()
        self.aux_stack = list()

    def __repr__(self):
        return str(self.main_stack)

    def enqueue(self, val):
        self.main_stack.append(val)

    def dequeue(self):
        if not self.main_stack:
            return None

        while self.main_stack:
            self.aux_stack.append(self.main_stack.pop())
        val = self.aux_stack.pop()
        while self.aux_stack:
            self.main_stack.append(self.aux_stack.pop())
        return val


q = Queue()
q.enqueue(1)
assert q.main_stack == [1]
q.enqueue(2)
assert q.main_stack == [1, 2]
q.enqueue(3)
assert q.main_stack == [1, 2, 3]
val = q.dequeue()
assert val == 1
assert q.main_stack == [2, 3]
val = q.dequeue()
assert val == 2
assert q.main_stack == [3]

```



Question 54

This problem was asked by Dropbox.

Sudoku is a puzzle where you're given a partially-filled 9 by 9 grid with digits. The objective is to fill the grid with the constraint that every row, column, and box (3 by 3 subgrid) must contain all of the digits from 1 to 9.

Implement an efficient sudoku solver.

Answer 54





```

BOARD_DIM = 9
GRID_DIM = 3


def get_grid_details(x, y):
    grid_row = x - (x % GRID_DIM)
    grid_col = y - (y % GRID_DIM)
    return grid_row, grid_col


def solveSudokuHelper(board, row_elements, col_elements, grid_elements):

    for i in range(BOARD_DIM):
        for k in range(BOARD_DIM):
            if board[i][k] == ".":
                grid_row, grid_col = get_grid_details(i, k)
                grid_key = "{}-{}".format(grid_row, grid_col)
                for m in range(1, BOARD_DIM + 1):
                    if str(m) not in row_elements[i] and \
                            str(m) not in col_elements[k] and \
                            str(m) not in grid_elements[grid_key]:
                        board[i][k] = str(m)
                        row_elements[i].add(str(m))
                        col_elements[k].add(str(m))
                        grid_elements[grid_key].add(str(m))
                        if solveSudokuHelper(
                                board, row_elements, col_elements, grid_elements):
                            return True
                        else:
                            board[i][k] = "."
                            row_elements[i].remove(str(m))
                            col_elements[k].remove(str(m))
                            grid_elements[grid_key].remove(str(m))
                return False
    return True


def solveSudoku(board):
    """
    :type board: List[List[str]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    row_elements = dict()
    col_elements = dict()
    grid_elements = dict()

    for i in range(BOARD_DIM):
        row_elements[i] = set()
        col_elements[i] = set()

    for i in range(BOARD_DIM):
        for k in range(BOARD_DIM):
            if not i % GRID_DIM and not k % GRID_DIM:
                grid_elements["{}-{}".format(i, k)] = set()

    for i in range(BOARD_DIM):
        for k in range(BOARD_DIM):
            if board[i][k].isnumeric():
                row_elements[i].add(board[i][k])
                col_elements[k].add(board[i][k])

                grid_row, grid_col = get_grid_details(i, k)
                grid_elements[
                    "{}-{}".format(grid_row, grid_col)].add(board[i][k])

    solveSudokuHelper(board, row_elements,
                      col_elements, grid_elements)


sudoku_board_1 = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]
solveSudoku(sudoku_board_1)
print(sudoku_board_1)

```



Question 55

This problem was asked by Microsoft.

Implement a URL shortener with the following methods:

* shorten(url), which shortens the url into a six-character alphanumeric string, such as zLg6wl.
* restore(short), which expands the shortened string into the original url. If no such shortened string exists, return null.

Hint: What if we enter the same URL twice?

Answer 55





```

import hashlib


class UrlShortener:

    def __init__(self):
        self.short_to_url_map = dict()
        self.m = hashlib.sha256
        self.prefix = "http://urlsho.rt/"

    def shorten(self, url):
        sha_signature = self.m(url.encode()).hexdigest()
        short_hash = sha_signature[:6]
        self.short_to_url_map[short_hash] = url
        return self.prefix + short_hash

    def restore(self, short):
        short_hash = short.replace(self.prefix, "")
        return self.short_to_url_map[short_hash]


url_0 = "https://www.tutorialspoint.com/python/string_replace.htm"
us = UrlShortener()
short_0 = us.shorten(url_0)
assert us.restore(short_0) == url_0
short_1 = us.shorten(url_0)
assert us.restore(short_1) == url_0

```



Question 56

This problem was asked by Google.

Given an undirected graph represented as an adjacency matrix and an integer k, write a function to determine whether each vertex in the graph can be colored such that no two adjacent vertices share the same color using at most k colors.

Answer 56





```

def can_color_graph(adjacency_matrix, k):
    max_adjacencies = 0
    for row in adjacency_matrix:
        max_adjacencies = max(max_adjacencies, sum(row))

    return k > max_adjacencies


adjacency_matrix_1 = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
]
assert can_color_graph(adjacency_matrix_1, 4)
assert not can_color_graph(adjacency_matrix_1, 3)

adjacency_matrix_2 = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
assert can_color_graph(adjacency_matrix_2, 4)
assert can_color_graph(adjacency_matrix_2, 1)

adjacency_matrix_3 = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]
assert can_color_graph(adjacency_matrix_3, 4)
assert can_color_graph(adjacency_matrix_3, 3)
assert not can_color_graph(adjacency_matrix_3, 2)

```



Question 57

This problem was asked by Amazon.

Given a string s and an integer k, break up the string into multiple texts such that each text has a length of k or less. You must break it up so that words don't break across lines. If there's no way to break the text up, then return null.

You can assume that there are no spaces at the ends of the string and that there is exactly one space between each word.

For example, given the string "the quick brown fox jumps over the lazy dog" and k = 10, you should return: ["the quick", "brown fox", "jumps over", "the lazy", "dog"]. No string in the list has a length of more than 10.

Answer 57





```

def break_test(sentence, k):
    words = sentence.split()

    broken_text = list()
    char_counter = -1
    current_words = list()
    index = 0
    while index < len(words):
        word = words[index]

        if len(word) > k:
            return None

        if char_counter + len(word) + 1 <= k:
            char_counter += len(word) + 1
            current_words.append(word)
            index += 1
        else:
            broken_text.append(" ".join(current_words))
            char_counter = -1
            current_words = list()

    broken_text.extend(current_words)
    return broken_text


assert not break_test("encyclopedia", 8)
assert break_test("the quick brown fox jumps over the lazy dog", 10) == [
    "the quick", "brown fox", "jumps over", "the lazy", "dog"]

```



Question 58

This problem was asked by Amazon.

An sorted array of integers was rotated an unknown number of times.

Given such an array, find the index of the element in the array in faster than linear time. If the element doesn't exist in the array, return null.

For example, given the array [13, 18, 25, 2, 8, 10] and the element 8, return 4 (the index of 8 in the array).

You can assume all the integers in the array are unique.

Answer 58





```

def find_element(arr, element, start, end):

    if start == end:
        return

    mid = start + ((end - start) // 2)

    if arr[mid] == element:
        return mid
    elif arr[mid] > element:
        if arr[start] >= element:
            return find_element(arr, element, start, mid)
        else:
            return find_element(arr, element, mid, end)
    elif arr[mid] < element:
        if arr[start] <= element:
            return find_element(arr, element, start, mid)
        else:
            return find_element(arr, element, mid, end)


def find_element_main(arr, element):
    element_pos = find_element(arr, element, 0, len(arr))
    return element_pos


assert find_element_main([13, 18, 25, 2, 8, 10], 2) == 3
assert find_element_main([13, 18, 25, 2, 8, 10], 8) == 4
assert find_element_main([25, 2, 8, 10, 13, 18], 8) == 2
assert find_element_main([8, 10, 13, 18, 25, 2], 8) == 0

```



Question 59

This problem was asked by Google.

Implement a file syncing algorithm for two computers over a low-bandwidth network. What if we know the files in the two computers are mostly the same?

Answer 59





```

import hashlib

m = hashlib.md5


class MerkleNode:
    def __init__(self):
        self.parent = None
        self.node_hash = None


class MerkleDirectory(MerkleNode):
    def __init__(self):
        MerkleNode.__init__(self)
        self.children = list()
        self.is_dir = True

    def recalculate_hash(self):
        if self.children:
            collated_hash = ""
            for child in self.children:
                collated_hash += child.node_hash
            self.node_hash = m(collated_hash.encode()).hexdigest()


class MerkleFile(MerkleNode):
    def __init__(self):
        MerkleNode.__init__(self)
        self.node_contents = None
        self.is_dir = False

    def update_contents(self, new_contents):
        self.node_hash = m(new_contents.encode()).hexdigest()
        self.node_contents = new_contents
        if self.parent:
            self.parent.recalculate_hash()

    def add_to_directory(self, dir_node):
        self.parent = dir_node
        dir_node.children.append(self)

        while dir_node:
            dir_node.recalculate_hash()
            dir_node = dir_node.parent


a_1 = MerkleFile()
b_1 = MerkleDirectory()
a_1.update_contents("abc")
a_1.add_to_directory(b_1)
a_1.update_contents("abcd")

a_2 = MerkleFile()
b_2 = MerkleDirectory()
a_2.update_contents("abc")
a_2.add_to_directory(b_2)

# Use a tree comparison algorithm to find differences and sync them

```



Question 60

This problem was asked by Facebook.

Given a multiset of integers, return whether it can be partitioned into two subsets whose sums are the same.

For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which both add up to 55.

Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't split it up into two subsets that add up to the same sum.

Answer 60





```

def partition_helper(mset, start, end, outer_sum, inner_sum):
    if start >= end:
        return False
    if outer_sum == inner_sum:
        return True

    return \
        partition_helper(mset, start + 1, end, outer_sum + mset[start],
                         inner_sum - mset[start]) or \
        partition_helper(mset, start, end - 1, outer_sum + mset[end],
                         inner_sum - mset[end])


def can_partition(mset):
    if not mset:
        return True

    if sum(mset) % 2 == 1:
        return False

    mset.sort()

    return partition_helper(mset, 0, len(mset) - 1, 0, sum(mset))


assert can_partition([15, 5, 20, 10, 35, 15, 10])
assert not can_partition([15, 5, 20, 10, 35])
assert can_partition([1, 2, 3, 4, 9, 1])
assert can_partition([1, 1, 1, 1, 1, 1, 6])

```



Question 61

This problem was asked by Google.

Implement integer exponentiation. That is, implement the pow(x, y) function, where x and y are integers and returns x^y.

Do this faster than the naive method of repeated multiplication.

For example, pow(2, 10) should return 1024.

Answer 61





```

def pow(x, y):
    if not x:
        return 0
    elif not y:
        return 1

    y_abs = abs(y)

    current_pow = 1
    prev_result, prev_pow = 0, 0
    result = x
    while current_pow <= y_abs:
        prev_result = result
        prev_pow = current_pow
        result *= result
        current_pow *= 2

    prev_result *= pow(x, y_abs - prev_pow)

    return 1/prev_result if y != y_abs else prev_result


assert pow(2, 2) == 4
assert pow(2, 10) == 1024
assert pow(2, 1) == 2
assert pow(3, 3) == 27
assert pow(10, 3) == 1000
assert pow(2, -3) == 0.125
assert pow(10, -2) == 0.01
assert pow(5, 0) == 1
assert pow(0, 2) == 0

```



Question 62

This problem was asked by Facebook.

There is an N by M matrix of zeroes. Given N and M, write a function to count the number of ways of starting at the top-left corner and getting to the bottom-right corner. You can only move right or down.

For example, given a 2 by 2 matrix, you should return 2, since there are two ways to get to the bottom-right:
* Right, then down
* Down, then right

Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.

Answer 62





```

def matrix_traversal_helper(row_count, col_count, curr_row, curr_col):

    if curr_row == row_count - 1 and curr_col == col_count - 1:
        return 1

    count = 0
    if curr_row < row_count - 1:
        count += matrix_traversal_helper(row_count, col_count,
                                         curr_row + 1, curr_col)
    if curr_col < col_count - 1:
        count += matrix_traversal_helper(row_count, col_count,
                                         curr_row, curr_col + 1)

    return count


def get_matrix_traversals(row_count, col_count):
    if not row_count or not col_count:
        return None
    count = matrix_traversal_helper(row_count, col_count, 0, 0)
    return count


assert not get_matrix_traversals(1, 0)
assert get_matrix_traversals(1, 1) == 1
assert get_matrix_traversals(2, 2) == 2
assert get_matrix_traversals(5, 5) == 70

```



Question 63

This problem was asked by Microsoft.

Given a 2D matrix of characters and a target word, write a function that returns whether the word can be found in the matrix by going left-to-right, or up-to-down.

For example, given the following matrix:

```
[['F', 'A', 'C', 'I'],
 ['O', 'B', 'Q', 'P'],
 ['A', 'N', 'O', 'B'],
 ['M', 'A', 'S', 'S']]
```

and the target word 'FOAM', you should return true, since it's the leftmost column. Similarly, given the target word 'MASS', you should return true, since it's the last row.

Answer 63





```

def get_row_word(matrix, word_len, rows, x, y):
    row_chars = list()
    for i in range(word_len):
        row_chars.append(matrix[x + i][y])

    return "".join(row_chars)


def get_col_word(matrix, word_len, cols, x, y):
    return "".join(matrix[x][y:y + word_len])


def word_checker(matrix, word, word_len, rows, cols, x, y):

    if x >= rows or y >= cols:
        return False

    row_word, col_word = None, None
    if x + word_len <= rows and y < cols:
        row_word = get_row_word(matrix, word_len, rows, x, y)
    if y + word_len <= cols and x < rows:
        col_word = get_col_word(matrix, word_len, cols, x, y)

    if row_word == word or col_word == word:
        return True

    check_1 = word_checker(matrix, word, word_len, rows, cols, x + 1, y) \
        if col_word else None
    check_2 = word_checker(matrix, word, word_len, rows, cols, x, y + 1) \
        if row_word else None

    return check_1 or check_2


def word_exists(matrix, word):
    rows = len(matrix)
    cols = len(matrix[0])
    word_len = len(word)

    return word_checker(matrix, word, word_len, rows, cols, 0, 0)


matrix = [['F', 'A', 'C', 'I'],
          ['O', 'B', 'Q', 'P'],
          ['A', 'N', 'O', 'B'],
          ['M', 'A', 'S', 'S']]

assert not word_exists(matrix, "FOAMS")
assert word_exists(matrix, "FOAM")
assert word_exists(matrix, "MASS")
assert not word_exists(matrix, "FORM")

```



Question 64

This problem was asked by Google.

A knight's tour is a sequence of moves by a knight on a chessboard such that all squares are visited once.

Given N, write a function to return the number of knight's tours on an N by N chessboard.

Answer 64





```

import random


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return "Position=[x={},y={}]".format(self.x, self.y)


def get_potential_knight_moves(start, size, visited):
    moves = list()
    moves.append(Position(start.x + 1, start.y + 2))
    moves.append(Position(start.x + 1, start.y - 2))
    moves.append(Position(start.x - 1, start.y + 2))
    moves.append(Position(start.x - 1, start.y - 2))
    moves.append(Position(start.x + 2, start.y + 1))
    moves.append(Position(start.x + 2, start.y - 1))
    moves.append(Position(start.x - 2, start.y + 1))
    moves.append(Position(start.x - 2, start.y - 1))

    valid_moves = [pos for pos in moves if
                   pos.x >= 0 and pos.x < size and
                   pos.y >= 0 and pos.y < size and
                   pos not in visited]

    return valid_moves


def run_knights_tour(start, size, visited):
    if len(visited) == size * size:
        return 1

    moves = get_potential_knight_moves(start, size, visited)

    count = 0
    for move in moves:
        tmp_visted = visited.copy()
        tmp_visted.add(move)
        count += run_knights_tour(move, size, tmp_visted)

    return count


def count_knights_tours(size):
    count = 0
    for i in range(size):
        for j in range(size):
            start = Position(i, j)
            count += run_knights_tour(start, size, set([start]))

    return count


assert count_knights_tours(1) == 1
assert count_knights_tours(2) == 0
assert count_knights_tours(3) == 0
assert count_knights_tours(4) == 0
assert count_knights_tours(5) == 1728

```



Question 65

This problem was asked by Amazon.

Given a N by M matrix of numbers, print out the matrix in a clockwise spiral.

For example, given the following matrix:

```
[[1,  2,  3,  4,  5],
 [6,  7,  8,  9,  10],
 [11, 12, 13, 14, 15],
 [16, 17, 18, 19, 20]]
```
You should print out the following:

```
1, 2, 3, 4, 5, 10, 15, 20, 19, 18, 17, 16, 11, 6, 7, 8, 9, 14, 13, 12
```

Answer 65





```

def get_spiral(matrix, srow, scol, erow, ecol):
    numbers = list()
    for i in range(scol, ecol + 1):
        numbers.append(matrix[srow][i])

    for i in range(srow + 1, erow + 1):
        numbers.append(matrix[i][ecol])

    if srow < erow:
        for i in range(ecol - 1, srow - 1, -1):
            numbers.append(matrix[erow][i])

    if scol < ecol:
        for i in range(erow - 1, srow, -1):
            numbers.append(matrix[i][scol])

    return numbers


def spiral_helper(matrix):
    srow, scol, erow, ecol = 0, 0, len(matrix) - 1, len(matrix[0]) - 1
    clockwise_numbers = list()
    while srow < erow or scol < ecol:
        clockwise_numbers.extend(get_spiral(matrix, srow, scol, erow, ecol))
        if srow < erow:
            srow += 1
            erow -= 1
        if scol < ecol:
            scol += 1
            ecol -= 1

    return clockwise_numbers


matrix_0 = [[1,  2,  3,  4,  5],
            [6,  7,  8,  9,  10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20]]
assert spiral_helper(matrix_0) == [
    1, 2, 3, 4, 5, 10, 15, 20, 19, 18, 17, 16, 11, 6, 7, 8, 9, 14, 13, 12]

matrix_1 = [[1,  2,  3],
            [6,  7,  8],
            [11, 12, 13],
            [16, 17, 18]]
assert spiral_helper(matrix_1) == [
    1, 2, 3, 8, 13, 18, 17, 16, 11, 6, 7, 12]

matrix_2 = [[1, 4], [2, 5], [3, 6]]
assert spiral_helper(matrix_2) == [1, 4, 5, 6, 3, 2]

matrix_3 = [[1, 2, 3]]
assert spiral_helper(matrix_3) == [1, 2, 3]

matrix_4 = [[1], [2], [3]]
assert spiral_helper(matrix_4) == [1, 2, 3]

```



Question 66

This problem was asked by Square.

Assume you have access to a function toss_biased() which returns 0 or 1 with a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know the bias of the coin.

Write a function to simulate an unbiased coin toss.

Answer 66





```

# assume HEADS = 1 and TAILS = 0 or vice-versa

from random import random


def get_biased_flip():
    rand = random()
    return 0 if rand < 0.7 else 1


num_experiments = 1000000
results_biased = {1: 0, 0: 0}
results_unbiased = {1: 0, 0: 0}
for i in range(num_experiments):
    flip = get_biased_flip()
    results_biased[flip] += 1
    flip = not flip if i % 2 == 0 else flip
    results_unbiased[flip] += 1

for key in results_biased:
    results_biased[key] /= num_experiments
    results_biased[key] = round(results_biased[key], 2)
for key in results_unbiased:
    results_unbiased[key] /= num_experiments
    results_unbiased[key] = round(results_unbiased[key], 2)

assert results_biased[0] == 0.7
assert results_biased[1] == 0.3
assert results_unbiased[0] == 0.5
assert results_unbiased[1] == 0.5

```



Question 67

This problem was asked by Google.

Implement an LFU (Least Frequently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:

* `set(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least frequently used item. If there is a tie, then the least recently used key should be removed.
* `get(key)`: gets the value at key. If no such key exists, return null. 

Each operation should run in O(1) time.

Answer 67





```

class DuplicateException(Exception):
    pass


class NotFoundException(Exception):
    pass


class Node(object):
    """Node containing data, pointers to previous and next node."""

    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
        # Number of nodes in list.
        self.count = 0

    def add_node(self, cls, data):
        """Add node instance of class cls."""

        return self.insert_node(cls, data, self.tail, None)

    def insert_node(self, cls, data, prev, next):
        """Insert node instance of class cls."""

        node = cls(data)
        node.prev = prev
        node.next = next
        if prev:
            prev.next = node
        if next:
            next.prev = node
        if not self.head or next is self.head:
            self.head = node
        if not self.tail or prev is self.tail:
            self.tail = node
        self.count += 1
        return node

    def remove_node(self, node):
        if node is self.tail:
            self.tail = node.prev
        else:
            node.next.prev = node.prev
        if node is self.head:
            self.head = node.next
        else:
            node.prev.next = node.next
        self.count -= 1

    def remove_node_by_data(self, data):
        """Remove node which data is equal to data."""

        node = self.head
        while node:
            if node.data == data:
                self.remove_node(node)
                break
            node = node.next

    def get_nodes_data(self):
        """Return list nodes data as a list."""

        data = []
        node = self.head
        while node:
            data.append(node.data)
            node = node.next
        return data


class FreqNode(DoublyLinkedList, Node):
    """Frequency node.

    Frequency node contains a linked list of item nodes with same frequency.
    """

    def __init__(self, data):
        DoublyLinkedList.__init__(self)
        Node.__init__(self, data)

    def add_item_node(self, data):
        node = self.add_node(ItemNode, data)
        node.parent = self
        return node

    def insert_item_node(self, data, prev, next):
        node = self.insert_node(ItemNode, data, prev, next)
        node.parent = self
        return node

    def remove_item_node(self, node):
        self.remove_node(node)

    def remove_item_node_by_data(self, data):
        self.remove_node_by_data(data)


class ItemNode(Node):
    def __init__(self, data):
        Node.__init__(self, data)
        self.parent = None


class LfuItem(object):
    def __init__(self, data, parent, node):
        self.data = data
        self.parent = parent
        self.node = node


class Cache(DoublyLinkedList):
    def __init__(self):
        DoublyLinkedList.__init__(self)
        self.items = dict()

    def insert_freq_node(self, data, prev, next):
        return self.insert_node(FreqNode, data, prev, next)

    def remove_freq_node(self, node):
        self.remove_node(node)

    def access(self, key):
        try:
            tmp = self.items[key]
        except KeyError:
            raise NotFoundException('Key not found')

        freq_node = tmp.parent
        next_freq_node = freq_node.next

        if not next_freq_node or next_freq_node.data != freq_node.data + 1:
            next_freq_node = self.insert_freq_node(
                freq_node.data + 1, freq_node, next_freq_node)
        item_node = next_freq_node.add_item_node(key)
        tmp.parent = next_freq_node

        freq_node.remove_item_node(tmp.node)
        if freq_node.count == 0:
            self.remove_freq_node(freq_node)

        tmp.node = item_node
        return tmp.data

    def insert(self, key, value):
        if key in self.items:
            raise DuplicateException('Key exists')
        freq_node = self.head
        if not freq_node or freq_node.data != 1:
            freq_node = self.insert_freq_node(1, None, freq_node)

        item_node = freq_node.add_item_node(key)
        self.items[key] = LfuItem(value, freq_node, item_node)

    def get_lfu(self):
        if not len(self.items):
            raise NotFoundException('Items list is empty.')
        return self.head.head.data, self.items[self.head.head.data].data

    def delete_lfu(self):
        """Remove the first item node from the first frequency node.

        Remove the LFU item from the dictionary.
        """
        if not self.head:
            raise NotFoundException('No frequency nodes found')
        freq_node = self.head
        item_node = freq_node.head
        del self.items[item_node.data]
        freq_node.remove_item_node(item_node)
        if freq_node.count == 0:
            self.remove_freq_node(freq_node)

    def __repr__(self):
        """Display access frequency list and items.

        Using the representation:
        freq1: [item, item, ...]
        freq2: [item, item]
        ...
        """
        s = ''
        freq_node = self.head
        while freq_node:
            s += '%s: %s\n' % (freq_node.data, freq_node.get_nodes_data())
            freq_node = freq_node.next
        return s


cache = Cache()
cache.insert('k1', 'v1')
cache.insert('k2', 'v2')
cache.insert('k3', 'v3')
print(cache)
assert cache.access('k2') == 'v2'
print(cache)
cache.get_lfu() == ('k1', 'v1')
cache.delete_lfu()
print(cache)

```



Question 68

This problem was asked by Google.

On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.

You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: (1, 2) is considered the same as (2, 1).

For example, given M = 5 and the list of bishops:
```
(0, 0)
(1, 2)
(2, 2)
(4, 0)
```

The board would look like this:

```
[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
```

You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.

Answer 68





```

def add_new_bishop(location, attack_positions, board_size):
    # count how many times existing bishops can attack
    count = 0
    if location in attack_positions:
        count += 1

    # add new attack positions for future bishops
    new_attack_positions = list()

    i, j = location
    while i > 0 and j > 0:
        i -= 1
        j -= 1
        new_attack_positions.append((i, j))
    i, j = location
    while i > 0 and j < board_size - 1:
        i -= 1
        j += 1
        new_attack_positions.append((i, j))
    i, j = location
    while i < board_size - 1 and j > 0:
        i += 1
        j -= 1
        new_attack_positions.append((i, j))
    i, j = location
    while i < board_size - 1 and j < board_size - 1:
        i += 1
        j += 1
        new_attack_positions.append((i, j))

    attack_positions.extend(new_attack_positions)

    return count, attack_positions


def get_attack_vectors(bishop_locations, board_size):
    attack_positions = list()
    total_count = 0
    for location in bishop_locations:
        count, attack_positions = add_new_bishop(
            location, attack_positions, board_size)
        total_count += count

    return total_count


assert get_attack_vectors([(0, 0), (1, 2), (2, 2), (4, 0)], 5) == 2
assert get_attack_vectors([(0, 0), (1, 2), (2, 2)], 5) == 1

```



Question 69

This problem was asked by Facebook.

Given a list of integers, return the largest product that can be made by multiplying any three integers.

For example, if the list is [-10, -10, 5, 2], we should return 500, since that's -10 * -10 * 5.

You can assume the list has at least three integers.

Answer 69





```

import sys


def get_pairwise_products(arr):
    pairwise_products = list()
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j:
                pairwise_products.append([set([i, j]), arr[i] * arr[j]])

    return pairwise_products


def get_largest_product(arr):
    pairwise_products = get_pairwise_products(arr)
    max_triple = -1 * sys.maxsize
    for i in range(len(arr)):
        for prev_indices, product in pairwise_products:
            if i not in prev_indices:
                triple_prod = arr[i] * product
                if triple_prod > max_triple:
                    max_triple = triple_prod

    return max_triple


assert get_largest_product([-10, -10, 5, 2]) == 500
assert get_largest_product([-10, 10, 5, 2]) == 100

```



Question 70

A number is considered perfect if its digits sum up to exactly 10.

Given a positive integer n, return the n-th perfect number.

For example, given 1, you should return 19. Given 2, you should return 28.

Answer 70





```

def get_perfect_number(n):
    tmp_sum = 0
    for char in str(n):
        tmp_sum += int(char)

    return (n * 10) + (10 - tmp_sum)


assert get_perfect_number(1) == 19
assert get_perfect_number(2) == 28
assert get_perfect_number(3) == 37
assert get_perfect_number(10) == 109
assert get_perfect_number(11) == 118
assert get_perfect_number(19) == 190

```



Question 71

This problem was asked by Two Sigma.

Using a function rand7() that returns an integer from 1 to 7 (inclusive) with uniform probability, implement a function rand5() that returns an integer from 1 to 5 (inclusive).

(repeated question - Problem 45)

Answer 71





```

from random import randint


def rand5():
    return randint(1, 5)


def rand7():
    i = 5*rand5() + rand5() - 5  # uniformly samples between 1-25
    if i < 22:
        return i % 7 + 1
    return rand7()


num_experiments = 100000
result_dict = dict()
for _ in range(num_experiments):
    number = rand7()
    if number not in result_dict:
        result_dict[number] = 0
    result_dict[number] += 1

desired_probability = 1 / 7
for number in result_dict:
    result_dict[number] = result_dict[number] / num_experiments
    assert round(desired_probability, 2) == round(result_dict[number], 2)

```



Question 72

This problem was asked by Google.

In a directed graph, each node is assigned an uppercase letter. We define a path's value as the number of most frequently-occurring letter along that path. For example, if a path in the graph goes through "ABACA", the value of the path is 3, since there are 3 occurrences of 'A' on the path.

Given a graph with n nodes and m directed edges, return the largest value path of the graph. If the largest value is infinite, then return null.

The graph is represented with a string and an edge list. The i-th character represents the uppercase letter of the i-th node. Each tuple in the edge list (i, j) means there is a directed edge from the i-th node to the j-th node. Self-edges are possible, as well as multi-edges.

For example, the following input graph:

```
ABACA
[(0, 1),
 (0, 2),
 (2, 3),
 (3, 4)]
```
Would have maximum value 3 using the path of vertices `[0, 2, 3, 4], (A, A, C, A)`.

The following input graph:

```
A
[(0, 0)]
```

Should return null, since we have an infinite loop.

Answer 72





```

class GraphPath:
    def __init__(self, nodes=set(), letter_counts=dict()):
        self.nodes = nodes
        self.letter_counts = letter_counts

    def __repr__(self):
        return "nodes={}, letters={}".format(self.nodes, self.letter_counts)


def get_max_value_string(graph_path, node, adjacency_map):
    if node in graph_path.nodes:
        return [graph_path]

    new_nodes = graph_path.nodes.copy()
    new_nodes.add(node)
    new_letter_counts = graph_path.letter_counts.copy()
    if node[0] not in new_letter_counts:
        new_letter_counts[node[0]] = 0
    new_letter_counts[node[0]] += 1

    new_graph_path = GraphPath(new_nodes, new_letter_counts)

    if node not in adjacency_map:
        return [new_graph_path]

    paths = list()
    for child_node in adjacency_map[node]:
        new_paths = get_max_value_string(
            new_graph_path, child_node, adjacency_map)
        paths.extend(new_paths)

    return paths


def get_max_value_string_helper(graph_string, edge_list):

    letter_counts = dict()
    nodes = list()
    for char in graph_string:
        if char not in letter_counts:
            letter_counts[char] = 0
        else:
            letter_counts[char] += 1
        nodes.append("{}{}".format(char, letter_counts[char]))

    adjacency_map = dict()
    for start, end in edge_list:
        if nodes[start] not in adjacency_map:
            adjacency_map[nodes[start]] = set()
        if nodes[start] != nodes[end]:
            adjacency_map[nodes[start]].add(nodes[end])

    paths = list()
    graph_path = GraphPath()
    for node in adjacency_map:
        new_paths = get_max_value_string(graph_path, node, adjacency_map)
        paths.extend(new_paths)

    max_value = 0
    for path in paths:
        max_path_value = max(path.letter_counts.values())
        if max_path_value > max_value:
            max_value = max_path_value

    return max_value if max_value > 0 else None


assert get_max_value_string_helper(
    "ABACA", [(0, 1), (0, 2), (2, 3), (3, 4)]) == 3
assert not get_max_value_string_helper("A", [(0, 0)])

```



Question 73

This problem was asked by Google.

Given the head of a singly linked list, reverse it in-place.

Answer 73





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes

def reverse_list(head, new_head):
    if not head:
        return new_head

    old_head = head.next
    head.next = new_head

    return reverse_list(old_head, head)


assert not get_list(reverse_list(get_nodes([]), None))
assert get_list(reverse_list(get_nodes([1]), None)) == [1]
assert get_list(reverse_list(get_nodes([1, 2]), None)) == [2, 1]
assert get_list(reverse_list(get_nodes([1, 2, 3]), None)) == [3, 2, 1]


```



Question 74

This problem was asked by Apple.

Suppose you have a multiplication table that is N by N. That is, a 2D array where the value at the i-th row and j-th column is (i + 1) * (j + 1) (if 0-indexed) or i * j (if 1-indexed).

Given integers N and X, write a function that returns the number of times X appears as a value in an N by N multiplication table.

For example, given N = 6 and X = 12, you should return 4, since the multiplication table looks like this:

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| 1   | 2   | 3   | 4   | 5   | 6   |
| 2   | 4   | 6   | 8   | 10  | 12  |
| 3   | 6   | 9   | 12  | 15  | 18  |
| 4   | 8   | 12  | 16  | 20  | 24  |
| 5   | 10  | 15  | 20  | 25  | 30  |
| 6   | 12  | 18  | 24  | 30  | 36  |

And there are 4 12's in the table.

Answer 74





```

def get_mult_count(n, x):
    if n == 1:
        return n

    tuples = list()
    for i in range(1, (x + 1) // 2):
        if not x % i:
            tuples.append((i, x // i))

    return len(tuples)


assert get_mult_count(1, 1) == 1
assert get_mult_count(6, 12) == 4
assert get_mult_count(2, 4) == 1
assert get_mult_count(3, 6) == 2

```



Question 75

This problem was asked by Microsoft.

Given an array of numbers, find the length of the longest increasing subsequence in the array. The subsequence does not necessarily have to be contiguous.

For example, given the array `[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]`, the longest increasing subsequence has length 6: it is `0, 2, 6, 9, 11, 15`.

Answer 75





```

cache = None


def get_subseq(arr, start):
    if start == len(arr):
        return 0

    current = arr[start]
    max_inc = 1
    for index in range(start + 1, len(arr)):
        if arr[index] >= current:
            if index in cache:
                count = cache[index]
            else:
                count = get_subseq(arr, index) + 1
                cache[index] = count
            if count > max_inc:
                max_inc = count

    return max_inc


def get_subseq_helper(arr):
    global cache
    cache = dict()
    return get_subseq(arr, 0)


assert get_subseq_helper([]) == 0
assert get_subseq_helper([0, 1]) == 2
assert get_subseq_helper([0, 2, 1]) == 2
assert get_subseq_helper([0, 1, 2]) == 3
assert get_subseq_helper([2, 1, 0]) == 1
assert get_subseq_helper(
    [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]) == 6

```



Question 76

This problem was asked by Google.

You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is lexicographically later as you go down each row. It does not matter whether each row itself is ordered lexicographically.

For example, given the following table:

```
cba
daf
ghi
```

This is not ordered because of the a in the center. We can remove the second column to make it ordered:

```
ca
df
gi
```

So your function should return 1, since we only needed to remove 1 column.

As another example, given the following table:

```
abcdef
```

Your function should return 0, since the rows are already ordered (there's only one row).

As another example, given the following table:

```
zyx
wvu
tsr
```

Your function should return 3, since we would need to remove all the columns to order it.

Answer 76





```

def get_col_rem_count(matrix):
    if not matrix:
        return 0

    rows = len(matrix)
    if rows == 1:
        return 0

    cols = len(matrix[0])

    col_drop_count = 0
    for i in range(cols):
        for k in range(1, rows):
            if matrix[k][i] < matrix[k-1][i]:
                col_drop_count += 1
                break

    return col_drop_count


assert get_col_rem_count(["cba", "daf", "ghi"]) == 1
assert get_col_rem_count(["abcdef"]) == 0
assert get_col_rem_count(["zyx", "wvu", "tsr"]) == 3

```



Question 77

This problem was asked by Snapchat.

Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.

The input list is not necessarily ordered in any way.

For example, given `[(1, 3), (5, 8), (4, 10), (20, 25)]`, you should return `[(1, 3), (4, 10), (20, 25)]`.

Answer 77





```

def merge_overlaps(intervals):
    interval_starts, interval_ends = set(), set()
    for start, end in intervals:
        interval_starts.add(start)
        interval_ends.add(end)

    min_start = min(interval_starts)
    max_end = max(interval_ends)
    current_active = 0
    instant_statuses = list([current_active])
    merged = list()
    for i in range(min_start, max_end + 1):
        if i in interval_ends:
            current_active -= 1
        if i in interval_starts:
            current_active += 1
        instant_statuses.append(current_active)

    start, end = None, None
    for i in range(len(instant_statuses)):
        if instant_statuses[i] and not instant_statuses[i-1]:
            start = i
        if not instant_statuses[i] and instant_statuses[i-1]:
            end = i
            merged.append((start, end))
            start, end = None, None
    return merged


assert merge_overlaps([(1, 3), (5, 8), (4, 10), (20, 25)]) == [
    (1, 3), (4, 10), (20, 25)]
assert merge_overlaps([(1, 3), (5, 8), (4, 10), (20, 25), (6, 12)]) == [
    (1, 3), (4, 12), (20, 25)]

```



Question 78

This problem was asked recently by Google.

Given k sorted singly linked lists, write a function to merge all the lists into one sorted singly linked list.

Answer 78





```

import sys
from heapq import heappush, heappop


class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def merge_lists(all_lists):
    merged_head = Node(None)
    merged_tail = merged_head
    candidates = list()
    counter = 0
    for llist in all_lists:
        heappush(candidates, (llist.val, counter, llist))
        counter += 1

    while candidates:
        _, _, new_node = heappop(candidates)

        if new_node.next:
            heappush(candidates, (new_node.next.val, counter, new_node.next))
            counter += 1

        merged_tail.next = new_node
        merged_tail = new_node

    return merged_head.next


assert get_list(merge_lists([get_nodes([1, 4, 6]),
                             get_nodes([1, 3, 7])])) == [1, 1, 3, 4, 6, 7]
assert get_list(merge_lists([get_nodes([1, 4, 6]),
                             get_nodes([2, 3, 9]),
                             get_nodes([1, 3, 7])])) == [1, 1, 2, 3, 3, 4, 6, 7, 9]


```



Question 79

This problem was asked by Facebook.

Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at most 1 element.

For example, given the array `[10, 5, 7]`, you should return true, since we can modify the 10 into a 1 to make the array non-decreasing.

Given the array `[10, 5, 1]`, you should return false, since we can't modify any one element to get a non-decreasing array.

Answer 79





```

def can_edit(arr):
    decr_pairs = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            decr_pairs += 1

    return decr_pairs <= 1


assert can_edit([10, 5, 7])
assert not can_edit([10, 5, 1])
assert can_edit([1, 10, 5, 7])


```



Question 80

This problem was asked by Google.

Given the root of a binary tree, return a deepest node. For example, in the following tree, return d.

```
    a
   / \
  b   c
 /
d
```

Answer 80





```

class Node:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.val)


def find_deepest_node(root, depth):
    if not root.left and not root.right:
        return (root, depth)

    left_depth = depth
    right_depth = depth
    if root.left:
        left_deepest, left_depth = find_deepest_node(root.left, depth + 1)
    if root.right:
        right_deepest, right_depth = find_deepest_node(root.right, depth + 1)

    return (left_deepest, left_depth) if left_depth > right_depth \
        else (right_deepest, right_depth)


def find_deepest_node_helper(root):
    return find_deepest_node(root, 0)[0]


a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')

a.left = b
a.right = c
b.left = d

assert find_deepest_node_helper(a) == d

c.left = e
e.left = f


assert find_deepest_node_helper(a) == f

```



Question 81

This problem was asked by Yelp.

Given a mapping of digits to letters (as in a phone number), and a digit string, return all possible letters the number could represent. You can assume each valid number in the mapping is a single digit.

For example if `{'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], }` then `"23"` should return `['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']`.

Answer 81





```

digit_mapping = {
    '2': ['a', 'b', 'c'],
    '3': ['d', 'e', 'f']
}


def get_letter_strings(number_string):
    if not number_string:
        return

    if len(number_string) == 1:
        return digit_mapping[number_string[0]]

    possible_strings = list()
    current_letters = digit_mapping[number_string[0]]
    strings_of_rem_nums = get_letter_strings(number_string[1:])
    for letter in current_letters:
        for string in strings_of_rem_nums:
            possible_strings.append(letter + string)

    return possible_strings


assert get_letter_strings("2") == [
    'a', 'b', 'c']
assert get_letter_strings("23") == [
    'ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
assert get_letter_strings("32") == [
    'da', 'db', 'dc', 'ea', 'eb', 'ec', 'fa', 'fb', 'fc']

```



Question 82

This problem was asked Microsoft.

Using a read7() method that returns 7 characters from a file, implement readN(n) which reads n characters.

For example, given a file with the content "Hello world", three read7() returns "Hello w", "orld" and then "".

Answer 82





```

class FileProxy:
    def __init__(self, contents):
        self.contents = contents
        self.offset = 0
        self.buffer = ""

    def read_7(self):
        start = self.offset
        end = min(self.offset + 7, len(self.contents))
        self.offset = end
        return self.contents[start:end].strip()

    def read_n(self, n):
        while len(self.buffer) < n:
            additional_chars = self.read_7()
            if not (additional_chars):
                break
            self.buffer += additional_chars

        n_chars = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return n_chars.strip()


fp = FileProxy("Hello world")
assert fp.read_7() == "Hello w"
assert fp.read_7() == "orld"
assert fp.read_7() == ""

fp = FileProxy("Hello world")
assert fp.read_n(8) == "Hello wo"
assert fp.read_n(8) == "rld"

fp = FileProxy("Hello world")
assert fp.read_n(4) == "Hell"
assert fp.read_n(4) == "o wo"
assert fp.read_n(4) == "rld"

```



Question 83

This problem was asked by Google.

Invert a binary tree.

For example, given the following tree:
```
    a
   / \
  b   c
 / \  /
d   e f
```

should become:
```
  a
 / \
 c  b
 \  / \
  f e  d
```

Answer 83





```

class Node:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        string = "{}=({},{})".format(self.val, self.left, self.right)
        return string


def invert_tree(root):
    if not root:
        return

    inverted_left = invert_tree(root.right)
    inverted_right = invert_tree(root.left)
    root.left = inverted_left
    root.right = inverted_right

    return root


a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')

a.left = b
a.right = c
b.left = d
b.right = e
c.left = f

inverted_a = invert_tree(a)
assert inverted_a.left == c
assert inverted_a.right == b
assert inverted_a.left.right == f
assert inverted_a.right.right == d

```



Question 84

This problem was asked by Amazon.

Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, so an island is a group of 1s that are neighboring and their perimeter is surrounded by water.

For example, this matrix has 4 islands.

```
1 0 0 0 0
0 0 1 1 0
0 1 1 0 0
0 0 0 0 0
1 1 0 0 1
1 1 0 0 1
```

Answer 84





```

def add_island(x, y, world_map, visited):
    coord = "{}-{}".format(x, y)

    if coord in visited:
        return 0

    visited.add(coord)

    if x > 0 and world_map[x-1][y]:
        add_island(x-1, y, world_map, visited)
    if x < len(world_map) - 1 and world_map[x+1][y]:
        add_island(x+1, y, world_map, visited)
    if y > 0 and world_map[x][y-1]:
        add_island(x, y-1, world_map, visited)
    if y < len(world_map[0]) - 1 and world_map[x][y+1]:
        add_island(x, y+1, world_map, visited)

    return 1


def count_islands(world_map):
    count = 0
    visited = set()
    for i in range(len(world_map)):
        for k in range(len(world_map[0])):
            if world_map[i][k]:
                count += add_island(i, k, world_map, visited)

    return count


world_map = [
    [1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
]
assert count_islands(world_map) == 4

```



Question 85

This problem was asked by Facebook.

Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0, using only mathematical or bit operations. You can assume b can only be 1 or 0.

Answer 85





```

def get_num(x, y, b):
    return x * b + y * abs(b-1)


assert get_num(3, 4, 1) == 3
assert get_num(3, 4, 0) == 4

```



Question 86

This problem was asked by Google.

Given a string of parentheses, write a function to compute the minimum number of parentheses to be removed to make the string valid (i.e. each open parenthesis is eventually closed).

For example, given the string "()())()", you should return 1. Given the string ")(", you should return 2, since we must remove all of them.

Answer 86





```

def get_chars_removed_helper(string, stack, removed):

    if not string and not stack:
        return removed
    elif not string:
        return len(stack) + removed

    if string[0] == ')' and stack and stack[-1] == '(':
        stack.pop()
        return get_chars_removed_helper(string[1:], stack, removed)

    removed_chars_add = get_chars_removed_helper(
        string[1:], stack + [string[0]], removed)
    removed_chars_ignore = get_chars_removed_helper(
        string[1:], stack, removed + 1)

    return min(removed_chars_add, removed_chars_ignore)


def get_chars_removed(string):
    chars_removed = get_chars_removed_helper(string, list(), 0)
    return chars_removed


assert get_chars_removed("()())()") == 1
assert get_chars_removed(")(") == 2
assert get_chars_removed("()(((") == 3
assert get_chars_removed(")()(((") == 4

```



Question 87

This problem was asked by Uber.

A rule looks like this:

`A NE B`

This means this means point A is located northeast of point B.

`A SW C`

means that point A is southwest of C.

Given a list of rules, check if the sum of the rules validate. For example:
```
A N B
B NE C
C N A
```
does not validate, since A cannot be both north and south of C.

```
A NW B
A N B
```

is considered valid.

Answer 87





```

opposites = {
    'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'
}


class Node:
    def __init__(self, val):
        self.val = val
        self.neighbours = {
            'N': set(),
            'S': set(),
            'E': set(),
            'W': set()
        }

    def __repr__(self):
        string = "{}={}".format(self.val, self.neighbours)
        return string

    def __hash__(self):
        return hash(self.val)


class Map:

    def add_rule(self, node_1, direction, node_2):

        for char in direction:
            if node_1 in node_2.neighbours[char] or \
                    node_2 in node_1.neighbours[opposites[char]]:
                raise Exception

            for node in node_1.neighbours[char]:
                self.add_rule(node, char, node_2)

        for char in direction:
            node_2.neighbours[char].add(node_1)
            node_1.neighbours[opposites[char]].add(node_2)


a = Node('a')
b = Node('b')
c = Node('c')

m = Map()
m.add_rule(a, 'N', b)
m.add_rule(b, 'NE', c)
try:
    m.add_rule(c, 'N', a)
except Exception:
    print("Invalid rule")

```



Question 88

This question was asked by ContextLogic.

Implement division of two positive integers without using the division, multiplication, or modulus operators. Return the quotient as an integer, ignoring the remainder.

Answer 88





```

def divide(dividend, divisor):
    if not divisor:
        return

    current_sum = 0
    quotient = 0
    while current_sum <= dividend:
        quotient += 1
        current_sum += divisor

    return quotient - 1


assert not divide(1, 0)
assert divide(1, 1) == 1
assert divide(0, 1) == 0
assert divide(12, 3) == 4
assert divide(13, 3) == 4
assert divide(25, 5) == 5
assert divide(25, 7) == 3

```



Question 89

This problem was asked by LinkedIn.

Determine whether a tree is a valid binary search tree.

A binary search tree is a tree with two children, left and right, and satisfies the constraint that the key in the left child must be less than or equal to the root and the key in the right child must be greater than or equal to the root.

Answer 89





```

import sys


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def is_valid_bst_node_helper(node, lb, ub):

    if node and node.val <= ub and node.val >= lb:
        return is_valid_bst_node_helper(node.left, lb, node.val) and \
            is_valid_bst_node_helper(node.right, node.val, ub)

    return not node  # if node is None, it's a valid BST


def is_valid_bst(root):
    return is_valid_bst_node_helper(root, -sys.maxsize, sys.maxsize)


# Tests


assert is_valid_bst(None)

a = Node(3)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(3)
f = Node(4)

a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert is_valid_bst(a)


a = Node(1)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(3)
f = Node(4)

a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert not is_valid_bst(a)

a = Node(3)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(4)
f = Node(4)

a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert not is_valid_bst(a)

```



Question 90

This question was asked by Google.

Given an integer n and a list of integers l, write a function that randomly generates a number from 0 to n-1 that isn't in l (uniform).

Answer 90





```

from random import randint


def generate_random_num(n, excluded_nums):
    rand = randint(0, n-1)
    if rand in excluded_nums:
        return generate_random_num(n, excluded_nums)

    return rand


def run_experiment(num_samples, n, l):
    error_tolerance = 0.01
    results = dict()
    excluded_nums = set(l)

    for num in range(n):
        results[num] = 0

    for _ in range(num_samples):
        rand = generate_random_num(n, excluded_nums)
        results[rand] += 1
    expected_prob = 1/(n - len(excluded_nums))

    for num in results:
        results[num] /= num_samples
        if num in excluded_nums:
            assert not results[num]
        else:
            assert results[num] > expected_prob - error_tolerance or \
                results[num] < expected_prob + error_tolerance


run_experiment(100000, 6, [])
run_experiment(100000, 6, [1, 5])
run_experiment(100000, 6, [1, 3, 5])

```



Question 91

This problem was asked by Dropbox.

What does the below code snippet print out? How can we fix the anonymous functions to behave as we'd expect?

```python
functions = []
for i in range(10):
    functions.append(lambda : i)

for f in functions:
    print(f())
```

Answer 91





```

# The function prints nine because it uses the variable
# i to determine what to return and i finishes looping
# at 9

functions = []
for i in range(10):
    functions.append(lambda: i)

i = 0
for f in functions:
    print(f())
    i += 1

```



Question 92

This problem was asked by Airbnb.

We're given a hashmap with a key courseId and value a list of courseIds, which represents that the prerequsite of courseId is courseIds. Return a sorted ordering of courses such that we can finish all courses.

Return null if there is no such ordering.

For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSCS300']`.

Answer 92





```

def get_course_order_helper(prereqs, indep, order):
    if not indep:
        return None, None, None
    elif not prereqs:
        return prereqs, indep, order

    new_indep = set()
    for dc in prereqs:
        required = prereqs[dc] - indep
        if not len(required):
            new_indep.add(dc)
            order.append(dc)

    for course in new_indep:
        del prereqs[course]

    return get_course_order_helper(prereqs, indep.union(new_indep), order)


def get_course_order(prereqs):

    indep = set()
    order = list()
    for course in prereqs:
        if not prereqs[course]:
            indep.add(course)
            order.append(course)
        else:
            prereqs[course] = set(prereqs[course])

    for course in indep:
        del prereqs[course]

    _, _, order = get_course_order_helper(prereqs, indep, order)

    return order


prereqs = {
    'CSC100': [],
    'CSC200': [],
    'CSC300': []
}
assert get_course_order(prereqs) == ['CSC100', 'CSC200', 'CSC300']

prereqs = {
    'CSC300': ['CSC100', 'CSC200'],
    'CSC200': ['CSC100'],
    'CSC100': []
}
assert get_course_order(prereqs) == ['CSC100', 'CSC200', 'CSC300']

prereqs = {
    'CSC400': ['CSC200'],
    'CSC300': ['CSC100', 'CSC200'],
    'CSC200': ['CSC100'],
    'CSC100': []
}
assert get_course_order(prereqs) == ['CSC100', 'CSC200', 'CSC400', 'CSC300']

prereqs = {
    'CSC400': ['CSC300'],
    'CSC300': ['CSC100', 'CSC200'],
    'CSC200': ['CSC100'],
    'CSC100': ['CSC400']
}
assert not get_course_order(prereqs)

```



Question 93

This problem was asked by Apple.

Given a tree, find the largest tree/subtree that is a BST.

Given a tree, return the size of the largest tree/subtree that is a BST.

Answer 93





```

import sys


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        string = "val={}:(left={}, right={})".format(
            self.val, self.left, self.right)
        return string


def get_largest_bst_helper(node):
    if not node:
        return (True, node, 0, sys.maxsize, -sys.maxsize)
    if not node.left and not node.right:
        return (True, node, 1, node.val, node.val)

    left_is_bst, left_bst, left_nodes, left_minval, left_maxval = \
        get_largest_bst_helper(node.left)
    right_is_bst, right_bst, right_nodes, right_minval, right_maxval = \
        get_largest_bst_helper(node.right)

    if left_is_bst and right_is_bst:
        if node.left and node.right:
            if node.val >= left_maxval and node.val <= right_minval:
                return (True, node, left_nodes + right_nodes + 1,
                        left_minval, right_maxval)
        elif node.left and node.val >= left_maxval:
            return (True, node, left_nodes + 1, left_minval, node.val)
        elif node.right and node.val >= right_minval:
            return (True, node, left_nodes + 1, node.val, right_maxval)

    if left_nodes > right_nodes:
        return (False, left_bst, left_nodes, left_minval, node.val)
    else:
        return (False, right_bst, right_nodes, node.val, right_maxval)


def get_largest_bst(root):
    _, largest_bst, nodes, _, _ = get_largest_bst_helper(root)
    return (largest_bst, nodes)


# tests

a = Node(3)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(3)
f = Node(4)
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert get_largest_bst(a) == (a, 6)

a = Node(1)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(3)
f = Node(4)
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert get_largest_bst(a) == (b, 3)

a = Node(3)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(4)
f = Node(4)
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert get_largest_bst(a) == (b, 3)

a = Node(3)
b = Node(2)
c = Node(6)
d = Node(1)
e = Node(1)
f = Node(4)
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
assert get_largest_bst(a) == (c, 2)

```



Question 94

This problem was asked by Google.

Given a binary tree of integers, find the maximum path sum between two nodes. The path must go through at least one node, and does not need to go through the root.

Answer 94





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def get_max_path_sum(root, current_max_sum, overall_max_sum):
    if not root:
        return overall_max_sum

    current_max_sum = max(root.val, current_max_sum + root.val)
    overall_max_sum = max(overall_max_sum, current_max_sum)

    left_path_sum = get_max_path_sum(
        root.left, current_max_sum, overall_max_sum)
    right_path_sum = get_max_path_sum(
        root.right, current_max_sum, overall_max_sum)

    return max(overall_max_sum, left_path_sum, right_path_sum)


assert get_max_path_sum(None, 0, 0) == 0

a = Node(1)
assert get_max_path_sum(a, 0, 0) == 1

b = Node(2)
a.left = b
assert get_max_path_sum(a, 0, 0) == 3

c = Node(3)
a.right = c
assert get_max_path_sum(a, 0, 0) == 4

a.val = -1
assert get_max_path_sum(a, 0, 0) == 3

d = Node(4)
b.left = d
assert get_max_path_sum(a, 0, 0) == 6

```



Question 95

This problem was asked by Palantir.

Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.

For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.

Can you perform the operation without allocating extra memory (disregarding the input memory)?

Answer 95





```

def get_greater_permutation(arr):

    if len(arr) < 2:
        return

    for index in range(len(arr) - 1, -1, -1):
        if index > 0 and arr[index - 1] < arr[index]:
            break

    if index == 0:
        arr.reverse()
    else:
        for k in range(len(arr) - 1, index - 1, -1):
            if arr[k] > arr[index - 1]:
                tmp = arr[k]
                arr[k] = arr[index - 1]
                arr[index - 1] = tmp
                break

        sub_array = arr[index:]
        sub_array.reverse()
        arr[index:] = sub_array

    return arr


assert get_greater_permutation([1, 2, 3]) == [1, 3, 2]
assert get_greater_permutation([1, 3, 2]) == [2, 1, 3]
assert get_greater_permutation([3, 2, 1]) == [1, 2, 3]

```



Question 96

This problem was asked by Microsoft.

Given a number in the form of a list of digits, return all possible permutations.

For example, given `[1,2,3]`, return `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`.

Answer 96





```

def get_permutations(arr):
    if len(arr) < 2:
        return [arr]

    permutations = list()
    for i, num in enumerate(arr):
        arr_cp = arr[:i] + arr[i+1:]
        child_perms = get_permutations(arr_cp)
        for perm in child_perms:
            permutations.append([num] + perm)

    return permutations


assert get_permutations([]) == [[]]
assert get_permutations([1]) == [[1]]
assert get_permutations([1, 2]) == [[1, 2], [2, 1]]
assert get_permutations([1, 2, 3]) == \
    [[1, 2, 3], [1, 3, 2], [2, 1, 3],
     [2, 3, 1], [3, 1, 2], [3, 2, 1]]

```



Question 97

This problem was asked by Stripe.

Write a map implementation with a get function that lets you retrieve the value of a key at a particular time.

It should contain the following methods:
```
    set(key, value, time): # sets key to value for t = time.
    get(key, time): # gets the key at t = time.
```

The map should work like this. If we set a key at a particular time, it will maintain that value forever or until it gets set at a later time. In other words, when we get a key at a time, it should return the value that was set for that key set at the most recent time.

Consider the following examples:

```
d.set(1, 1, 0) # set key 1 to value 1 at time 0
d.set(1, 2, 2) # set key 1 to value 2 at time 2
d.get(1, 1) # get key 1 at time 1 should be 1
d.get(1, 3) # get key 1 at time 3 should be 2

d.set(1, 1, 5) # set key 1 to value 1 at time 5
d.get(1, 0) # get key 1 at time 0 should be null
d.get(1, 10) # get key 1 at time 10 should be 1

d.set(1, 1, 0) # set key 1 to value 1 at time 0
d.set(1, 2, 0) # set key 1 to value 2 at time 0
d.get(1, 0) # get key 1 at time 0 should be 2
```

Answer 97





```

import bisect


class TimedMap:
    def __init__(self):
        self.map = dict()

    def __repr__(self):
        return str(self.map)

    def setval(self, key, value, time):
        if key not in self.map:
            self.map[key] = ([time], [value])
            return

        times, values = self.map[key]
        insertion_point = bisect.bisect(times, time)
        times.insert(insertion_point, time)
        values.insert(insertion_point, value)

    def getval(self, key, time):
        if key not in self.map:
            return

        times, values = self.map[key]
        insertion_point = bisect.bisect(times, time)
        if not insertion_point:
            return

        return values[insertion_point - 1]


d = TimedMap()
d.setval(1, 1, 0)
d.setval(1, 2, 2)
assert d.getval(1, 1) == 1
assert d.getval(1, 3) == 2

d = TimedMap()
d.setval(1, 1, 5)
assert not d.getval(1, 0)
assert d.getval(1, 10) == 1

d = TimedMap()
d.setval(1, 1, 0)
d.setval(1, 2, 0)
assert d.getval(1, 0) == 2

```



Question 98

This problem was asked by Coursera.

Given a 2D board of characters and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example, given the following board:
```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
```
`exists(board, "ABCCED")` returns true, `exists(board, "SEE")` returns true, `exists(board, "ABCB")` returns false.

Answer 98





```

def check_new_coordinate(word, row, col, used_coordinates):
    expected_char = word[0]
    copy_coordinates = used_coordinates.copy()
    char = board[row][col]
    result = False
    if expected_char == char and "{}-{}".format(row, col) not in copy_coordinates:
        copy_coordinates.add("{}-{}".format(row, col))
        result = existence_helper(
            word[1:], board, row, col, copy_coordinates)
    return result


def existence_helper(word, board, crow, ccol, used_coordinates):
    if not word:
        return True

    top, bottom, left, right = (False, False, False, False)
    if crow > 0:
        top = check_new_coordinate(word, crow - 1, ccol, used_coordinates)
    if crow < len(board) - 1:
        bottom = check_new_coordinate(word, crow + 1, ccol, used_coordinates)
    if ccol > 0:
        left = check_new_coordinate(word, crow, ccol - 1, used_coordinates)
    if ccol < len(board[0]) - 1:
        right = check_new_coordinate(word, crow, ccol + 1, used_coordinates)

    return top or bottom or left or right


def exists(board, word):
    if not word:
        return False

    first_char = word[0]
    result = False
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == first_char:
                result = result or existence_helper(
                    word[1:], board, row, col, set(["{}-{}".format(row, col)]))

    return result


board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
assert exists(board, "ABCCED")
assert exists(board, "SEE")
assert not exists(board, "ABCB")

```



Question 99

This problem was asked by Microsoft.

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

For example, given [100, 4, 200, 1, 3, 2], the longest consecutive element sequence is [1, 2, 3, 4]. Return its length: 4.

Your algorithm should run in O(n) complexity.

Answer 99





```

class NumRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __hash__(self):
        return hash(self.start, self.end)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        return "[{}, {}]".format(self.start, self.end)


def add_number(num, starts, ends):
    if num + 1 in starts and num - 1 in ends:
        num_range_1 = ends[num - 1]
        num_range_2 = starts[num + 1]
        num_range = NumRange(num_range_1.start, num_range_2.end)
        starts[num_range.start] = num_range
        ends[num_range.end] = num_range
        del starts[num_range_2.start]
        del ends[num_range_1.end]
        return
    elif num + 1 in starts:
        num_range = starts[num + 1]
        num_range.start = num
        starts[num] = num_range
        del starts[num + 1]
        return
    elif num - 1 in ends:
        num_range = ends[num - 1]
        num_range.end = num
        ends[num] = num_range
        del ends[num - 1]
        return

    num_range = NumRange(num, num)
    starts[num] = num_range
    ends[num] = num_range


def get_seq_length(arr):
    starts = dict()
    ends = dict()

    for num in arr:
        add_number(num, starts, ends)

    max_len = 0
    for start in starts:
        num_range = starts[start]
        length = num_range.end - num_range.start + 1
        max_len = length if length > max_len else max_len

    return max_len


assert get_seq_length([100, 4, 200, 1]) == 1
assert get_seq_length([100, 4, 200, 1, 3]) == 2
assert get_seq_length([100, 4, 200, 1, 3, 2]) == 4
assert get_seq_length([100, 4, 200, 1, 3, 2, 5]) == 5

```



Question 100

This problem was asked by Google.

You are in an infinite 2D grid where you can move in any of the 8 directions:

```
 (x,y) to
    (x+1, y),
    (x - 1, y),
    (x, y+1),
    (x, y-1),
    (x-1, y-1),
    (x+1,y+1),
    (x-1,y+1),
    (x+1,y-1)
```

You are given a sequence of points and the order in which you need to cover the points. Give the minimum number of steps in which you can achieve it. You start from the first point.

Example:
Input: `[(0, 0), (1, 1), (1, 2)]`
Output: 2
It takes 1 step to move from (0, 0) to (1, 1). It takes one more step to move from (1, 1) to (1, 2).

Answer 100





```

def get_distance(current_point, next_point, accumulated_distance):
    x_diff = next_point[0] - current_point[0]
    y_diff = next_point[1] - current_point[1]

    if not x_diff:
        return abs(y_diff) + accumulated_distance
    if not y_diff:
        return abs(x_diff) + accumulated_distance

    updated_current = (current_point[0] + int(x_diff/abs(x_diff)),
                       current_point[1] + int(y_diff/abs(y_diff)))

    return get_distance(updated_current, next_point, accumulated_distance + 1)


def get_min_steps_helper(current_point, remaining_points, steps):
    if not remaining_points:
        return steps

    next_point = remaining_points[0]
    min_distance = get_distance(current_point, next_point, 0)

    return get_min_steps_helper(next_point, remaining_points[1:], steps + min_distance)


def get_min_steps(points):
    if not points:
        return 0

    return get_min_steps_helper(points[0], points[1:], 0)


assert get_min_steps([]) == 0
assert get_min_steps([(0, 0)]) == 0
assert get_min_steps([(0, 0), (1, 1), (1, 2)]) == 2
assert get_min_steps([(0, 0), (1, 1), (1, 2), (3, 4)]) == 4
assert get_min_steps([(0, 0), (1, 1), (1, 2), (3, 6)]) == 6

```



Question 101

This problem was asked by Alibaba.

Given an even number (greater than 2), return two prime numbers whose sum will be equal to the given number.

A solution will always exist. See Goldbach’s conjecture.

Example:

Input: 4
Output: 2 + 2 = 4
If there are more than one solution possible, return the lexicographically smaller solution.

If `[a, b]` is one solution with `a <= b`, and `[c, d]` is another solution with `c <= d`, then

```python
[a, b] < [c, d]
if a < c or a==c and b < d.
```

Answer 101





```

def is_prime(num, primes):
    for prime in primes:
        if prime == num:
            return True
        if not num % prime:
            return False
    return True


def get_primes(num):
    limit = (num // 2) + 1

    candidates = list()
    primes = list()
    for i in range(2, limit):
        if is_prime(i, primes):
            primes.append(i)
            candidates.append((i, num - i))

    new_candidates = list()
    for first, second in candidates[::-1]:
        if is_prime(second, primes):
            primes.append(second)
            new_candidates.append((first, second))

    return new_candidates[-1]


assert get_primes(4) == (2, 2)
assert get_primes(10) == (3, 7)
assert get_primes(100) == (3, 97)

```



Question 102

This problem was asked by Lyft.

Given a list of integers and a number K, return which contiguous elements of the list sum to K.

For example, if the list is [1, 2, 3, 4, 5] and K is 9, then it should return [2, 3, 4].

Answer 102





```

def get_cont_arr(arr, target):
    summed = 0

    start, end = 0, 0
    i = 0
    while i < len(arr):
        if summed == target:
            return arr[start:end]
        elif summed > target:
            summed -= arr[start]
            start += 1
        else:
            summed += arr[i]
            end = i + 1
            i += 1


assert get_cont_arr([1, 2, 3, 4, 5], 0) == []
assert get_cont_arr([1, 2, 3, 4, 5], 1) == [1]
assert get_cont_arr([1, 2, 3, 4, 5], 5) == [2, 3]
assert get_cont_arr([5, 4, 3, 4, 5], 12) == [5, 4, 3]
assert get_cont_arr([5, 4, 3, 4, 5], 11) == [4, 3, 4]
assert get_cont_arr([1, 2, 3, 4, 5], 9) == [2, 3, 4]
assert get_cont_arr([1, 2, 3, 4, 5], 3) == [1, 2]

```



Question 103

This problem was asked by Square.

Given a string and a set of characters, return the shortest substring containing all the characters in the set.

For example, given the string "figehaeci" and the set of characters {a, e, i}, you should return "aeci".

If there is no substring containing all the characters in the set, return null.

Answer 103





```

from queue import Queue


def get_min_string(string, charset):
    curr_queue = list()
    ind_queue = list()
    curr_seen = set()

    candidate = None
    i = 0
    while i < len(string):
        if string[i] in charset:
            curr_queue.append(string[i])
            ind_queue.append(i)
            curr_seen.add(string[i])

        shift = 0
        for k in range(len(curr_queue)//2):
            if curr_queue[k] == curr_queue[-k-1]:
                shift += 1
        curr_queue = curr_queue[shift:]
        ind_queue = ind_queue[shift:]

        if len(curr_seen) == len(charset):
            if not candidate or len(candidate) > (ind_queue[-1] - ind_queue[0] + 1):
                candidate = string[ind_queue[0]:ind_queue[-1]+1]

        i += 1

    return candidate


assert not get_min_string("abcdedbc", {'g', 'f'})
assert get_min_string("abccbbbccbcb", {'a', 'b', 'c'}) == "abc"
assert get_min_string("figehaeci", {'a', 'e', 'i'}) == "aeci"
assert get_min_string("abcdedbc", {'d', 'b', 'b'}) == "db"
assert get_min_string("abcdedbc", {'b', 'c'}) == "bc"
assert get_min_string("abcdecdb", {'b', 'c'}) == "bc"
assert get_min_string("abcdecdb", {'b', 'c', 'e'}) == "bcde"

```



Question 104

This problem was asked by Google.

Determine whether a doubly linked list is a palindrome. What if it’s singly linked?

For example, `1 -> 4 -> 3 -> 4 -> 1` returns true while `1 -> 4` returns false.

Answer 104





```

# Solution works for both single and doubly linked lists


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __repr__(self):
        return "{} -> {}".format(self.val, self.next)


def reverse(head):
    if not head.next:
        new_head = Node(head.val)
        return new_head, new_head

    rev_head, rev_tail = reverse(head.next)
    rev_tail.next = Node(head.val)

    return rev_head, rev_tail.next


def is_palindrome(head):
    if not head:
        return None

    reversed_head = reverse(head)
    curr = head
    curr_rev = reversed_head[0]
    while curr:
        if curr.val != curr_rev.val:
            return False
        curr = curr.next
        curr_rev = curr_rev.next

    return True


# Tests

a0 = Node('a')
a1 = Node('a')
b0 = Node('b')
c0 = Node('c')

a0.next = b0
b0.next = c0

assert not is_palindrome(a0)
b0.next = a1
assert is_palindrome(a0)

a0 = Node('a')
assert is_palindrome(a0)
b0 = Node('b')
a0.next = b0
assert not is_palindrome(a0)

```



Question 105

This problem was asked by Facebook.

Given a function f, and N return a debounced f of N milliseconds.

That is, as long as the debounced f continues to be invoked, f itself will not be called for N milliseconds.

Answer 105





```

# source - https://gist.github.com/kylebebak/ee67befc156831b3bbaa88fb197487b0

import time


def debounce(s):
    """
    Decorator ensures function that can only be called once every `s` seconds.
    """
    interval = s * (10**(-3))

    def decorate(f):
        current_time = None

        def wrapped(*args, **kwargs):
            nonlocal current_time
            start_time = time.time()
            if current_time is None or start_time - current_time >= interval:
                result = f(*args, **kwargs)
                current_time = time.time()
                return result
        return wrapped
    return decorate


@debounce(3000)
def add_nums(x, y):
    return x + y


assert add_nums(1, 1) == 2
time.sleep(1)
assert not add_nums(1, 2)
time.sleep(1)
assert not add_nums(1, 3)
time.sleep(1)
assert add_nums(1, 4) == 5

```



Question 106

This problem was asked by Pinterest.

Given an integer list where each number represents the number of hops you can make, determine whether you can reach to the last index starting at index 0.

For example, `[2, 0, 1, 0]` returns `true` while `[1, 1, 0, 1]` returns `false`.

Answer 106





```

# assuming only integers > 0 are allowed


def reaches_last_helper(arr, start_index, target_index):
    if start_index == target_index:
        return True

    hop = arr[start_index]
    if not hop or (start_index + hop > target_index):
        return False

    return reaches_last_helper(arr, start_index + hop, target_index)


def reaches_last(arr):
    return reaches_last_helper(arr, 0, len(arr) - 1)


assert reaches_last([2, 0, 1, 0])
assert not reaches_last([1, 1, 0, 1])
assert not reaches_last([2, 1])

```



Question 107

This problem was asked by Microsoft.

Print the nodes in a binary tree level-wise. For example, the following should print `1, 2, 3, 4, 5`.

```
  1
 / \
2   3
   / \
  4   5
```

Answer 107





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def _print(self):
        print(self.val)
        if self.left:
            self.left._print()
        if self.right:
            self.right._print()


# Tests
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)

a.left = b
a.right = c
c.left = d
c.right = e

a._print()

```



Question 108

This problem was asked by Google.

Given two strings A and B, return whether or not A can be shifted some number of times to get B.

For example, if A is `abcde` and B is `cdeab`, return true. If A is `abc` and B is `acb`, return false.

Answer 108





```

def can_shift(target, string):
    return \
        target and string and \
        len(target) == len(string) and \
        string in target * 2


assert can_shift("abcde", "cdeab")
assert not can_shift("abc", "acb")

```



Question 109

This problem was asked by Cisco.

Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd bit should be swapped, the 3rd and 4th bit should be swapped, and so on.

For example, `10101010` should be `01010101`. `11100010` should be `11010001`.

Bonus: Can you do this in one line?

Answer 109





```

# 85 is the odd-bit filter '01010101'


def swap_bits(num):
    return ((num & 85) << 1) | ((num & (85 << 1)) >> 1)


assert swap_bits(0) == 0
assert swap_bits(255) == 255
assert swap_bits(210) == 225
assert swap_bits(170) == 85
assert swap_bits(226) == 209

```



Question 110

This problem was asked by Facebook.

Given a binary tree, return all paths from the root to leaves.

For example, given the tree

```
   1
  / \
 2   3
    / \
   4   5
```

it should return `[[1, 2], [1, 3, 4], [1, 3, 5]]`.

Answer 110





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.val)


def get_all_paths(node):
    if not node:
        return []

    node_paths = list()
    left_paths = get_all_paths(node.left)
    for path in left_paths:
        node_paths.append([node.val] + path)

    right_paths = get_all_paths(node.right)
    for path in right_paths:
        node_paths.append([node.val] + path)

    return node_paths if node_paths else [[node.val]]


# Tests

a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)

a.left = b
a.right = c
c.left = d
c.right = e

assert get_all_paths(a) == [[1, 2], [1, 3, 4], [1, 3, 5]]
assert get_all_paths(b) == [[2]]
assert get_all_paths(c) == [[3, 4], [3, 5]]

```



Question 111

This problem was asked by Google.

Given a word W and a string S, find all starting indices in S which are anagrams of W.

For example, given that W is "ab", and S is "abxaba", return 0, 3, and 4.

Answer 111





```

def add_needed_char(needed, char):
    if char not in needed:
        needed[char] = 0
    needed[char] += 1


def remove_gained_char(needed, char):
    needed[char] -= 1


def get_anagram_starts(string, word):
    if len(word) > len(string):
        return []

    anagram_indices = list()
    charset = set(word)

    needed = dict()
    for i, char_needed in enumerate(word):
        add_needed_char(needed, char_needed)

    for i in range(len(word)):
        char_gained = string[i]
        remove_gained_char(needed, char_gained)

    # This is a constant time operation because it has
    # a fixed upper bound of 26 read operations
    if all([x < 1 for x in needed.values()]):
        anagram_indices.append(0)

    for i in range(len(word), len(string)):
        window_start = i - len(word)
        char_removed = string[window_start]
        char_gained = string[i]

        if char_removed in charset:
            add_needed_char(needed, char_removed)

        if char_gained in needed:
            remove_gained_char(needed, char_gained)

        if all([x < 1 for x in needed.values()]):
            anagram_indices.append(window_start + 1)

    return anagram_indices


# Tests
assert get_anagram_starts("abxaba", "ab") == [0, 3, 4]
assert get_anagram_starts("cataract", "tac") == [0, 5]

```



Question 112

This problem was asked by Twitter.

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree. Assume that each node in the tree also has a pointer to its parent.

According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself)."

Answer 112





```

# The problem description is needlessly complicated.
# The problem is the same as finding the intersection of linked list


class Node:
    def __init__(self):
        self.parent = None


def get_list_length(node):
    if not node:
        return 0

    return 1 + get_list_length(node.parent)


def get_lca(node_a, node_b):
    len_a = get_list_length(node_a)
    len_b = get_list_length(node_b)

    (longer, max_len, shorter, min_len) = \
        (node_a, len_a, node_b, len_b) if len_a > len_b \
        else (node_b, len_b, node_a, len_a)

    for _ in range(max_len - min_len):
        longer = longer.parent

    while longer and shorter:
        if longer == shorter:
            return longer

        longer = longer.parent
        shorter = shorter.parent


# Test
def test_1():
    a = Node()
    b = Node()
    c = Node()
    d = Node()
    e = Node()
    f = Node()
    g = Node()

    a.parent = c
    c.parent = e
    e.parent = f

    b.parent = d
    d.parent = f

    f.parent = g

    assert get_lca(a, b) == f


def test_2():
    a = Node()
    b = Node()
    c = Node()
    d = Node()
    e = Node()
    f = Node()
    g = Node()

    a.parent = c
    c.parent = e

    b.parent = d
    d.parent = e

    e.parent = f
    f.parent = g

    assert get_lca(a, b) == e
    assert get_lca(c, b) == e
    assert get_lca(e, b) == e


test_1()
test_2()

```



Question 113

This problem was asked by Google.

Given a string of words delimited by spaces, reverse the words in string. For example, given "hello world here", return "here world hello"

Follow-up: given a mutable string representation, can you perform this operation in-place?

Answer 113





```

def reverse_words(string):
    return " ".join(reversed(string.split()))


assert reverse_words("hello world here") == "here world hello"

```



Question 114

This problem was asked by Facebook.

Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters. For example, given "hello/world:here", return "here/world:hello"

Follow-up: Does your solution work for the following cases: "hello/world:here/", "hello//world:here"

Answer 114





```

def reverse_words(string, delimiters):
    words = list()
    delims = list()
    delim_positions = list()  # stores positions of the delimiters seen

    start = 0
    i = 0
    while i < len(string):
        char = string[i]

        if char in delimiters:
            word = string[start:i]
            if i - start > 1:
                words.append(word)
            delims.append(char)
            delim_positions.append(len(words) + len(delims) - 1)
            start = i + 1

        i += 1

    # get last word if present
    if i - start > 1:
        words.append(string[start:i])

    words.reverse()  # reverse just the words

    reversed_order = list()
    word_index = 0
    delim_index = 0

    # merging the reversed words and the delimiters
    for i in range(len(words) + len(delims)):
        if delim_index < len(delim_positions) and delim_positions[delim_index] == i:
            # insert next delimiter if the position is saved for a delimiter
            reversed_order.append(delims[delim_index])
            delim_index += 1
        else:
            reversed_order.append(words[word_index])
            word_index += 1

    reversed_string = "".join(reversed_order)

    return reversed_string


assert reverse_words("hello/world:here/",
                     set([':', '/'])) == "here/world:hello/"
assert reverse_words(":hello//world:here/",
                     set([':', '/'])) == ":here//world:hello/"
assert reverse_words("hello//world:here",
                     set([':', '/'])) == "here//world:hello"
assert reverse_words("hello/world:here",
                     set([':', '/'])) == "here/world:hello"

```



Question 115

This problem was asked by Google.

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Answer 115





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.val)

    def is_exact_tree(self, other):
        if self.val != other.val:
            return False
        if bool(self.left) ^ bool(other.left):
            return False
        if bool(self.right) ^ bool(other.right):
            return False

        check = True
        if self.left:
            check = check and self.left.is_exact_tree(other.left)
        if self.right:
            check = check and self.right.is_exact_tree(other.right)

        return check

    def is_subtree(self, other):
        if self.is_exact_tree(other):
            return True

        if self.left and self.left.is_subtree(other):
            return True

        if self.right and self.right.is_subtree(other):
            return True

        return False


# Tests

s_0 = Node(0)
s_1 = Node(1)
s_2 = Node(2)
s_0.left = s_1
s_0.right = s_2
s_3 = Node(3)
s_4 = Node(4)
s_2.left = s_3
s_2.right = s_4

t_0 = Node(2)
t_1 = Node(3)
t_2 = Node(4)
t_0.left = t_1
t_0.right = t_2

r_0 = Node(2)
r_1 = Node(3)
r_2 = Node(5)
r_0.left = r_1
r_0.right = r_2

assert s_2.is_exact_tree(t_0)
assert s_0.is_subtree(t_0)
assert not s_0.is_subtree(r_0)

```



Question 116

This problem was asked by Jane Street.

Generate a finite, but an arbitrarily large binary tree quickly in `O(1)`.

That is, `generate()` should return a tree whose size is unbounded but finite.

Answer 116





```

from random import random, randint


class Node:
    def __init__(self, val):
        self.val = val
        self._left = None
        self._right = None

    def __repr__(self):
        string = "{"
        string += "{}: ".format(self.val)
        string += "{"
        string += "l: {}, ".format(self._left if self._left else -1)
        string += "r: {}".format(self._right if self._right else -1)
        string += "}"
        string += "}"

        return string

    @staticmethod
    def generate():
        return Node(randint(1, 100000))

    @property
    def left(self):
        if not self._left:
            self._left = Node.generate()
        return self._left

    @property
    def right(self):
        if not self._right:
            self._right = Node.generate()
        return self._right


tree_size = 100

a = Node.generate()
ref = a
nodes = set()
i = 0
while i < tree_size:
    ref = ref.left if random() < 0.5 else ref.right
    nodes.add(ref)

    i += 1

assert len(nodes) == tree_size
print(a)

```



Question 117

This problem was asked by Facebook.

Given a binary tree, return the level of the tree with minimum sum.

Answer 117





```

import sys


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "{}=(l={}, r={})".format(self.val, self.left, self.right)


def get_maxsum_level_helper(level, nodes, parent_sum):
    child_nodes = list()
    nodes_sum = 0
    for node in nodes:
        nodes_sum += node.val
        if node.left:
            child_nodes.append(node.left)
        if node.right:
            child_nodes.append(node.right)

    max_sum = max(nodes_sum, parent_sum)
    if child_nodes:
        max_sum = get_maxsum_level_helper(level + 1, child_nodes, max_sum)

    return max_sum


def get_maxsum_level(root):
    max_sum = get_maxsum_level_helper(0, [root], -sys.maxsize)
    return max_sum


a = Node(1)
b = Node(2)
c = Node(3)
a.left = b
a.right = c

d = Node(4)
e = Node(5)
b.left = d
b.right = e

f = Node(6)
g = Node(7)
c.left = f
c.right = g

h = Node(8)
d.right = h

assert get_maxsum_level(a) == 22
a.val = 100
assert get_maxsum_level(a) == 100
b.val = 150
assert get_maxsum_level(a) == 153
h.val = 200
assert get_maxsum_level(a) == 200

```



Question 118

This problem was asked by Google.

Given a sorted list of integers, square the elements and give the output in sorted order.

For example, given `[-9, -2, 0, 2, 3]`, return `[0, 4, 4, 9, 81]`.

Answer 118





```

def merge_sorted_lists(arr1, arr2):
    i, k = 0, 0
    merged = list()
    while i < len(arr1) and k < len(arr2):
        if arr1[i] <= arr2[k]:
            merged.append(arr1[i])
            i += 1
        else:
            merged.append(arr2[k])
            k += 1

    merged += arr1[i:]
    merged += arr2[k:]

    return merged


def sort_squares(arr):
    first_pos_index = 0
    for num in arr:
        if num >= 0:
            break
        first_pos_index += 1

    neg_nums = [x ** 2 for x in reversed(arr[:first_pos_index])]
    pos_nums = [x ** 2 for x in arr[first_pos_index:]]

    return merge_sorted_lists(pos_nums, neg_nums)


assert sort_squares([]) == []
assert sort_squares([0]) == [0]
assert sort_squares([-1, 1]) == [1, 1]
assert sort_squares([0, 2, 3]) == [0, 4, 9]
assert sort_squares([-9, -2, 0]) == [0, 4, 81]
assert sort_squares([-9, -2, 0, 2, 3]) == [0, 4, 4, 9, 81]

```



Question 119

This problem was asked by Google.

Given a set of closed intervals, find the smallest set of numbers that covers all the intervals. If there are multiple smallest sets, return any of them.

For example, given the intervals `[0, 3], [2, 6], [3, 4], [6, 9]`, one set of numbers that covers all these intervals is `{3, 6}`.

Answer 119





```

import sys


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return "[{}-{}]".format(self.start, self.end)


def get_smallest_stab_set_helper(remaining_intervals, used_points,
                                 remaining_points, num_to_intervals):

    if not remaining_intervals:
        return used_points

    min_len = sys.maxsize
    smallest_stab_set = None
    for current_point in remaining_points:
        if current_point not in num_to_intervals:
            continue

        current_point_intervals = num_to_intervals[current_point]
        new_rem_intervals = remaining_intervals - current_point_intervals

        new_used_points = used_points.copy()
        new_used_points.add(current_point)
        new_rem_points = remaining_points.copy()
        new_rem_points.remove(current_point)

        stab_set = get_smallest_stab_set_helper(
            new_rem_intervals, new_used_points, new_rem_points, num_to_intervals)
        if len(stab_set) < min_len:
            smallest_stab_set = stab_set
            min_len = len(stab_set)

    return smallest_stab_set


def get_smallest_stab_set(interval_tuples):
    intervals = set()
    num_to_intervals = dict()

    endpoints = set()
    for (start, end) in interval_tuples:
        endpoints.add(end)

        interval = Interval(start, end)
        intervals.add(interval)

        for num in range(start, end + 1):
            if num not in num_to_intervals:
                num_to_intervals[num] = set()
            num_to_intervals[num].add(interval)

    smallest_stab_set = get_smallest_stab_set_helper(
        intervals, set(), endpoints, num_to_intervals)

    return smallest_stab_set


assert get_smallest_stab_set([[0, 3]]) == {3}
assert get_smallest_stab_set([[0, 3], [2, 6]]) == {3}
assert get_smallest_stab_set([[0, 3], [2, 6], [3, 4]]) == {3}
assert get_smallest_stab_set([[0, 3], [2, 6], [3, 4], [6, 7]]) == {3, 6}
assert get_smallest_stab_set([[0, 3], [2, 6], [3, 4], [6, 9]]) == {3, 9}
assert get_smallest_stab_set([[0, 3], [2, 6], [3, 4], [6, 100]]) == {3, 100}

```



Question 120

This problem was asked by Microsoft.

Implement the singleton pattern with a twist. First, instead of storing one instance, store two instances. And in every even call of getInstance(), return the first instance and in every odd call of getInstance(), return the second instance.

Answer 120





```

class SampleClass:
    instances = dict()
    even_instance = False

    def __init__(self, instance_num):
        self.instance_num = instance_num

    @staticmethod
    def initialize():
        SampleClass.instances[0] = SampleClass(0)
        SampleClass.instances[1] = SampleClass(1)

    @staticmethod
    def get_instance():
        if not SampleClass.instances:
            SampleClass.initialize()

        SampleClass.even_instance = not SampleClass.even_instance
        return SampleClass.instances[int(SampleClass.even_instance)]


# Tests


SampleClass.initialize()

i1 = SampleClass.get_instance()
assert i1.instance_num == 1
i2 = SampleClass.get_instance()
assert i2.instance_num == 0
i3 = SampleClass.get_instance()
assert i3.instance_num == 1
i4 = SampleClass.get_instance()
assert i4.instance_num == 0
i5 = SampleClass.get_instance()
assert i5.instance_num == 1

```



Question 121

This problem was asked by Google.

Given a string which we can delete at most k, return whether you can make a palindrome.

For example, given 'waterrfetawx' and a k of 2, you could delete f and x to get 'waterretaw'.

Answer 121





```

def is_palindrome(string):
    return bool(string) and string == string[::-1]


def make_palindrome(string, num_delete):
    if is_palindrome(string):
        return True
    if not num_delete:
        return False

    for i, _ in enumerate(string):
        new_string = string[:i] + string[i+1:]
        if make_palindrome(new_string, num_delete - 1):
            return True

    return False


# Tests
assert make_palindrome("a", 0)
assert make_palindrome("aaa", 2)
assert not make_palindrome("add", 0)
assert make_palindrome("waterrfetawx", 2)
assert not make_palindrome("waterrfetawx", 1)
assert make_palindrome("waterrfetawx", 3)
assert make_palindrome("malayalam", 0)
assert make_palindrome("malayalam", 1)
assert make_palindrome("asdf", 5)
assert not make_palindrome("asdf", 2)

```



Question 122

This question was asked by Zillow.

You are given a 2-d matrix where each cell represents number of coins in that cell. Assuming we start at `matrix[0][0]`, and can only move right or down, find the maximum number of coins you can collect by the bottom right corner.

For example, in this matrix
```
0 3 1 1
2 0 0 4
1 5 3 1
```

The most we can collect is `0 + 2 + 1 + 5 + 3 + 1 = 12` coins.

Answer 122





```

def get_max_coins_helper(matrix, crow, ccol, rows, cols):
    cval = matrix[crow][ccol]

    if crow == rows - 1 and ccol == cols - 1:
        return cval

    down, right = cval, cval
    if crow < rows - 1:
        down += get_max_coins_helper(
            matrix, crow + 1, ccol, rows, cols)
    if ccol < cols - 1:
        right += get_max_coins_helper(
            matrix, crow, ccol + 1, rows, cols)

    return max(down, right)


def get_max_coins(matrix):
    if matrix:
        return get_max_coins_helper(
            matrix, 0, 0, len(matrix), len(matrix[0]))


coins = [[0, 3, 1, 1],
         [2, 0, 0, 4],
         [1, 5, 3, 1]]
assert get_max_coins(coins) == 12

coins = [[0, 3, 1, 1],
         [2, 8, 9, 4],
         [1, 5, 3, 1]]
assert get_max_coins(coins) == 25

```



Question 123

This problem was asked by LinkedIn.

Given a string, return whether it represents a number. Here are the different kinds of numbers:
* "10", a positive integer
* "-10", a negative integer
* "10.1", a positive real number
* "-10.1", a negative real number
* "1e5", a number in scientific notation

And here are examples of non-numbers:
* "a"
* "x 1"
* "a -2"
* "-"

Answer 123





```

def strip_neg(num):
    if num[0] != '-':
        return num
    elif len(num) > 1:
        return num[1:]


def is_valid_number_helper(string, dec):
    if not string:
        return True

    char = string[0]
    if (dec and char == '.') or (not char.isdigit() and char != '.'):
        return False
    elif char == '.':
        dec = True

    return is_valid_number_helper(string[1:], dec)


def is_valid_number(string):
    if not string:
        return False

    split_list = string.split('e')
    if len(split_list) > 2:
        return False

    if len(split_list) == 2:
        string_1 = strip_neg(split_list[0])
        string_2 = strip_neg(split_list[1])
        return string_1 and string_2 and \
            is_valid_number_helper(string_1, False) and \
            is_valid_number_helper(string_2, False)
    else:
        string = strip_neg(split_list[0])
        return string and is_valid_number_helper(string, False)


# Tests
assert is_valid_number("10")
assert is_valid_number("-10")
assert is_valid_number("10.1")
assert is_valid_number("-10.1")
assert is_valid_number("1e5")
assert is_valid_number("-5")
assert is_valid_number("1e-5")
assert is_valid_number("1e-5.2")
assert not is_valid_number("a")
assert not is_valid_number("x 1")
assert not is_valid_number("a -2")
assert not is_valid_number("-")

```



Question 124

This problem was asked by Microsoft.

You have 100 fair coins and you flip them all at the same time. Any that come up tails you set aside. The ones that come up heads you flip again. How many rounds do you expect to play before only one coin remains?

Write a function that, given $n$, returns the number of rounds you'd expect to play until one coin remains.

Answer 124





```

from math import log, ceil


def get_num_expected(coin_tosses):
    return ceil(log(coin_tosses, 2))


assert get_num_expected(1) == 0
assert get_num_expected(2) == 1
assert get_num_expected(100) == 7
assert get_num_expected(200) == 8

```



Question 125

This problem was asked by Google.

Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

For example, given the following tree and K of 20

```
    10
   /   \
 5      15
       /  \
     11    15
```

Return the nodes 5 and 15.

Answer 125





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.val)


def get_array(root):
    arr = [root]

    if not root.left and not root.right:
        return arr

    if root.left:
        arr = get_array(root.left) + arr
    if root.right:
        arr = arr + get_array(root.right)

    return arr


def search_pair(root, val):
    arr = get_array(root)
    i, k = 0, len(arr) - 1
    while i < k:
        summed = arr[i].val + arr[k].val
        if summed == val:
            return (arr[i], arr[k])
        elif summed < val:
            i += 1
        else:
            k -= 1


# Tests
a = Node(10)
b = Node(5)
c = Node(15)
d = Node(11)
e = Node(15)
a.left = b
a.right = c
c.left = d
c.right = e

assert search_pair(a, 15) == (b, a)
assert search_pair(a, 20) == (b, e)
assert search_pair(a, 30) == (c, e)
assert search_pair(a, 26) == (d, e)

```



Question 126

This problem was asked by Facebook.

Write a function that rotates a list by k elements. For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes `[3, 4, 5, 6, 1, 2]`. Try solving this without creating a copy of the list. How many swap or move operations do you need?

Answer 126





```

# Solution 1
def rotate_list_once(arr):
    first = arr[0]
    for i in range(len(arr) - 1):
        arr[i] = arr[i + 1]
    arr[-1] = first
    return arr


def rotate_list(arr, k):
    for _ in range(k):
        arr = rotate_list_once(arr)

    return arr


# Solution 2
def rotate_list_alt(arr, k):
    return arr[k:] + arr[:k]


# Tests
assert rotate_list([1, 2, 3, 4, 5, 6], 0) == [1, 2, 3, 4, 5, 6]
assert rotate_list_alt([1, 2, 3, 4, 5, 6], 0) == [1, 2, 3, 4, 5, 6]

assert rotate_list([1, 2, 3, 4, 5, 6], 2) == [3, 4, 5, 6, 1, 2]
assert rotate_list_alt([1, 2, 3, 4, 5, 6], 2) == [3, 4, 5, 6, 1, 2]

assert rotate_list([1, 2, 3, 4, 5, 6], 4) == [5, 6, 1, 2, 3, 4]
assert rotate_list_alt([1, 2, 3, 4, 5, 6], 4) == [5, 6, 1, 2, 3, 4]

```



Question 127

This problem was asked by Microsoft.

Let's represent an integer in a linked list format by having each node represent a digit in the number. The nodes make up the number in reversed order.

For example, the following linked list:

`1 -> 2 -> 3 -> 4 -> 5`
is the number `54321`.

Given two linked lists in this format, return their sum in the same linked list format.

For example, given

`9 -> 9`
`5 -> 2`
return `124 (99 + 25)` as:

`4 -> 2 -> 1`

Answer 127





```

FACTOR = 10


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __repr__(self):
        return "{} -> {}".format(self.val, self.next)

    @staticmethod
    def convert_num_to_list(num):
        numstr = str(num)
        dummy_head = Node(0)
        prev = dummy_head
        for num in numstr[::-1]:
            curr = Node(int(num))
            prev.next = curr
            prev = curr

        return dummy_head.next

    @staticmethod
    def convert_list_to_num(head):
        curr = head
        num = 0
        factor = 1
        while curr:
            num += factor * curr.val
            curr = curr.next
            factor *= FACTOR

        return num

    @staticmethod
    def add_nums(list_a, list_b):
        new_dummy_head = Node(0)
        prev_res = new_dummy_head

        curr_a, curr_b = list_a, list_b
        carry = 0

        while carry or curr_a or curr_b:
            res_digit = 0
            if carry:
                res_digit += carry
            if curr_a:
                res_digit += curr_a.val
                curr_a = curr_a.next
            if curr_b:
                res_digit += curr_b.val
                curr_b = curr_b.next

            carry = res_digit // FACTOR
            curr_res = Node(res_digit % FACTOR)

            prev_res.next = curr_res
            prev_res = curr_res

        return new_dummy_head.next


assert Node.convert_list_to_num(
    Node.add_nums(Node.convert_num_to_list(0),
                  Node.convert_num_to_list(1))) == 1
assert Node.convert_list_to_num(
    Node.add_nums(Node.convert_num_to_list(1000),
                  Node.convert_num_to_list(1))) == 1001
assert Node.convert_list_to_num(
    Node.add_nums(Node.convert_num_to_list(99),
                  Node.convert_num_to_list(25))) == 124

```



Question 128

The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.

All the disks start off on the first rod in a stack. They are ordered by size, with the largest disk on the bottom and the smallest one at the top.

The goal of this puzzle is to move all the disks from the first rod to the last rod while following these rules:
* You can only move one disk at a time.
* A move consists of taking the uppermost disk from one of the stacks and placing it on top of another stack.
* You cannot place a larger disk on top of a smaller disk.

Write a function that prints out all the steps necessary to complete the Tower of Hanoi. You should assume that the rods are numbered, with the first rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod being 3.

For example, with n = 3, we can do this in 7 moves:

* Move 1 to 3
* Move 1 to 2
* Move 3 to 2
* Move 1 to 3
* Move 2 to 1
* Move 2 to 3
* Move 1 to 3

Answer 128





```

def toh_helper(disks, source, inter, target):
    if disks == 1:
        target.append(source.pop())
    else:
        toh_helper(disks - 1, source, target, inter)
        print(source, inter, target)
        target.append(source.pop())
        print(source, inter, target)
        toh_helper(disks - 1, inter, source, target)

    print(source, inter, target)


def towers_of_hanoi(n):
    if not n:
        return

    print("{} Towers of Hanoi".format(n))
    stack_1, stack_2, stack_3 = \
        list(range(1, n + 1))[::-1], list(), list()
    toh_helper(n, stack_1, stack_2, stack_3)


towers_of_hanoi(3)
towers_of_hanoi(4)
towers_of_hanoi(5)
towers_of_hanoi(6)

```



Question 129

Given a real number n, find the square root of n. For example, given n = 9, return 3.

Answer 129





```

TOLERANCE = 10 ** -6


def almost_equal(first, second):
    # check equality with some acceptable error tolerance
    return \
        second > first - TOLERANCE and \
        second < first + TOLERANCE


def get_sqrt_helper(n, start, end):
    mid = start + ((end - start) / 2)

    if almost_equal(mid * mid, n):
        return mid
    elif mid * mid > n:
        return get_sqrt_helper(n, start, mid)
    else:
        return get_sqrt_helper(n, mid, end)


def get_sqrt(n):
    return get_sqrt_helper(n, 0, n)


assert almost_equal(get_sqrt(9), 3)
assert almost_equal(get_sqrt(2), 1.41421356237)
assert almost_equal(get_sqrt(10000), 100)

```



Question 130

This problem was asked by Facebook.

Given an array of numbers representing the stock prices of a company in chronological order and an integer k, return the maximum profit you can make from k buys and sells. You must buy the stock before you can sell it, and you must sell the stock before you can buy it again.

For example, given `k = 2` and the array `[5, 2, 4, 0, 1]`, you should return `3`.

Answer 130





```

def profit_helper(prices, curr_index, curr_profit, buys_left, sells_left):

    if curr_index == len(prices) or not sells_left:
        # reached end of chronology or exhausted trades
        return curr_profit

    if buys_left == sells_left:
        # buy or wait
        return max(
            # buy
            profit_helper(prices, curr_index + 1, curr_profit - prices[curr_index],
                          buys_left - 1, sells_left),
            # wait
            profit_helper(prices, curr_index + 1, curr_profit,
                          buys_left, sells_left)
        )
    else:
        # sell or hold
        return max(
            # sell
            profit_helper(prices, curr_index + 1, curr_profit + prices[curr_index],
                          buys_left, sells_left - 1),
            # hold
            profit_helper(prices, curr_index + 1, curr_profit,
                          buys_left, sells_left)
        )


def get_max_profit(prices, k):
    return profit_helper(prices, 0, 0, k, k)


assert get_max_profit([5, 2, 4, 0, 1], 2) == 3
assert get_max_profit([5, 2, 4], 2) == 2
assert get_max_profit([5, 2, 4], 1) == 2

```



Question 131

This question was asked by Snapchat.

Given the head to a singly linked list, where each node also has a 'random' pointer that points to anywhere in the linked list, deep clone the list.

Answer 131





```

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.rand = None

    def __repr__(self):
        return self.val


def deep_clone(source):
    node_to_index = dict()
    node_index_to_rand_index = dict()

    curr = source
    i = 0
    while curr:
        node_to_index[curr] = i
        curr = curr.next
        i += 1

    curr = source
    i = 0
    while curr:
        rand_index = node_to_index[curr.rand]
        node_index_to_rand_index[i] = rand_index
        curr = curr.next
        i += 1

    dummy_head = Node('0')
    tail = dummy_head
    curr = source
    index_to_node = dict()
    i = 0
    while curr:
        new_node = Node(curr.val)
        index_to_node[i] = new_node
        curr = curr.next
        i += 1
        tail.next = new_node
        tail = new_node

    curr = dummy_head.next
    i = 0
    while curr:
        rand_index = node_index_to_rand_index[i]
        curr.rand = index_to_node[rand_index]
        curr = curr.next
        i += 1

    return dummy_head.next


# Tests

a = Node('a')
b = Node('b')
a.next = b
c = Node('c')
b.next = c
d = Node('d')
c.next = d
e = Node('e')
d.next = e

a.rand = a
b.rand = a
c.rand = e
d.rand = c
e.rand = c

cloned = deep_clone(a)
assert cloned.val == 'a'
assert cloned.rand.val == 'a'
assert cloned.next.val == 'b'
assert cloned.next.rand.val == 'a'

```



Question 132

This question was asked by Riot Games.

Design and implement a HitCounter class that keeps track of requests (or hits). It should support the following operations:
* `record(timestamp)`: records a hit that happened at timestamp
* `total()`: returns the total number of hits recorded
* `range(lower, upper)`: returns the number of hits that occurred between timestamps lower and upper (inclusive)

Follow-up: What if our system has limited memory?

Answer 132





```

from time import time
from random import random, randrange
import bisect

REQUESTS_PER_FILE = 100


class PersistedFile:
    # in a memory efficient implementation, this would be persisted to disk
    # once full and only the path and metadata would remain in memory
    def __init__(self):
        self.start_timestamp = None
        self.end_timestamp = None
        self.request_timestamps = list()

    def __repr__(self):
        return "start={}, end={}, size={}".format(
            self.start_timestamp, self.end_timestamp, len(self.request_timestamps))


class RequestQuery:
    def __init__(self):
        self.current_file = PersistedFile()
        self.prev_files = list()

    def record(self, timestamp):
        if not self.current_file.start_timestamp:
            self.current_file.start_timestamp = timestamp
        self.current_file.request_timestamps.append(timestamp)
        self.current_file.end_timestamp = timestamp

        if len(self.current_file.request_timestamps) == REQUESTS_PER_FILE:
            self.prev_files.append(self.current_file)
            self.current_file = PersistedFile()

    def total(self):
        return (len(self.prev_files) * REQUESTS_PER_FILE) + \
            len(self.current_file.request_timestamps)

    def range(self, lower, upper):
        all_files = self.prev_files + [self.current_file]
        start_times = [x.start_timestamp for x in all_files]
        end_times = [x.end_timestamp for x in all_files]

        start_file_index = bisect.bisect_left(start_times, lower) - 1
        end_file_index = bisect.bisect_left(end_times, upper)
        start_file = all_files[start_file_index]
        end_file = all_files[end_file_index]

        start_file_pos = bisect.bisect(start_file.request_timestamps, lower)
        end_file_pos = bisect.bisect(end_file.request_timestamps, upper)

        num_req = 0
        num_req += len(start_file.request_timestamps[start_file_pos:])
        num_req += len(end_file.request_timestamps[:end_file_pos])
        num_req += (end_file_index - start_file_index) * REQUESTS_PER_FILE

        return num_req


def run_experiments(requests):
    rq = RequestQuery()
    lower, upper = None, None

    for i in range(requests):
        rq.record(i)

        if random() < 0.001:
            if not lower:
                lower = i
            else:
                upper = randrange(lower, i)

            if lower and upper:
                num_req = rq.range(lower, upper)
                print("{} requests made between {} and {}".format(
                    num_req, lower, upper))
                print("Total: {}".format(rq.total()))
                lower, upper = None, None


run_experiments(112367)

```



Question 133

This problem was asked by Amazon.

Given a node in a binary tree, return the next bigger element, also known as the inorder successor.
(NOTE: I'm assuming this is a binary search tree, because otherwise, the problem makes no sense at all)

For example, the inorder successor of 22 is 30.

```
   10
  /  \
 5    30
     /  \
   22    35
```
You can assume each node has a parent pointer.

Answer 133





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return str(self.val)


def inorder_helper(root):
    list_rep = [root]
    if root.left:
        list_rep = inorder_helper(root.left) + list_rep
    if root.right:
        list_rep = list_rep + inorder_helper(root.right)

    return list_rep


def find_next_inorder(target_node):
    root = target_node
    while root.parent:
        root = root.parent
    all_nodes = inorder_helper(root)
    for i, node in enumerate(all_nodes):
        if node == target_node:
            if i == len(all_nodes) - 1:
                return None
            return all_nodes[i + 1]


# Tests

root = Node(10)
root.left = Node(5)
root.right = Node(30)
root.left.parent = root
root.right.parent = root
root.right.left = Node(22)
root.right.right = Node(35)
root.right.left.parent = root.right
root.right.right.parent = root.right

assert not find_next_inorder(root.right.right)
assert find_next_inorder(root.right.left) == root.right
assert find_next_inorder(root) == root.right.left
assert find_next_inorder(root.left) == root

```



Question 134

This problem was asked by Facebook.

You have a large array with most of the elements as zero.

Use a more space-efficient data structure, SparseArray, that implements the same interface:
* `init(arr, size)`: initialize with the original large array and size.
* `set(i, val)`: updates index at i with val.
* `get(i)`: gets the value at index i.

Answer 134





```

class SparseArray:

    def __init__(self):
        self.storage = dict()

    def __repr__(self):
        return str(self.storage)

    def init(self, arr, size):
        for i, num in enumerate(arr):
            if num:
                self.storage[i] = num

    def set_val(self, i, val):
        if not val:
            del self.storage[i]
        else:
            self.storage[i] = val

    def get_val(self, i):
        if i not in self.storage:
            return None

        return self.storage[i]


# Tests

arr = [1, 0, 0, 0, 3, 0, 2]

sa = SparseArray()
sa.init(arr, len(arr))
assert sa.storage == {0: 1, 4: 3, 6: 2}

sa.set_val(2, 4)
assert sa.get_val(2) == 4
sa.set_val(4, 1)
assert sa.get_val(4) == 1
sa.set_val(0, 0)
assert not sa.get_val(0)

```



Question 135

This question was asked by Apple.

Given a binary tree, find a minimum path sum from root to a leaf.

For example, the minimum path in this tree is `[10, 5, 1, -1]`, which has sum 15.

```
  10
 /  \
5    5
 \     \
   2    1
       /
     -1
```

Answer 135





```

import sys


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.val)


def min_path_helper(root, path, path_sum):

    candidate_sum = path_sum + root.val
    candidate_path = path + [root.val]

    if not root.left and not root.right:
        return (candidate_sum, candidate_path)

    left_min_val, right_min_val = sys.maxsize, sys.maxsize
    if root.left:
        left_min_val, left_path = min_path_helper(
            root.left, candidate_path, candidate_sum)
    if root.right:
        right_min_val, right_path = min_path_helper(
            root.right, candidate_path, candidate_sum)

    return (left_min_val, left_path) if left_min_val < right_min_val \
        else (right_min_val, right_path)


def find_min_path(root):
    _, min_path = min_path_helper(root, list(), 0)
    return min_path


# Tests

a = Node(10)
b = Node(5)
c = Node(5)
a.left = b
a.right = c
d = Node(2)
b.right = d
e = Node(1)
c.right = e
f = Node(-1)
e.left = f
assert find_min_path(a) == [10, 5, 1, -1]

f.val = 5
assert find_min_path(a) == [10, 5, 2]

```



Question 136

This question was asked by Google.

Given an N by M matrix consisting only of 1's and 0's, find the largest rectangle containing only 1's and return its area.

For example, given the following matrix:

```
[[1, 0, 0, 0],
 [1, 0, 1, 1],
 [1, 0, 1, 1],
 [0, 1, 0, 0]]
```

Return 4.

Answer 136





```

def extendable_row(matrix, erow, scol, ecol):
    return all(matrix[erow][scol:ecol])


def extendable_col(matrix, ecol, srow, erow):
    for row in range(srow, erow):
        if not matrix[row][ecol]:
            return False

    return True


def area_helper(matrix, num_rows, num_cols, srow, erow, scol, ecol):
    current_area = (erow - srow) * (ecol - scol)
    row_ex_area, col_ex_area = 0, 0

    ex_row = erow < num_rows and extendable_row(matrix, erow, scol, ecol)
    if ex_row:
        row_ex_area = area_helper(matrix, num_rows, num_cols,
                                  srow, erow + 1, scol, ecol)

    ex_col = ecol < num_cols and extendable_col(matrix, ecol, srow, erow)
    if ex_col:
        col_ex_area = area_helper(matrix, num_rows, num_cols,
                                  srow, erow, scol, ecol + 1)

    return max(current_area, row_ex_area, col_ex_area)


def get_largest_area(matrix):
    max_area = 0
    if not matrix:
        return max_area

    num_rows, num_cols = len(matrix), len(matrix[0])

    for i in range(num_rows):
        for j in range(num_cols):
            upper_bound_area = (num_rows - i) * (num_cols - j)
            if matrix[i][j] and upper_bound_area > max_area:
                area = area_helper(
                    matrix, num_rows, num_cols, i, i + 1, j, j + 1)
                max_area = area if area > max_area else max_area

    return max_area


# Tests

matrix = [[1, 0, 0, 0],
          [1, 0, 1, 1],
          [1, 0, 1, 1],
          [0, 1, 0, 0]]
assert get_largest_area(matrix) == 4


matrix = [[1, 0, 0, 0],
          [1, 0, 1, 1],
          [1, 0, 1, 1],
          [0, 1, 1, 1]]
assert get_largest_area(matrix) == 6

matrix = [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]
assert get_largest_area(matrix) == 16

matrix = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]
assert get_largest_area(matrix) == 0

matrix = [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 0, 0],
          [0, 0, 0, 0]]
assert get_largest_area(matrix) == 8

matrix = [[1, 1, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 0, 0]]
assert get_largest_area(matrix) == 4

```



Question 137

This problem was asked by Amazon.

Implement a bit array.

A bit array is a space efficient array that holds a value of 1 or 0 at each index.
* init(size): initialize the array with size
* set(i, val): updates index at i with val where val is either 1 or 0.
* get(i): gets the value at index i.

Answer 137





```

class BitArray:
    def __init__(self, size):
        self.set_indices = set()
        self.size = size

    def get_list_rep(self):
        arr = list()
        for i in range(self.size):
            if i in self.set_indices:
                arr.append(1)
            else:
                arr.append(0)
        return arr

    def set_val(self, i, val):
        if i >= self.size:
            raise LookupError("Invalid Index")

        if val and i not in self.set_indices:
            self.set_indices.add(i)
        elif not val and i in self.set_indices:
            self.set_indices.remove(i)

    def get_val(self, i):
        if i >= self.size:
            raise LookupError("Invalid Index")
        return int(i in self.set_indices)


# Tests
ba = BitArray(10)
assert ba.get_list_rep() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ba.set_val(3, 1)
assert ba.get_list_rep() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
ba.set_val(6, 1)
ba.set_val(8, 1)
ba.set_val(3, 0)
assert ba.get_list_rep() == [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]


# Exceptions
try:
    ba.set_val(10, 1)
except LookupError as le:
    print(le)
try:
    ba.get_val(10)
except LookupError as le:
    print(le)

```



Question 138

This problem was asked by Google.

Find the minimum number of coins required to make n cents.

You can use standard American denominations, that is, 1¢, 5¢, 10¢, and 25¢.

For example, given n = 16, return 3 since we can make it with a 10¢, a 5¢, and a 1¢.

Answer 138





```

def get_min_coins(amount, denoms):
    if amount == 0:
        return 0

    denom_to_use, index_cutoff = None, None
    for i, denom in enumerate(denoms):
        if amount >= denom:
            denom_to_use = denom
            index_cutoff = i
            break

    coins_used = amount // denom_to_use

    return coins_used + get_min_coins(amount - (denom_to_use * coins_used),
                                      denoms[index_cutoff + 1:])


# Tests
denoms = [25, 10, 5, 1]
assert get_min_coins(16, denoms) == 3
assert get_min_coins(90, denoms) == 5
assert get_min_coins(100, denoms) == 4

```



Question 139

This problem was asked by Google.

Given an iterator with methods next() and hasNext(), create a wrapper iterator, PeekableInterface, which also implements peek(). peek shows the next element that would be returned on next().

Here is the interface:

```
class PeekableInterface(object):
    def __init__(self, iterator):
        pass

    def peek(self):
        pass

    def next(self):
        pass

    def hasNext(self):
        pass
```

Answer 139





```

class PeekableInterface(object):
    def __init__(self, iterator):
        self.iterator = iterator
        self.next_val = next(iterator)
        self.has_next_val = True

    def peek(self):
        return self.next_val

    def next(self):
        next_val = self.next_val
        try:
            self.next_val = next(iterator)
        except StopIteration:
            self.has_next_val = False
            self.next_val = None
        return next_val

    def has_next(self):
        return self.has_next_val


# Tests

sample_list = [1, 2, 3, 4, 5]
iterator = iter(sample_list)
peekable = PeekableInterface(iterator)

assert peekable.peek() == 1
assert peekable.has_next()

assert peekable.next() == 1
assert peekable.next() == 2
assert peekable.next() == 3

assert peekable.peek() == 4
assert peekable.has_next()

assert peekable.next() == 4
assert peekable.has_next()
assert peekable.peek() == 5
assert peekable.next() == 5

assert not peekable.has_next()
assert not peekable.peek()

```



Question 140

This problem was asked by Facebook.

Given an array of integers in which two elements appear exactly once and all other elements appear exactly twice, find the two elements that appear only once.

For example, given the array `[2, 4, 6, 8, 10, 2, 6, 10]`, return 4 and 8. The order does not matter.

Follow-up: Can you do this in linear time and constant space?

Answer 140





```

def get_singles(arr):
    xored = arr[0]
    for num in arr[1:]:
        xored ^= num
    x, y = 0, 0

    rightmost_set_bit = (xored & ~(xored - 1))
    for num in arr:
        if num & rightmost_set_bit:
            x ^= num
        else:
            y ^= num

    return (x, y)


# Tests

get_singles([2, 4, 6, 8, 10, 2, 6, 10]) == (4, 8)
get_singles([2, 4, 8, 8, 10, 2, 6, 10]) == (4, 6)

```



Question 141

This problem was asked by Microsoft.

Implement 3 stacks using a single list:

```
class Stack:
    def __init__(self):
        self.list = []

    def pop(self, stack_number):
        pass

    def push(self, item, stack_number):
        pass
```

Answer 141





```

class Stack:
    def __init__(self):
        self.list = []

    def __repr__(self):
        return str(self.list)

    def pop(self, stack_number):
        if not stack_number < len(self.list):
            return

        stack = self.list[stack_number]
        if not stack:
            return

        val = stack.pop()
        return val

    def push(self, item, stack_number):
        if stack_number < len(self.list):
            stack = self.list[stack_number]
            stack.append(item)
        elif stack_number == len(self.list):
            new_stack = list()
            new_stack.append(item)
            self.list.append(new_stack)


# Tests
st = Stack()

st.push(1, 0)
assert st.list == [[1]]
st.push(2, 1)
st.push(3, 2)
assert st.list == [[1], [2], [3]]
val = st.pop(3)
assert not val
val = st.pop(2)
assert val == 3
assert st.list == [[1], [2], []]
st.push(6, 0)
st.push(7, 0)
assert st.list == [[1, 6, 7], [2], []]
val = st.pop(0)
assert st.list == [[1, 6], [2], []]

```



Question 142

This problem was asked by Google.

You're given a string consisting solely of `(`, `)`, and `*`. 
`*` can represent either a `(`, `)`, or an empty string. Determine whether the parentheses are balanced.

For example, `(()*` and `(*)` are balanced. `)*(` is not balanced.

Answer 142





```

def valid_paren(string, stack=list()):

    if not string and not stack:
        return True
    elif not string:
        return False

    cchar = string[0]
    remaining = string[1:]
    if cchar == '*':
        return \
            valid_paren('(' + remaining, stack) or \
            valid_paren(')' + remaining, stack) or \
            valid_paren(remaining, stack)

    cstack = stack.copy()
    if cchar == ')' and not stack:
        return False
    elif cchar == ')':
        cstack.pop()
    else:
        cstack.append(cchar)

    return valid_paren(remaining, cstack)


# Tests
assert valid_paren("(()*")
assert valid_paren("(*)")
assert not valid_paren(")*(")

```



Question 143

This problem was asked by Amazon.

Given a pivot `x`, and a list `lst`, partition the list into three parts.
* The first part contains all elements in `lst` that are less than `x`
* The second part contains all elements in `lst` that are equal to `x`
* The third part contains all elements in `lst` that are larger than `x`
Ordering within a part can be arbitrary.

For example, given `x = 10` and `lst = [9, 12, 3, 5, 14, 10, 10]`, one partition may be `[9, 3, 5, 10, 10, 12, 14]`

Answer 143





```

def swap_indices(arr, i, k):
    tmp = arr[i]
    arr[i] = arr[k]
    arr[k] = tmp


def separate_with_pivot(arr, i, k, x):
    if not arr:
        return

    while i < k:
        if arr[i] >= x and arr[k] < x:
            swap_indices(arr, i, k)
        else:
            if arr[i] < x:
                i += 1
            if arr[k] >= x:
                k -= 1

    return i + 1 if (arr[i] < x and i + 1 < len(arr)) else i


def pivot_list(arr, x):
    mid = separate_with_pivot(arr, 0, len(arr) - 1, x)
    separate_with_pivot(arr, mid, len(arr) - 1, x + 1)

    return arr



# Tests
assert pivot_list([9, 12, 3, 5, 14, 10, 10], 10) == [9, 5, 3, 10, 10, 14, 12]
assert pivot_list([9, 12, 3, 5, 14, 10, 10], 8) == [5, 3, 12, 9, 14, 10, 10]
assert pivot_list([9, 12, 14, 10, 10], 8) == [9, 12, 14, 10, 10]
assert pivot_list([3, 5], 8) == [3, 5]
assert pivot_list([8, 8, 8], 8) == [8, 8, 8]
assert pivot_list([], 8) == []

```



Question 144

This problem was asked by Google.

Given an array of numbers and an index `i`, return the index of the nearest larger number of the number at index `i`, where distance is measured in array indices.

For example, given `[4, 1, 3, 5, 6]` and index `0`, you should return `3`.

If two distances to larger numbers are equal, then return any one of them. If the array at `i` doesn't have a nearest larger integer, then return `null`.

Follow-up: If you can preprocess the array, can you do this in constant time?

Answer 144





```

def get_mapping_indices(arr):
    nl_indices = dict()
    sorted_tuples = [(x, i) for i, x in enumerate(arr)]
    sorted_tuples.sort(key=lambda x: x[0])

    for k, (_, i) in enumerate(sorted_tuples[:-1]):
        min_dist = len(arr)
        for m in range(k + 1, len(sorted_tuples)):
            dist = abs(i - sorted_tuples[m][1])
            if dist < min_dist:
                min_dist = dist
                nl_indices[i] = sorted_tuples[m][1]

    return nl_indices


def nearest_larger(arr, index):
    nl_indices = get_mapping_indices(arr)

    if index not in nl_indices:
        return None

    return nl_indices[index]


# Tests
assert nearest_larger([4, 1, 3, 5, 6], 0) == 3
assert nearest_larger([4, 1, 3, 5, 6], 1) == 0 or \
    nearest_larger([4, 1, 3, 5, 6], 1) == 2
assert not nearest_larger([4, 1, 3, 5, 6], 4)
assert nearest_larger([4, 1, 3, 5, 6], 3) == 4

```



Question 145

This problem was asked by Google.

Given the head of a singly linked list, swap every two nodes and return its head.

For example, given `1 -> 2 -> 3 -> 4`, return `2 -> 1 -> 4 -> 3`.

Answer 145





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def swap_two_helper(node):
    if not node or not node.next:
        return node

    first = node
    second = node.next
    tail = swap_two_helper(second.next)

    second.next = first
    first.next = tail

    return second


def swap_two(head):
    dummy_head = Node(0)
    dummy_head.next = head

    return swap_two_helper(head)



# Tests
assert get_list(swap_two(get_nodes([1]))) == [1]
assert get_list(swap_two(get_nodes([1, 2]))) == [2, 1]
assert get_list(swap_two(get_nodes([1, 2, 3, 4]))) == [2, 1, 4, 3]
assert get_list(swap_two(get_nodes([1, 2, 3, 4, 5]))) == [2, 1, 4, 3, 5]
assert get_list(swap_two(get_nodes([1, 2, 3, 4, 5, 6]))) == [2, 1, 4, 3, 6, 5]

```



Question 146

This question was asked by BufferBox.

Given a binary tree where all nodes are either 0 or 1, prune the tree so that subtrees containing all 0s are removed.

For example, given the following tree:
```
   0
  / \
 1   0
    / \
   1   0
  / \
 0   0
```

should be pruned to:
```
   0
  / \
 1   0
    /
   1
```

We do not remove the tree at the root or its left child because it still has a 1 as a descendant.

Answer 146





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        string = "{}=[l={}, r={}]".format(self.val, self.left, self.right)
        return string

    def prune(self):
        if not self.left and not self.right and not self.val:
            return None

        if self.left:
            self.left = self.left.prune()
        if self.right:
            self.right = self.right.prune()

        return self


# Tests
root = Node(0)
root.left = Node(1)
root.right = Node(0)
root.right.left = Node(1)
root.right.right = Node(0)
root.right.left.left = Node(0)
root.right.left.right = Node(0)

assert root.right.right
assert root.right.left.left
assert root.right.left.right
root.prune()

assert not root.right.right
assert not root.right.left.left
assert not root.right.left.right

assert root.left
assert root.right

```



Question 147

Given a list, sort it using this method: `reverse(lst, i, j)`, which sorts `lst` from `i` to `j`.

Answer 147





```

def merge_sorted(arr_1, arr_2):
    merged_array = list()
    ind_1, ind_2 = 0, 0
    while ind_1 < len(arr_1) and ind_2 < len(arr_2):
        if arr_1[ind_1] <= arr_2[ind_2]:
            merged_array.append(arr_1[ind_1])
            ind_1 += 1
        else:
            merged_array.append(arr_2[ind_2])
            ind_2 += 1

    while ind_1 < len(arr_1):
        merged_array.append(arr_1[ind_1])
        ind_1 += 1
    while ind_2 < len(arr_2):
        merged_array.append(arr_2[ind_2])
        ind_2 += 1

    return merged_array


def reverse(lst, i, j):
    return list(reversed(lst[i:j+1]))


def custom_sort(lst):
    # create segments of sorted sub-arrays
    start, end = None, None
    last_end = -1
    sorted_segments = list()
    for i in range(1, len(lst)):
        if lst[i] < lst[i-1]:
            if not start:
                segment = lst[last_end+1: i-1]
                if segment:
                    sorted_segments.append(segment)
                start = i - 1
        elif start:
            end = i - 1
            if end > start:
                sorted_segments.append(reverse(lst, start, end))
                last_end = end
                start, end = None, None
    if start:
        end = len(lst) - 1
        if end > start:
            sorted_segments.append(reverse(lst, start, end))
    else:
        segment = lst[last_end+1:]
        if segment:
            sorted_segments.append(segment)

    # merge the sorted sub-arrays
    final_sorted = list()
    for segment in sorted_segments:
        final_sorted = merge_sorted(final_sorted, segment)

    return final_sorted


# Tests
assert custom_sort([0, 6, 4, 2, 5, 3, 1]) == [
    0, 1, 2, 3, 4, 5, 6]
assert custom_sort([0, 6, 4, 2, 5, 3, 1, 10, 9]) == [
    0, 1, 2, 3, 4, 5, 6, 9, 10]
assert custom_sort([0, 6, 4, 2, 5, 3, 1, 2, 3]) == [
    0, 1, 2, 2, 3, 3, 4, 5, 6]
assert custom_sort([0, 6, 4, 2, 5, 3, 1, 11]) == [
    0, 1, 2, 3, 4, 5, 6, 11]

```



Question 148

This problem was asked by Apple.

Gray code is a binary code where each successive value differ in only one bit, as well as when wrapping around. Gray code is common in hardware so that we don't see temporary spurious values during transitions.

Given a number of bits `n`, generate a possible gray code for it.

For example, for `n = 2`, one gray code would be `[00, 01, 11, 10]`.

Answer 148





```

def get_gray_code(n):
    """
    n: bits 
    """
    if n == 0:
        return ['']

    lower_grey_codes = get_gray_code(n - 1)
    l0 = ['0' + x for x in lower_grey_codes]
    l1 = ['1' + x for x in reversed(lower_grey_codes)]

    return l0 + l1


# Tests
assert get_gray_code(1) == ['0', '1']
assert get_gray_code(2) == ['00', '01', '11', '10']
assert get_gray_code(3) == ['000', '001', '011',
                            '010', '110', '111', '101', '100']
assert get_gray_code(4) == ['0000', '0001', '0011', '0010', '0110', '0111',
                            '0101', '0100', '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']

```



Question 149

This problem was asked by Goldman Sachs.

Given a list of numbers `L`, implement a method `sum(i, j)` which returns the sum from the sublist `L[i:j]` (including i, excluding j).

For example, given `L = [1, 2, 3, 4, 5]`, `sum(1, 3)` should return `sum([2, 3])`, which is `5`.

You can assume that you can do some pre-processing. `sum()` should be optimized over the pre-processing step.

Answer 149





```

class SubarraySumOptimizer:
    def __init__(self, arr):
        self.arr = arr
        total = 0
        self.larr = [total]
        for num in self.arr:
            total += num
            self.larr.append(total)

    def sum(self, start, end):
        if start < 0 or end > len(self.arr) or start > end:
            return 0
        return self.larr[end] - self.larr[start]


# Tests
sso = SubarraySumOptimizer([1, 2, 3, 4, 5])
assert sso.sum(1, 3) == 5
assert sso.sum(0, 5) == 15
assert sso.sum(0, 4) == 10
assert sso.sum(3, 4) == 4
assert sso.sum(3, 3) == 0

```



Question 150

This problem was asked by LinkedIn.

Given a list of points, a central point, and an integer k, find the nearest k points from the central point.

For example, given the list of points `[(0, 0), (5, 4), (3, 1)]`, the central point `(1, 2)`, and `k = 2`, return `[(0, 0), (3, 1)]`.

Answer 150





```

import math


def calculate_distance(source, target):
    return math.sqrt(
        (source[0] - target[0]) ** 2 +
        (source[1] - target[1]) ** 2
    )


def get_closest_points(source, targets, k):
    if k >= len(targets):
        return targets

    closest_points = \
        sorted(targets, key=lambda x: calculate_distance(source, x))[:k]

    return closest_points



# Tests
assert calculate_distance((0, 0), (3, 4)) == 5
assert get_closest_points(
    (1, 2), [(0, 0), (5, 4), (3, 1)], 2) == [(0, 0), (3, 1)]

```



Question 151

Given a 2-D matrix representing an image, a location of a pixel in the screen and a color C, replace the color of the given pixel and all adjacent same colored pixels with C.

For example, given the following matrix, and location pixel of `(2, 2)`, and `'G'` for green:
```
B B W
W W W
W W W
B B B
```

Becomes
```
B B G
G G G
G G G
B B B
```

Answer 151





```

# pixel is a tuple of (x, y) co-ordinates in the matrix


def get_adj_pixels(pixel, matrix, rows, cols):
    adj_pixels = list()

    # add row above if it exists
    if pixel[0] > 0:
        adj_pixels.append((pixel[0]-1, pixel[1]))
        if pixel[1] > 0:
            adj_pixels.append((pixel[0]-1, pixel[1]-1))
        if pixel[1] < cols - 1:
            adj_pixels.append((pixel[0]-1, pixel[1]+1))

    # add row below if it exists
    if pixel[0] < rows - 1:
        adj_pixels.append((pixel[0]+1, pixel[1]))
        if pixel[1] > 0:
            adj_pixels.append((pixel[0]+1, pixel[1]-1))
        if pixel[1] < cols - 1:
            adj_pixels.append((pixel[0]+1, pixel[1]+1))

    # add left cell if it exists
    if pixel[1] > 0:
        adj_pixels.append((pixel[0], pixel[1]-1))

    # add right cell if it exists
    if pixel[1] < cols - 1:
        adj_pixels.append((pixel[0], pixel[1]+1))

    return adj_pixels


def change_color(pixel, matrix, new_color):
    if not matrix:
        return matrix

    # switch to 0-indexed co-ordinates
    x, y = pixel[0] - 1, pixel[1] - 1

    rows = len(matrix)
    cols = len(matrix[0])
    if x < 0 or y < 0 or x >= rows or y >= cols:
        return matrix

    c = matrix[x][y]
    adj_pixels = get_adj_pixels((x, y), matrix, rows, cols)

    for ap in adj_pixels:
        if matrix[ap[0]][ap[1]] == c:
            matrix[ap[0]][ap[1]] = new_color
    matrix[x][y] = new_color

    return matrix


# Tests
matrix = [['B', 'B', 'W'],
          ['W', 'W', 'W'],
          ['W', 'W', 'W'],
          ['B', 'B', 'B']]
assert change_color((2, 2), matrix, 'G') == \
    [['B', 'B', 'G'],
     ['G', 'G', 'G'],
     ['G', 'G', 'G'],
     ['B', 'B', 'B']]

```



Question 152

This problem was asked by Triplebyte.

You are given `n` numbers as well as `n` probabilities that sum up to `1`. Write a function to generate one of the numbers with its corresponding probability.

For example, given the numbers `[1, 2, 3, 4]` and probabilities `[0.1, 0.5, 0.2, 0.2]`, your function should return `1` `10%` of the time, `2` `50%` of the time, and `3` and `4` `20%` of the time.

You can generate random numbers between 0 and 1 uniformly.

Answer 152





```

from random import random
import bisect


class ProbablisticGenerator:

    def __init__(self, numbers, probabilities):
        assert sum(probabilities) == 1

        self.numbers = numbers
        self.cum_probs = list()
        current_cum_prob = 0
        for prob in probabilities:
            current_cum_prob += prob
            self.cum_probs.append(current_cum_prob)

    def generate_number(self):
        rand = random()
        index = bisect.bisect_left(self.cum_probs, rand)

        return self.numbers[index]


# Tests
numbers = [1, 2, 3, 4]
probs = [0.1, 0.5, 0.2, 0.2]
pg = ProbablisticGenerator(numbers, probs)

num_exp = 10000
outcomes = dict()
for num in numbers:
    outcomes[num] = 0

for _ in range(num_exp):
    gen_num = pg.generate_number()
    outcomes[gen_num] += 1

for num, prob in zip(numbers, probs):
    outcomes[num] = round(outcomes[num] / num_exp, 1)
    assert outcomes[num] == prob

```



Question 153

Find an efficient algorithm to find the smallest distance (measured in number of words) between any two given words in a string.

For example, given words "hello", and "world" and a text content of "dog cat hello cat dog dog hello cat world", return 1 because there's only one word "cat" in between the two words.

Answer 153





```

def get_smallest_dist(text, w1, w2):
    dist = None
    ls_word, ls_index = None, None
    for index, word in enumerate(text.split()):
        if word == w1 or word == w2:
            if (word == w1 and ls_word == w2) or \
                    (word == w2 and ls_word == w1):
                dist = index - ls_index - 1
            ls_word = word
            ls_index = index

    return dist


# Tests
assert not get_smallest_dist(
    "hello", "hello", "world")
assert get_smallest_dist(
    "hello world", "hello", "world") == 0
assert get_smallest_dist(
    "dog cat hello cat dog dog hello cat world", "hello", "world") == 1
assert get_smallest_dist(
    "dog cat hello cat dog dog hello cat world", "dog", "world") == 2

```



Question 154

This problem was asked by Amazon.

Implement a stack API using only a heap. A stack implements the following methods:
* `push(item)`, which adds an element to the stack
* `pop()`, which removes and returns the most recently added element (or throws an error if there is nothing on the stack)

Recall that a heap has the following operations:
* `push(item)`, which adds a new key to the heap
* `pop()`, which removes and returns the max value of the heap

Answer 154





```

import sys
from heapq import heappush, heappop


class Stack:

    def __init__(self):
        self.counter = sys.maxsize
        self.stack = list()

    def push(self, item):
        heappush(self.stack, (self.counter, item))
        self.counter -= 1

    def pop(self):
        if not self.stack:
            return None

        _, item = heappop(self.stack)
        return item


# Tests
stk = Stack()
stk.push(1)
stk.push(7)
stk.push(4)
assert stk.pop() == 4
stk.push(2)
assert stk.pop() == 2
assert stk.pop() == 7
assert stk.pop() == 1
assert not stk.pop()

```



Question 155

Given a list of elements, find the majority element, which appears more than half the times `(> floor(len(lst) / 2.0))`.

You can assume that such an element exists.

For example, given `[1, 2, 1, 1, 3, 4, 0]`, return `1`.

Answer 155





```

import math


def get_majority_element(arr):
    threshold = math.floor(len(arr) / 2)
    occurrences = dict()
    for num in arr:
        if num not in occurrences:
            occurrences[num] = 0
        occurrences[num] += 1
    for num in occurrences:
        if occurrences[num] > threshold:
            return num



# Tests
assert get_majority_element([]) == None
assert get_majority_element([0]) == 0
assert get_majority_element([0, 2, 2]) == 2
assert get_majority_element([1, 2, 1, 1, 3, 4, 1]) == 1

```



Question 156

This problem was asked by Facebook.

Given a positive integer `n`, find the smallest number of squared integers which sum to `n`.

For example, given n = `13`, return `2` since `13 = 3^2 + 2^2 = 9 + 4`.

Given `n = 27`, return `3` since `27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9`.

Answer 156





```

import math


def get_candidate_squares(num):
    candidates = list()
    for candidate_root in range(1, int(num/2)):
        candidate = candidate_root * candidate_root
        if candidate > num:
            break
        candidates.append(candidate)
    candidates.reverse()
    return candidates


def get_min_squares_helper(num, candidates):
    candidate_square = candidates[0]
    max_used = int(num / candidate_square)
    remaining = num % candidate_square

    if remaining == 0:
        return max_used

    return max_used + get_min_squares_helper(remaining, candidates[1:])


def get_min_squares(num):
    candidates = get_candidate_squares(num)
    return get_min_squares_helper(num, candidates)


# Tests
assert get_min_squares(13) == 2
assert get_min_squares(25) == 1
assert get_min_squares(27) == 3

```



Question 157

This problem was asked by Amazon.

Given a string, determine whether any permutation of it is a palindrome.

For example, `carrace` should return `true`, since it can be rearranged to form `racecar`, which is a palindrome. `daily` should return `false`, since there's no rearrangement that can form a palindrome.

Answer 157





```

def check_palindrome_rearrangement(string):
    chars = set()
    for char in string:
        if char not in chars:
            chars.add(char)
        else:
            chars.remove(char)

    return len(chars) < 2


# Tests
assert check_palindrome_rearrangement("carrace")
assert not check_palindrome_rearrangement("daily")
assert not check_palindrome_rearrangement("abac")
assert check_palindrome_rearrangement("abacc")
assert check_palindrome_rearrangement("aabb")
assert not check_palindrome_rearrangement("aabbcccd")

```



Question 158

This problem was asked by Slack.

You are given an `N * M` matrix of `0`s and `1`s. Starting from the top left corner, how many ways are there to reach the bottom right corner?

You can only move right and down. `0` represents an empty space while `1` represents a wall you cannot walk through.

For example, given the following matrix:

```
[[0, 0, 1],
 [0, 0, 1],
 [1, 0, 0]]
```
Return `2`, as there are only two ways to get to the bottom right:
* `Right, down, down, right`
* `Down, right, down, right`

The top left corner and bottom right corner will always be `0`.

Answer 158





```

class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Coord=(x={}, y={})".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


def get_adjacent_cells(coord, rows, cols):
    adj_cells = set()

    if coord.x > 0:
        adj_cells.add(Coord(coord.x-1, coord.y))
    if coord.x < rows - 1:
        adj_cells.add(Coord(coord.x+1, coord.y))
    if coord.y > 0:
        adj_cells.add(Coord(coord.x, coord.y-1))
    if coord.y < cols - 1:
        adj_cells.add(Coord(coord.x, coord.y+1))

    return adj_cells


def valid_path_helper(matrix, coord, rows, cols, current_path):
    if coord.x == rows - 1 and coord.y == cols - 1:
        # base case when already at bottom-right cell
        return 1

    # get adjacent cells as candidates
    adj_cells = get_adjacent_cells(coord, rows, cols)

    # exclude already traversed cells as candidates
    next_candidates = adj_cells - current_path

    # exclude wall cells (=1) as candidates
    next_candidates = [nc for nc in next_candidates if matrix[nc.x][nc.y] == 0]

    new_path = current_path.copy()
    new_path.add(coord)
    path_count = 0
    for cand in next_candidates:
        sub_path_count = valid_path_helper(
            matrix, cand, rows, cols, new_path)
        path_count += sub_path_count

    return path_count


def find_paths(matrix):
    num_paths = valid_path_helper(
        matrix, Coord(0, 0), len(matrix), len(matrix[0]), set())
    return num_paths


# Tests
matrix = [[0, 0, 1],
          [0, 0, 1],
          [1, 0, 0]]
assert find_paths(matrix) == 2

matrix = [[0, 0, 1],
          [1, 0, 1],
          [1, 0, 0]]
assert find_paths(matrix) == 1

matrix = [[0, 0, 0],
          [1, 1, 0],
          [0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]
assert find_paths(matrix) == 1

matrix = [[0, 0, 0],
          [1, 0, 0],
          [0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]
assert find_paths(matrix) == 4

```



Question 159

This problem was asked by Google.

Given a string, return the first recurring character in it, or `null` if there is no recurring chracter.

For example, given the string `"acbbac"`, return `"b"`. Given the string `"abcdef"`, return `null`.

Answer 159





```

def get_first_recurring(string):
    seen = set()
    for char in string:
        if char in seen:
            return char
        seen.add(char)

    return None


# Tests
assert get_first_recurring("acbbac") == "b"
assert not get_first_recurring("abcdef")

```



Question 160

This problem was asked by Uber.

Given a tree where each edge has a weight, compute the length of the longest path in the tree.

For example, given the following tree:

```
   a
  /|\
 b c d
    / \
   e   f
  / \
 g   h
```

and the weights: `a-b: 3`, `a-c: 5`, `a-d: 8`, `d-e: 2`, `d-f: 4`, `e-g: 1`, `e-h: 1`, the longest path would be `c -> a -> d -> f`, with a length of `17`.

The path does not have to pass through the root, and each node can have any amount of children.

Answer 160





```

class Node:
    def __init__(self, iden):
        self.iden = iden
        self.max_path = 0
        self.child_dists = list()

    def __repr__(self):
        return "Node(id={},chi={},mp={})".format(
            self.iden, self.child_dists, self.max_path)


def get_path_maxlen(root):
    if not root.child_dists:
        return 0

    path_lens = list()
    child_max_path_lens = list()
    for child, dist in root.child_dists:
        path_lens.append(child.max_path + dist)
        child_max_path_lens.append(get_path_maxlen(child))

    child_max_path_len = max(child_max_path_lens)

    return max(sum(sorted(path_lens)[-2:]), child_max_path_len)


def update_max_paths(root):
    if not root.child_dists:
        root.max_path = 0
        return

    root_paths = list()
    for child, dist in root.child_dists:
        update_max_paths(child)
        root_paths.append(child.max_path + dist)

    root.max_path = max(root_paths)


def get_longest_path(root):
    update_max_paths(root)
    return get_path_maxlen(root)


# Tests
a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
g = Node('g')
h = Node('h')

e.child_dists = [(g, 1), (h, 1)]
d.child_dists = [(e, 2), (f, 4)]
a.child_dists = [(b, 3), (c, 5), (d, 8)]

assert get_longest_path(a) == 17

```



Question 161

This problem was asked by Facebook.

Given a 32-bit integer, return the number with its bits reversed.

For example, given the binary number `1111 0000 1111 0000 1111 0000 1111 0000`, return `0000 1111 0000 1111 0000 1111 0000 1111`.

Answer 161





```

def reverse_bits(num):
    inverted = list()
    for char in num:
        if char == '0':
            inverted.append('1')
        else:
            inverted.append('0')

    return "".join(inverted)


# Tests
assert reverse_bits('101') == '010'
assert reverse_bits('11110000111100001111000011110000') == \
    '00001111000011110000111100001111'

```



Question 162

This problem was asked by Square.

Given a list of words, return the shortest unique prefix of each word. For example, given the list:
* dog
* cat
* apple
* apricot
* fish

Return the list:
* d
* c
* app
* apr
* f

Answer 162





```

class Trie:
    def __init__(self):
        self.size = 0
        self.letter_map = dict()

    def __repr__(self):
        return str(self.letter_map)

    def add_word(self, word):
        if not word:
            return
        letter = word[0]

        sub_trie = None
        if letter in self.letter_map:
            sub_trie = self.letter_map[letter]
        else:
            sub_trie = Trie()
            self.letter_map[letter] = sub_trie

        self.size += 1
        sub_trie.add_word(word[1:])

    def get_sup(self, word, prev):
        # get shortest unique prefix
        if self.size < 2:
            return prev

        letter = word[0]
        sub_trie = self.letter_map[letter]
        return sub_trie.get_sup(word[1:], prev + letter)


def get_sups(words):
    trie = Trie()
    for word in words:
        trie.add_word(word)

    sups = list()
    for word in words:
        sups.append(trie.get_sup(word, ""))

    return sups


# Tests
assert get_sups(["dog", "cat", "apple", "apricot", "fish"]) == \
    ["d", "c", "app", "apr", "f"]

```



Question 163

This problem was asked by Jane Street.

Given an arithmetic expression in Reverse Polish Notation, write a program to evaluate it.

The expression is given as a list of numbers and operands. For example: `[5, 3, '+']` should return `5 + 3 = 8`.

For example, `[15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']` should return `5`, since it is equivalent to `((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) = 5`.

You can assume the given expression is always valid.

Answer 163





```

OPERATORS = {'+', '*', '/', '-'}


def evaluate_expression(exp_list):
    stack = list()

    for exp in exp_list:
        if exp in OPERATORS:
            op2 = stack.pop()
            op1 = stack.pop()
            ans = eval(str(op1) + exp + str(op2))
            stack.append(ans)
        else:
            stack.append(exp)

    return stack[0]


# Tests
assert evaluate_expression([5, 3, '+']) == 8
assert evaluate_expression(
    [15, 7, 1, 1, '+', '-', '/', 3, '*',
     2, 1, 1, '+', '+', '-']) == 5

```



Question 164

This problem was asked by Google.

You are given an array of length n + 1 whose elements belong to the set `{1, 2, ..., n}`. By the pigeonhole principle, there must be a duplicate. Find it in linear time and space.

Answer 164





```

def find_duplicate(arr, n):
    expected_sum = int((n * (n+1)) / 2)
    actual_sum = sum(arr)

    return actual_sum - expected_sum


# Tests

assert find_duplicate([1, 1, 2], 2) == 1
assert find_duplicate([1, 2, 3, 3], 3) == 3
assert find_duplicate([1, 2, 3, 4, 3], 4) == 3

```



Question 165

This problem was asked by Google.

Given an array of integers, return a new array where each element in the new array is the number of smaller elements to the right of that element in the original input array.

For example, given the array `[3, 4, 9, 6, 1]`, return `[1, 1, 2, 1, 0]`, since:
* There is 1 smaller element to the right of `3`
* There is 1 smaller element to the right of `4`
* There are 2 smaller elements to the right of `9`
* There is 1 smaller element to the right of `6`
* There are no smaller elements to the right of `1`

Answer 165





```

def get_smaller_right(arr):
    smaller_right_arr = list()
    for i in range(len(arr)):
        smaller_count = 0
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[i]:
                smaller_count += 1
        smaller_right_arr.append(smaller_count)

    return smaller_right_arr


# Tests

assert get_smaller_right([3, 4, 9, 6, 1]) == [1, 1, 2, 1, 0]

```



Question 166

This problem was asked by Uber.

Implement a 2D iterator class. It will be initialized with an array of arrays, and should implement the following methods:
* `next()`: returns the next element in the array of arrays. If there are no more elements, raise an exception.
* `has_next()`: returns whether or not the iterator still has elements left.

For example, given the input `[[1, 2], [3], [], [4, 5, 6]]`, calling `next()` repeatedly should output `1, 2, 3, 4, 5, 6`.

Do not use flatten or otherwise clone the arrays. Some of the arrays can be empty.

Answer 166





```

class TwoDimIterator:

    def __init__(self, arrays):
        self.arrays = arrays
        self.gen = self.get_generator()
        self.next_val = next(self.gen)

    def get_generator(self):
        for array in self.arrays:
            for num in array:
                yield num

    def has_next(self):
        return self.next_val != None

    def next(self):
        val = self.next_val
        try:
            self.next_val = next(self.gen)
        except StopIteration:
            self.next_val = None

        return val


# Tests

tdi = TwoDimIterator([[0, 1, 2], [3], [], [4, 5, 6]])

assert tdi.has_next()
assert tdi.next() == 0
assert tdi.next() == 1
assert tdi.next() == 2
assert tdi.has_next()
assert tdi.next() == 3
assert tdi.next() == 4
assert tdi.next() == 5
assert tdi.has_next()
assert tdi.next() == 6
assert not tdi.has_next()
assert not tdi.next()

```



Question 167

This problem was asked by Airbnb.

Given a list of words, find all pairs of unique indices such that the concatenation of the two words is a palindrome.

For example, given the list `["code", "edoc", "da", "d"]`, return `[(0, 1), (1, 0), (2, 3)]`.

Answer 167





```

def is_palindrome(word):
    return word == word[::-1]


def get_unique_index_tuples(words):
    index_tuples = list()
    for i, word_i in enumerate(words):
        for j, word_j in enumerate(words):
            if i != j:
                composite = word_i + word_j
                if is_palindrome(composite):
                    index_tuples.append((i, j))

    return index_tuples


# Tests
assert get_unique_index_tuples(["code", "edoc", "da", "d"]) == [
    (0, 1), (1, 0), (2, 3)]

```



Question 168

This problem was asked by Facebook.

Given an N by N matrix, rotate it by 90 degrees clockwise.

For example, given the following matrix:
```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

you should return:
```
[[7, 4, 1],
 [8, 5, 2],
 [9, 6, 3]]
```

Follow-up: What if you couldn't use any extra space?

Answer 168





```

def rotate_matrix(m):
    num_layers = len(m) // 2
    max_ind = len(m) - 1

    for layer in range(num_layers):
        # rotate all numbers
        for ind in range(layer, max_ind - layer):
            # rotate 4 numbers
            temp = m[layer][ind]
            m[layer][ind] = m[max_ind - ind][layer]
            m[max_ind - ind][layer] = m[max_ind - layer][max_ind - ind]
            m[max_ind - layer][max_ind - ind] = m[ind][max_ind - layer]
            m[ind][max_ind - layer] = temp


# Tests
matrix_1 = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
rotate_matrix(matrix_1)
assert matrix_1 == [[7, 4, 1],
                    [8, 5, 2],
                    [9, 6, 3]]

matrix_2 = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
rotate_matrix(matrix_2)
assert matrix_2 == [[13, 9, 5, 1],
                    [14, 10, 6, 2],
                    [15, 11, 7, 3],
                    [16, 12, 8, 4]]

```



Question 169

This problem was asked by Google.

Given a linked list, sort it in `O(n log n)` time and constant space.

For example, the linked list `4 -> 1 -> -3 -> 99` should become `-3 -> 1 -> 4 -> 99`.

Answer 169





```

import sys


class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def merge_sorted_lists(list_a, list_b):
    head = Node(-sys.maxsize)
    node = head
    node_a = list_a
    node_b = list_b

    while node_a and node_b:
        if node_a.val < node_b.val:
            tmp = node_a
            node_a = node_a.next
        else:
            tmp = node_b
            node_b = node_b.next
        tmp.next = None
        node.next = tmp
        node = node.next

    if node_a:
        node.next = node_a
    if node_b:
        node.next = node_b

    return head.next


def sort_ll_helper(llist, node_count):
    if node_count == 1:
        return llist

    mid = node_count // 2
    right_head = llist
    left_tail = None
    for _ in range(mid):
        left_tail = right_head
        right_head = right_head.next

    left_tail.next = None

    sorted_left = sort_ll_helper(llist, mid)
    sorted_right = sort_ll_helper(right_head, node_count - mid)

    return merge_sorted_lists(sorted_left, sorted_right)


def sort_ll(llist):
    count = 0
    node = llist
    while node:
        count += 1
        node = node.next

    return sort_ll_helper(llist, count)


# Tests
llist = get_nodes([4, 1, -3, 99])
assert get_list(sort_ll(llist)) == [-3, 1, 4, 99]

```



Question 170

This problem was asked by Facebook.

Given a start word, an end word, and a dictionary of valid words, find the shortest transformation sequence from start to end such that only one letter is changed at each step of the sequence, and each transformed word exists in the dictionary. If there is no possible transformation, return null. Each word in the dictionary have the same length as start and end and is lowercase.

For example, given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "dop", "dat", "cat"}`, return `["dog", "dot", "dat", "cat"]`.

Given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "tod", "dat", "dar"}`, return null as there is no possible transformation from dog to cat.

Answer 170





```

import sys

chars = set("abcdefghijklmnopqrstuvwxyz")


def transition_helper(start, end, dictionary, changes, seen):
    if start == end:
        return changes

    candidates = list()
    for index, _ in enumerate(start):
        for char in chars:
            candidate = start[:index] + char + start[index + 1:]
            if candidate in dictionary and candidate not in seen:
                candidates.append(candidate)

    min_results = list()
    min_len = sys.maxsize
    for candidate in candidates:
        new_seen = seen.copy()
        new_seen.add(candidate)
        result_changes = transition_helper(
            candidate, end, dictionary, changes + [candidate], new_seen)
        if result_changes and len(result_changes) < min_len:
            min_len = len(result_changes)
            min_results = result_changes

    return min_results


def get_transition(start, end, dictionary):
    result = transition_helper(start, end, dictionary, [start], {start})
    return result


# Tests
assert get_transition("dog", "cat", {"dot", "dop", "dat", "cat"}) == \
    ["dog", "dot", "dat", "cat"]
assert not get_transition("dog", "cat", {"dot", "tod", "dat", "dar"})

```



Question 171

This problem was asked by Amazon.

You are given a list of data entries that represent entries and exits of groups of people into a building. An entry looks like this:

`{"timestamp": 1526579928, "count": 3, "type": "enter"}`

This means 3 people entered the building. An exit looks like this:

`{"timestamp": 1526580382, "count": 2, "type": "exit"}`

This means that 2 people exited the building. timestamp is in Unix time.

Find the busiest period in the building, that is, the time with the most people in the building. Return it as a pair of `(start, end)` timestamps. You can assume the building always starts off and ends up empty, i.e. with 0 people inside.

Answer 171





```

import sys

ENTER = "enter"
EXIT = "exit"


def get_busiest_slot(events):
    ts_entries, ts_exits = dict(), dict()
    max_time, min_time = -sys.maxsize, sys.maxsize

    for event in events:
        ts_dict = None
        timestamp = event["timestamp"]
        if event["type"] == ENTER:
            ts_dict = ts_entries
        else:
            ts_dict = ts_exits

        ts_dict[timestamp] = event["count"]
        if timestamp < min_time:
            min_time = timestamp
        elif timestamp > max_time:
            max_time = timestamp

    people_inside = 0
    max_people_inside = 0
    start_time, end_time = None, None
    for timestamp in range(min_time, max_time + 1):
        if timestamp in ts_entries:
            people_inside += ts_entries[timestamp]
            if people_inside > max_people_inside:
                max_people_inside = people_inside
                start_time = timestamp
        if timestamp in ts_exits:
            if people_inside == max_people_inside:
                end_time = timestamp
            people_inside -= ts_exits[timestamp]

    return (start_time, end_time)


# Tests
events = [
    {"timestamp": 1526579928, "count": 3, "type": "enter"},
    {"timestamp": 1526579982, "count": 4, "type": "enter"},
    {"timestamp": 1526580054, "count": 5, "type": "exit"},
    {"timestamp": 1526580128, "count": 1, "type": "enter"},
    {"timestamp": 1526580382, "count": 3, "type": "exit"}
]
assert get_busiest_slot(events) == (1526579982, 1526580054)

```



Question 172

This problem was asked by Dropbox.

Given a string `s` and a list of words `words`, where each word is the same length, find all starting indices of substrings in `s` that is a concatenation of every word in `words` exactly once.

For example, given `s = "dogcatcatcodecatdog"` and `words = ["cat", "dog"]`, return `[0, 13]`, since `"dogcat"` starts at index `0` and `"catdog"` starts at index `13`.

Given `s = "barfoobazbitbyte"` and `words = ["dog", "cat"]`, return `[]` since there are no substrings composed of `"dog"` and `"cat"` in `s`.

The order of the indices does not matter.

Answer 172





```

from itertools import permutations


def get_indices(s, words):
    perms = list(permutations(words))
    perms = [x + y for (x, y) in perms]

    indices = [s.find(x) for x in perms]
    indices = [x for x in indices if x >= 0]

    return sorted(indices)


# Tests
assert get_indices("dogcatcatcodecatdog", ["cat", "dog"]) == [0, 13]
assert not get_indices("barfoobazbitbyte", ["cat", "dog"])

```



Question 173

This problem was asked by Stripe.

Write a function to flatten a nested dictionary. Namespace the keys with a period.

For example, given the following dictionary:

```
{
    "key": 3,
    "foo": {
        "a": 5,
        "bar": {
            "baz": 8
        }
    }
}
```

it should become:

```
{
    "key": 3,
    "foo.a": 5,
    "foo.bar.baz": 8
}
```

You can assume keys do not contain dots in them, i.e. no clobbering will occur.

Answer 173





```

def is_dict(var):
    return str(type(var)) == "<class 'dict'>"


def flatten_helper(d, flat_d, path):
    if not is_dict(d):
        flat_d[path] = d
        return

    for key in d:
        new_keypath = "{}.{}".format(path, key) if path else key
        flatten_helper(d[key], flat_d, new_keypath)


def flatten(d):
    flat_d = dict()
    flatten_helper(d, flat_d, "")
    return flat_d


# Tests

d = {
    "key": 3,
    "foo": {
        "a": 5,
        "bar": {
            "baz": 8
        }
    }
}

assert flatten(d) == {
    "key": 3,
    "foo.a": 5,
    "foo.bar.baz": 8
}

```



Question 174

This problem was asked by Microsoft.

Describe and give an example of each of the following types of polymorphism:
* Ad-hoc polymorphism
* Parametric polymorphism
* Subtype polymorphism

Answer 174





```

* Ad-hoc polymorphism
    * Allow multiple functions that perform an operation on different 
    * `add(int, int)` and `add(str, str)` would be separately implemented
* Parametric polymorphism
    * This allows a function to deal with generics, and therefore on any concrete definition of the generic.
    * e.g. A `List` type in Java, regardless of which objects are in the list
* Subtype polymorphism
    * This allows subclass instances to be treated by a function they same way as it would superclass instances
    * e.g. Instances of `Cat` will be operated on a function the same way as instances of it's superclass `Animal`.

```



Question 175

This problem was asked by Google.

You are given a starting state start, a list of transition probabilities for a Markov chain, and a number of steps num_steps. Run the Markov chain starting from start for num_steps and compute the number of times we visited each state.

For example, given the starting state `a`, number of steps `5000`, and the following transition probabilities:

```
[
  ('a', 'a', 0.9),
  ('a', 'b', 0.075),
  ('a', 'c', 0.025),
  ('b', 'a', 0.15),
  ('b', 'b', 0.8),
  ('b', 'c', 0.05),
  ('c', 'a', 0.25),
  ('c', 'b', 0.25),
  ('c', 'c', 0.5)
]
```
One instance of running this Markov chain might produce `{'a': 3012, 'b': 1656, 'c': 332 }`.

Answer 175





```

from random import random
import bisect


def get_transition_map(transition_probs):
    transition_map = dict()
    for source, target, prob in transition_probs:
        if source not in transition_map:
            transition_map[source] = ([], [])

        if not transition_map[source][0]:
            transition_map[source][0].append(prob)
        else:
            prev_prob = transition_map[source][0][-1]
            transition_map[source][0].append(prev_prob + prob)
        transition_map[source][1].append(target)

    return transition_map


def get_new_state(state_trans_probs):
    rand = random()
    probs, states = state_trans_probs
    index = bisect.bisect(probs, rand)

    return states[index]


def calculate_visits(start_state, transition_probs, steps):
    transition_map = get_transition_map(transition_probs)

    visit_counter = dict()
    for state in transition_map:
        visit_counter[state] = 0

    for _ in range(steps):
        visit_counter[start_state] += 1

        state_trans_probs = transition_map[start_state]
        start_state = get_new_state(state_trans_probs)

    return visit_counter


# Tests
transition_probs = [
    ('a', 'a', 0.9),
    ('a', 'b', 0.075),
    ('a', 'c', 0.025),
    ('b', 'a', 0.15),
    ('b', 'b', 0.8),
    ('b', 'c', 0.05),
    ('c', 'a', 0.25),
    ('c', 'b', 0.25),
    ('c', 'c', 0.5)
]
visit_counter = calculate_visits('a', transition_probs, 50000)
assert visit_counter['a'] > visit_counter['b']
assert visit_counter['a'] > visit_counter['c']
assert visit_counter['b'] > visit_counter['c']

```



Question 176

This problem was asked by Bloomberg.

Determine whether there exists a one-to-one character mapping from one string `s1` to another `s2`.

For example, given `s1 = abc` and `s2 = bcd`, return `true` since we can map `a` to `b`, `b` to `c`, and `c` to `d`.

Given `s1 = foo` and `s2 = bar`, return `false` since the `o` cannot map to two characters.

Answer 176





```

def is_char_mapped(str_a, str_b):
    if len(str_a) != len(str_b):
        return False

    char_map = dict()
    for char_a, char_b in zip(str_a, str_b):
        if char_a not in char_map:
            char_map[char_a] = char_b
        elif char_map[char_a] != char_b:
            return False

    return True


# Tests
assert is_char_mapped("abc", "bcd")
assert not is_char_mapped("foo", "bar")

```



Question 177

This problem was asked by Airbnb.

Given a linked list and a positive integer `k`, rotate the list to the right by `k` places.

For example, given the linked list `7 -> 7 -> 3 -> 5` and `k = 2`, it should become `3 -> 5 -> 7 -> 7`.

Given the linked list `1 -> 2 -> 3 -> 4 -> 5` and `k = 3`, it should become `3 -> 4 -> 5 -> 1 -> 2`.

Answer 177





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def rotate_ll(llist, k):
    cnode = llist
    head = cnode
    size = 0
    while cnode:
        tail = cnode
        cnode = cnode.next
        size += 1

    new_head = llist
    new_tail = None
    for _ in range(size - k):
        new_tail = new_head
        new_head = new_head.next

    tail.next = head
    new_tail.next = None

    return new_head


# Tests

assert get_list(rotate_ll(get_nodes([7, 7, 3, 5]), 2)) == [3, 5, 7, 7]
assert get_list(rotate_ll(get_nodes([1, 2, 3, 4, 5]), 3)) == [3, 4, 5, 1, 2]

```



Question 178

This problem was asked by Two Sigma.

Alice wants to join her school's Probability Student Club. Membership dues are computed via one of two simple probabilistic games.

The first game: roll a die repeatedly. Stop rolling once you get a five followed by a six. Your number of rolls is the amount you pay, in dollars.

The second game: same, except that the stopping condition is a five followed by a five.

Which of the two games should Alice elect to play? Does it even matter? Write a program to simulate the two games and calculate their expected value.

Answer 178





```

# I would think that there is no difference between either game
# because the `nth` and the `(n-1)th` dice rolls should be
# independent events

from random import randint


def roll_dice(sought_values, index, prev_tosses):
    rand = randint(1, 6)

    if sought_values[index] == rand:
        index += 1

        if index == len(sought_values):
            return prev_tosses + 1

    return roll_dice(sought_values, index, prev_tosses + 1)


def simulate_game(sought_values):
    penalty = roll_dice(sought_values, 0, 0)
    return penalty


# Tests

num_exp = 100000
sought_value_pairs = [[5, 6], [5, 5]]

for sought_values in sought_value_pairs:
    total_penalty = 0
    for _ in range(num_exp):
        total_penalty += simulate_game(sought_values)
    avg_penalty = total_penalty / num_exp

    #  the expected value is approx. 12 on average for a single game
    assert round(avg_penalty) == 12

```



Question 179

This problem was asked by Google.

Given the sequence of keys visited by a postorder traversal of a binary search tree, reconstruct the tree.

For example, given the sequence `2, 4, 3, 8, 7, 5`, you should construct the following tree:

```
    5
   / \
  3   7
 / \   \
2   4   8
```

Answer 179





```

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def __repr__(self):
        return "{} => (l: {}, r: {})".format(
            self.data, self.left, self.right)


def get_tree(seq):
    head = Node(seq[-1])
    if len(seq) == 1:
        return head

    for i in range(len(seq) - 1):
        if seq[i] > head.data:
            sep_ind = i
            break

    leq, gt = seq[:sep_ind], seq[sep_ind:-1]

    head.left = get_tree(leq) if leq else None
    head.right = get_tree(gt) if gt else None

    return head


# Tests
tree = get_tree([2, 4, 3, 8, 7, 5])
assert tree.data == 5
assert tree.left.data == 3
assert tree.right.data == 7
assert tree.left.left.data == 2
assert tree.left.right.data == 4
assert tree.right.right.data == 8

```



Question 180

This problem was asked by Google.

Given a stack of `N` elements, interleave the first half of the stack with the second half reversed using only one other queue. This should be done in-place.

Recall that you can only push or pop from a stack, and enqueue or dequeue from a queue.

For example, if the stack is `[1, 2, 3, 4, 5]`, it should become `[1, 5, 2, 4, 3]`. If the stack is `[1, 2, 3, 4]`, it should become `[1, 4, 2, 3]`.

Hint: Try working backwards from the end state.

Answer 180





```

from queue import Queue


def interleave_stack(stack, queue, index=1):
    for _ in range(len(stack) - index):
        que.put(stack.pop())

    while que.qsize():
        stk.append(que.get())

    if len(stack) - index > 1:
        interleave_stack(stack, queue, index + 1)


# Tests
stk = [1, 2, 3, 4, 5]
que = Queue()
interleave_stack(stk, que)
assert stk == [1, 5, 2, 4, 3]

stk = [1, 2, 3, 4]
que = Queue()
interleave_stack(stk, que)
assert stk == [1, 4, 2, 3]

```



Question 181

This problem was asked by Google.

Given a string, split it into as few strings as possible such that each string is a palindrome.

For example, given the input string `"racecarannakayak"`, return `["racecar", "anna", "kayak"]`.

Given the input string `"abc"`, return `["a", "b", "c"]`.

Answer 181





```

def is_palindrome(string):
    return bool(string) and string == string[::-1]


def split_into_pals(string, curr="", prev_pals=[]):

    if not string and not curr:
        return prev_pals
    elif not string:
        return prev_pals + list(curr)

    candidate = curr + string[0]

    alt_1 = []
    if is_palindrome(candidate):
        alt_1 = split_into_pals(string[1:], "", prev_pals + [candidate])
    alt_2 = split_into_pals(string[1:], candidate, prev_pals)

    return alt_1 if bool(alt_1) and len(alt_1) < len(alt_2) else alt_2


# Tests
assert split_into_pals("racecarannakayak") == ["racecar", "anna", "kayak"]
assert split_into_pals("abc") == ["a", "b", "c"]
assert split_into_pals("madam") == ["madam"]
assert split_into_pals("madama") == ["madam", "a"]


```



Question 182

This problem was asked by Facebook.

A graph is minimally-connected if it is connected and there is no edge that can be removed while still leaving the graph connected. For example, any binary tree is minimally-connected.

Given an undirected graph, check if the graph is minimally-connected. You can choose to represent the graph as either an adjacency matrix or adjacency list.

Answer 182





```

from copy import deepcopy


class Node:
    def __init__(self, id_str):
        self.id = id_str

    def __repr__(self):
        return str(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class Graph:
    def __init__(self):
        self.nodes = set()
        self.adj_lists = dict()

    def add_edge(self, node_a, node_b):
        for node in [node_a, node_b]:
            if node not in self.nodes:
                self.nodes.add(node)
                self.adj_lists[node] = set()

        self.adj_lists[node_a].add(node_b)
        self.adj_lists[node_b].add(node_a)


def are_connected(g, source, target, seen=set()):
    if source == target:
        return True

    seen_cp = seen.copy()
    seen_cp.add(source)

    return any(are_connected(g, x, target, seen_cp)
               for x in g.adj_lists[source] if x not in seen)


def is_min_graph(g):
    if all(len(x) == 1 for x in g.adj_lists.values()):
        return True

    for node in g.nodes:
        cp_g = deepcopy(g)

        assert node in cp_g.nodes
        adj_nodes = list(cp_g.adj_lists[node])
        cp_g.adj_lists.pop(node)

        for an in adj_nodes:
            cp_g.adj_lists[an].remove(node)

        for i in range(len(adj_nodes)):
            for j in range(i + 1, len(adj_nodes)):
                if are_connected(cp_g, adj_nodes[i], adj_nodes[j]):
                    return False

    return True


# Tests
a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')

g = Graph()
g.add_edge(a, b)
assert is_min_graph(g)

g.add_edge(b, c)
assert is_min_graph(g)

g.add_edge(a, c)
assert not is_min_graph(g)

g.add_edge(a, d)
assert not is_min_graph(g)

```



Question 183

This problem was asked by Twitch.

Describe what happens when you type a URL into your browser and press Enter.

Answer 183





```

# Process of a URL being fetched

1. Check browsers host resolution cache.
2. If IP not found, request DNS for the IP of the host/load balancing host.
3. Send a GET request to the host
4. Host responds with a payload (usually in HTML)
5. Browser renders HTML


```



Question 184

This problem was asked by Amazon.

Given n numbers, find the greatest common denominator between them.

For example, given the numbers `[42, 56, 14]`, return `14`.

Answer 184





```

SMALLEST_PRIME = 2


def is_prime(cand, primes):
    for prime in primes:
        res = cand / prime
        if not res % 1:
            return False

    return True


def get_possible_primes(num):
    primes = [SMALLEST_PRIME]

    for cand in range(SMALLEST_PRIME + 1, num//2 + 1):
        if is_prime(cand, primes):
            primes.append(cand)

    return primes


def get_factors(num, primes):
    factors = dict()

    pi = 0
    while num > 1:
        if pi >= len(primes):
            break
        if not num % primes[pi]:
            if primes[pi] not in factors:
                factors[primes[pi]] = 0
            factors[primes[pi]] += 1
            num /= primes[pi]
        else:
            pi += 1
    return factors


def get_gcd(nums):
    min_num = min(nums)
    primes = get_possible_primes(min_num)
    base_factors = get_factors(min_num, primes)

    factorized_nums = dict()
    for num in nums:
        factorized_nums[num] = get_factors(num, primes)

    common_factors = dict()
    for base_factor in base_factors:
        common_factors[base_factor] = 0
        num_factors = list()
        for num in nums:
            factors = factorized_nums[num]
            num_factors.append(factors[base_factor])
        common_factors[base_factor] = min(num_factors)

    gcd = 1
    for factor in common_factors:
        gcd *= factor ** common_factors[factor]

    return gcd


# Tests
assert get_gcd([42, 56, 14]) == 14
assert get_gcd([3, 5]) == 1
assert get_gcd([9, 15]) == 3

```



Question 185

This problem was asked by Google.

Given two rectangles on a 2D graph, return the area of their intersection. If the rectangles don't intersect, return `0`.

For example, given the following rectangles:
```
{
    "top_left": (1, 4),
    "dimensions": (3, 3) # width, height
}
```
and
```
{
    "top_left": (0, 5),
    "dimensions" (4, 3) # width, height
}
```
return `6`.

Answer 185





```

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "(x={},y={})".format(self.x, self.y)


class Rectangle:
    def __init__(self, json):
        self.tl = Point(json["top_left"][0], json["top_left"][1])
        width, height = json["dimensions"]
        self.br = Point(self.tl.x + width, self.tl.y - height)

    def __repr__(self):
        return "(tl={},br={})".format((self.tl), (self.br))

    def get_area(self):
        return (self.br.x - self.tl.x) * (self.tl.y - self.br.y)

    def envelopes(self, other):
        return self.tl.x < other.tl.x and self.tl.y > other.tl.y and \
            self.br.x > other.br.x and self.br.y < other.br.y

    def is_disjoint_with(self, other):
        return self.br.y > other.tl.y or other.br.y > self.tl.y or \
            self.tl.x > other.br.x or other.tl.x > self.br.x


def calculate_intersect_area(r1, r2):
    if r1.envelopes(r2):
        area = r2.get_area()
    elif r2.envelopes(r1):
        area = r1.get_area()
    elif r1.is_disjoint_with(r2) or r2.is_disjoint_with(r1):
        area = 0
    else:
        heights = list(sorted([r1.tl.y, r1.br.y, r2.tl.y, r2.br.y]))[1:-1]
        height = heights[1] - heights[0]
        widths = list(sorted([r1.tl.x, r1.br.x, r2.tl.x, r2.br.x]))[1:-1]
        width = widths[1] - widths[0]
        area = height * width

    return area


# Tests
r1 = Rectangle({"top_left": (1, 4), "dimensions": (3, 3)})
r2 = Rectangle({"top_left": (0, 5), "dimensions": (4, 3)})
assert calculate_intersect_area(r1, r2) == 6

r1 = Rectangle({"top_left": (1, 1), "dimensions": (1, 1)})
r2 = Rectangle({"top_left": (5, 5), "dimensions": (1, 1)})
assert calculate_intersect_area(r1, r2) == 0

r1 = Rectangle({"top_left": (0, 5), "dimensions": (5, 5)})
r2 = Rectangle({"top_left": (1, 4), "dimensions": (2, 2)})
assert calculate_intersect_area(r1, r2) == 4

r1 = Rectangle({"top_left": (0, 5), "dimensions": (5, 5)})
r2 = Rectangle({"top_left": (4, 4), "dimensions": (3, 3)})
assert calculate_intersect_area(r1, r2) == 3

```



Question 186

This problem was asked by Microsoft.

Given an array of positive integers, divide the array into two subsets such that the difference between the sum of the subsets is as small as possible.

For example, given `[5, 10, 15, 20, 25]`, return the sets `{10, 25}` and `{5, 15, 20}`, which has a difference of `5`, which is the smallest possible difference.

Answer 186





```

def get_diff(s1, s1_sum, s2, s2_sum, score):
    min_diff, min_cand = score, None
    for i, num in enumerate(s1):
        new_s1_sum, new_s2_sum = s1_sum - num, s2_sum + num
        new_score = abs(new_s1_sum - new_s2_sum)
        if new_score < min_diff:
            min_diff = new_score
            min_cand = (s1[:i] + s1[i + 1:], new_s1_sum,
                        s2 + [num], new_s2_sum)

    if not min_cand:
        return (set(s1), set(s2))

    return get_diff(min_cand[0], min_cand[1], min_cand[2], min_cand[3], min_diff)


def divide_numbers(nums):
    sum_nums = sum(nums)
    best_sets = get_diff(nums.copy(), sum_nums, [], 0, sum_nums)

    return best_sets


# Tests
assert divide_numbers([5, 10, 15, 20, 25]) == ({5, 15, 20}, {10, 25})
assert divide_numbers([5, 10, 15, 20]) == ({10, 15}, {20, 5})

```



Question 187

This problem was asked by Google.

You are given given a list of rectangles represented by min and max x- and y-coordinates. Compute whether or not a pair of rectangles overlap each other. If one rectangle completely covers another, it is considered overlapping.

For example, given the following rectangles:
```
{
    "top_left": (1, 4),
    "dimensions": (3, 3) # width, height
},
{
    "top_left": (-1, 3),
    "dimensions": (2, 1)
},
{
    "top_left": (0, 5),
    "dimensions": (4, 3)
}
```

return `true` as the first and third rectangle overlap each other.

Answer 187





```

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "(x={},y={})".format(self.x, self.y)


class Rectangle:
    def __init__(self, json):
        self.tl = Point(json["top_left"][0], json["top_left"][1])
        width, height = json["dimensions"]
        self.br = Point(self.tl.x + width, self.tl.y - height)

    def __repr__(self):
        return "(tl={},br={})".format((self.tl), (self.br))

    def envelopes(self, other):
        return self.tl.x <= other.tl.x and self.tl.y >= other.tl.y and \
            self.br.x >= other.br.x and self.br.y <= other.br.y


def contains_overlapping_pair(rectangles):

    for i in range(len(rectangles) - 1):
        for j in range(i + 1, len(rectangles)):
            if rectangles[i].envelopes(rectangles[j]) or \
                    rectangles[j].envelopes(rectangles[i]):
                return True

    return False


# Tests

# example provided in the question is incorrect
r1 = Rectangle({"top_left": (1, 4), "dimensions": (3, 3)})
r2 = Rectangle({"top_left": (-1, 3), "dimensions": (2, 1)})
r3 = Rectangle({"top_left": (0, 5), "dimensions": (4, 4)})
assert contains_overlapping_pair([r1, r2, r3])

r1 = Rectangle({"top_left": (1, 4), "dimensions": (3, 3)})
r2 = Rectangle({"top_left": (-1, 3), "dimensions": (2, 1)})
r3 = Rectangle({"top_left": (0, 5), "dimensions": (4, 3)})
assert not contains_overlapping_pair([r1, r2, r3])

```



Question 188

This problem was asked by Google.

What will this code print out?

```
def make_functions():
    flist = []

    for i in [1, 2, 3]:
        def print_i():
            print(i)
        flist.append(print_i)

    return flist

functions = make_functions()
for f in functions:
    f()
```

How can we make it print out what we apparently want?

Answer 188





```

# Unchanged code prints 3, 3, 3
# To make it print 1, 2, 3, use the following version


def make_functions():
    flist = []

    for i in [1, 2, 3]:
        def print_i(i):
            print(i)
        flist.append((print_i, i))

    return flist


functions = make_functions()
for f, i in functions:
    f(i)

```



Question 189

This problem was asked by Google.

Given an array of elements, return the length of the longest subarray where all its elements are distinct.

For example, given the array `[5, 1, 3, 5, 2, 3, 4, 1]`, return `5` as the longest subarray of distinct elements is `[5, 2, 3, 4, 1]`.

Answer 189





```

def get_longest_uqsub(arr, seen=set()):
    if not arr:
        return len(seen)

    curr = arr[0]
    if curr in seen:
        return len(seen)

    seen_cp = seen.copy()
    seen_cp.add(curr)

    return max(get_longest_uqsub(arr[1:], seen_cp), 
               get_longest_uqsub(arr[1:]))



# Tests
assert get_longest_uqsub([]) == 0
assert get_longest_uqsub([5, 5, 5]) == 1
assert get_longest_uqsub([5, 1, 3, 5, 2, 3, 4, 1]) == 5
assert get_longest_uqsub([5, 1, 3, 5, 2, 3, 4]) == 4
assert get_longest_uqsub([5, 1, 3, 5, 2, 3]) == 4

```



Question 190

This problem was asked by Facebook.

Given a circular array, compute its maximum subarray sum in `O(n)` time.

For example, given `[8, -1, 3, 4]`, return `15` as we choose the numbers `3`, `4`, and `8` where the `8` is obtained from wrapping around.

Given `[-4, 5, 1, 0]`, return `6` as we choose the numbers `5` and `1`.

Answer 190





```

def get_max_circ_sarray(arr):
    warr = arr * 2

    items = []
    csum = msum = 0

    for num in warr:
        while len(items) >= len(arr) or (items and items[0] < 1):
            csum -= items[0]
            items = items[1:]

        items.append(num)
        csum += num

        msum = max(msum, csum)

    return msum


# Tests
assert get_max_circ_sarray([8, -1, 3, 4]) == 15
assert get_max_circ_sarray([-4, 5, 1, 0]) == 6

```



Question 191

This problem was asked by Stripe.

Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Intervals can "touch", such as `[0, 1]` and `[1, 2]`, but they won't be considered overlapping.

For example, given the intervals `(7, 9), (2, 4), (5, 8)`, return `1` as the last interval can be removed and the first two won't overlap.

The intervals are not necessarily sorted in any order.

Answer 191





```

def get_min_removals(intervals, reserved_intervals=list(), removed=0):
    print(reserved_intervals)

    if not intervals:
        return removed

    curr_interval = intervals[0]
    if_removed = get_min_removals(
        intervals[1:], reserved_intervals, removed + 1)

    for ri in reserved_intervals:
        if curr_interval[0] in ri or curr_interval[1] in ri:
            return if_removed

    new_reserved_intervals = reserved_intervals + \
        [range(curr_interval[0], curr_interval[1])]

    return min(if_removed, get_min_removals(intervals[1:], new_reserved_intervals, removed))


# Tests
assert get_min_removals([(0, 1), (1, 2)]) == 0
assert get_min_removals([(7, 9), (2, 4), (5, 8)]) == 1
assert get_min_removals([(7, 9), (2, 4), (5, 8), (1, 3)]) == 2

```



Question 192

This problem was asked by Google.

You are given an array of nonnegative integers. Let's say you start at the beginning of the array and are trying to advance to the end. You can advance at most, the number of steps that you're currently on. Determine whether you can get to the end of the array.

For example, given the array `[1, 3, 1, 2, 0, 1]`, we can go from indices `0 -> 1 -> 3 -> 5`, so return `true`.

Given the array `[1, 2, 1, 0, 0]`, we can't reach the end, so return `false`.

Answer 192





```

def end_reachable(arr):
    if len(arr) < 2:
        return True

    for i in range(2, len(arr) + 1):
        if arr[len(arr) - i] >= i - 1:
            return end_reachable(arr[:len(arr) - i + 1])


# Tests
assert end_reachable([1, 3, 1, 2, 0, 1])
assert not end_reachable([1, 2, 1, 0, 0])

```



Question 193

This problem was asked by Affirm.

Given a array of numbers representing the stock prices of a company in chronological order, write a function that calculates the maximum profit you could have made from buying and selling that stock. You're also given a number fee that represents a transaction fee for each buy and sell transaction.

You must buy before you can sell the stock, but you can make as many transactions as you like.

For example, given `[1, 3, 2, 8, 4, 10]` and `fee = 2`, you should return `9`, since you could buy the stock at `$1`, and sell at `$8`, and then buy it at `$4` and sell it at `$10`. Since we did two transactions, there is a `$4` fee, so we have `7 + 6 = 13` profit minus `$4` of fees.

Answer 193





```

def get_max_profit(prices, fee, reserve=0, buyable=True):
    if not prices:
        return reserve

    price_offset = -prices[0] - fee if buyable else prices[0]
    return max(
        get_max_profit(prices[1:], fee, reserve, buyable),
        get_max_profit(prices[1:], fee, reserve + price_offset, not buyable)
    )



# Tests
assert get_max_profit([1, 3, 2, 8, 4, 10], 2) == 9
assert get_max_profit([1, 3, 2, 1, 4, 10], 2) == 7

```



Question 194

This problem was asked by Facebook.

Suppose you are given two lists of n points, one list `p1, p2, ..., pn` on the line `y = 0` and the other list `q1, q2, ..., qn` on the line `y = 1`. Imagine a set of `n` line segments connecting each point `pi` to `qi`. Write an algorithm to determine how many pairs of the line segments intersect.

Answer 194





```

def get_intersections(parr, qarr):
    segments = list(zip(parr, qarr))

    count = 0
    for i in range(len(segments)):
        for k in range(i):
            p1, p2 = segments[i], segments[k]
            if (p1[0] < p2[0] and p1[1] > p2[1]) or \
                    (p1[0] > p2[0] and p1[1] < p2[1]):
                count += 1

    return count


# Tests
assert get_intersections([1, 4, 5], [4, 2, 3]) == 2
assert get_intersections([1, 4, 5], [2, 3, 4]) == 0

```



Question 195

This problem was asked by Google.

Let `M` be an `N` by `N` matrix in which every row and every column is sorted. No two elements of `M` are equal.

Given `i1`, `j1`, `i2`, and `j2`, compute the number of elements of `M` smaller than `M[i1, j1]` and larger than `M[i2, j2]`.

Answer 195





```

def get_num_betn(matrix, i1, j1, i2, j2):
    num_1, num_2 = matrix[i1][j1], matrix[i2][j2]
    sm, lg = (num_1, num_2) if num_1 < num_2 else (num_2, num_1)
    count = 0
    for row in matrix:
        count += len([x for x in row if (x > sm and x < lg)])

    return count


# Tests
matrix = [
    [1, 2, 3, 4],
    [5, 8, 9, 13],
    [6, 10, 12, 14],
    [7, 11, 15, 16]
]
assert get_num_betn(matrix, 1, 3, 3, 1) == 1

matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [10, 11, 12, 13],
    [20, 21, 22, 23]
]
assert get_num_betn(matrix, 1, 0, 3, 3) == 10

```



Question 196

This problem was asked by Apple.

Given the root of a binary tree, find the most frequent subtree sum. The subtree sum of a node is the sum of all values under a node, including the node itself.

For example, given the following tree:

```
  5
 / \
2  -5
```

Return `2` as it occurs twice: once as the left leaf, and once as the sum of `2 + 5 - 5.`

Answer 196





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def get_freq_tree_sum(root, counts):
    if not root:
        return 0

    tree_sum = root.val + \
        get_freq_tree_sum(root.left, counts) + \
        get_freq_tree_sum(root.right, counts)

    if not tree_sum in counts:
        counts[tree_sum] = 0
    counts[tree_sum] += 1

    return tree_sum


def get_freq_tree_sum_helper(root):
    counts = dict()
    get_freq_tree_sum(root, counts)

    return max(counts.items(), key=lambda x: x[1])[0]


# Tests
root = Node(5)
root.left = Node(2)
root.right = Node(-5)
assert get_freq_tree_sum_helper(root) == 2

```



Question 197

This problem was asked by Amazon.

Given an array and a number `k` that's smaller than the length of the array, rotate the array to the right `k` elements in-place.

Answer 197





```

def rotate_index(arr, k, src_ind, src_num, count=0):
    if count == len(arr):
        return

    des_ind = (src_ind + k) % len(arr)
    des_num = arr[des_ind]

    arr[des_ind] = src_num

    rotate_index(arr, k, des_ind, des_num, count + 1)


def rotate_k(arr, k):
    if k < 1:
        return arr

    start = 0
    rotate_index(arr, k, start, arr[start])


# Tests
arr = [1, 2, 3, 4, 5]
rotate_k(arr, 2)
assert arr == [4, 5, 1, 2, 3]
rotate_k(arr, 2)
assert arr == [2, 3, 4, 5, 1]
rotate_k(arr, 4)
assert arr == [3, 4, 5, 1, 2]

```



Question 198


This problem was asked by Google.

Given a set of distinct positive integers, find the largest subset such that every pair of elements in the subset `(i, j)` satisfies either `i % j = 0` or `j % i = 0`.

For example, given the set `[3, 5, 10, 20, 21]`, you should return `[5, 10, 20]`. Given `[1, 3, 6, 24]`, return `[1, 3, 6, 24]`.

Answer 198





```

def get_largest_subset(arr, prev_num=1, curr_ind=0, prev_subset=[]):
    if curr_ind == len(arr):
        return prev_subset

    curr_num = arr[curr_ind]

    alt_0 = get_largest_subset(arr, prev_num, curr_ind + 1, prev_subset)
    if curr_num % prev_num == 0:
        alt_1 = get_largest_subset(
            arr, curr_num, curr_ind + 1, prev_subset + [curr_num])
        return alt_1 if len(alt_1) > len(alt_0) else alt_0

    return alt_0


def get_largest_subset_helper(arr):
    arr.sort()
    return get_largest_subset(arr)


# Tests
assert get_largest_subset([]) == []
assert get_largest_subset([2]) == [2]
assert get_largest_subset([2, 3]) == [3]
assert get_largest_subset([3, 5, 10, 20, 21]) == [5, 10, 20]
assert get_largest_subset([1, 3, 6, 24]) == [1, 3, 6, 24]
assert get_largest_subset([3, 9, 15, 30]) == [3, 15, 30]

```



Question 199

This problem was asked by Facebook.

Given a string of parentheses, find the balanced string that can be produced from it using the minimum number of insertions and deletions. If there are multiple solutions, return any of them.

For example, given `"(()"`, you could return `"(())"`. Given `"))()("`, you could return `"()()()()"`.

Answer 199





```

def correct_braces(braces):
    # get rid of all initial closing braces
    i = 0
    while i < len(braces) and braces[i] == ')':
        i += 1
    braces = braces[i:]

    # base case for recursion
    if not braces:
        return ''

    # check for the first balanced group of braces
    open_braces = 0
    for i, brace in enumerate(braces):
        if brace == '(':
            open_braces += 1
        elif brace == ')':
            open_braces -= 1

        if not open_braces:
            break

    # if there is one, process the rest separately, else truncate the excess opening braces
    return braces[open_braces:] if open_braces else braces[:i+1] + correct_braces(braces[i+1:])


# Tests
assert correct_braces("()(()") == "()()"
assert correct_braces("()(()))") == "()(())"
assert correct_braces(")(())") == "(())"
assert correct_braces("())(") == "()"

```



Question 200

This problem was asked by Microsoft.

Let `X` be a set of `n` intervals on the real line. We say that a set of points `P` "stabs" `X` if every interval in `X` contains at least one point in `P`. Compute the smallest set of points that stabs `X`.

For example, given the intervals `[(1, 4), (4, 5), (7, 9), (9, 12)]`, you should return `[4, 9]`.

Answer 200





```

def get_stab_points(intervals):
    starts, ends = zip(*intervals)
    return (min(ends), max(starts))


# Tests
assert get_stab_points([(1, 4), (4, 5), (7, 9), (9, 12)]) == (4, 9)
assert get_stab_points([(1, 4), (-2, 6), (4, 5), (7, 9), (9, 12)]) == (4, 9)
assert get_stab_points([(1, 4), (-2, 0), (4, 5), (7, 9)]) == (0, 7)

```



Question 201

This problem was asked by Google.

You are given an array of arrays of integers, where each array corresponds to a row in a triangle of numbers. For example, `[[1], [2, 3], [1, 5, 1]]` represents the triangle:

```
  1
 2 3
1 5 1
```

We define a path in the triangle to start at the top and go down one row at a time to an adjacent value, eventually ending with an entry on the bottom row. For example, `1 -> 3 -> 5`. The weight of the path is the sum of the entries.

Write a program that returns the weight of the maximum weight path.

Answer 201





```

def get_max_path(triangle, index, level, path, path_val):
    if level == len(triangle):
        return path, path_val

    ind_a, ind_b = index, index + 1
    val_a, val_b = triangle[level][ind_a], triangle[level][ind_b]

    path_a, path_val_a = \
        get_max_path(triangle, ind_a, level + 1,
                     path + [val_a], path_val + val_a)
    path_b, path_val_b = \
        get_max_path(triangle, ind_b, level + 1,
                     path + [val_b], path_val + val_b)

    return (path_a, path_val_a) if path_val_a > path_val_b else (path_b, path_val_b)


def get_max_path_helper(triangle):
    return get_max_path(triangle, index=0, level=1,
                        path=[triangle[0][0]], path_val=triangle[0][0])[0]


# Tests
assert get_max_path_helper([[1], [2, 3], [1, 5, 1]]) == [1, 3, 5]
assert get_max_path_helper([[1], [2, 3], [7, 5, 1]]) == [1, 2, 7]

```



Question 202

This problem was asked by Palantir.

Write a program that checks whether an integer is a palindrome. For example, `121` is a palindrome, as well as `888`. `678` is not a palindrome. Do not convert the integer into a string.

Answer 202





```

DEC_FACT = 10


def is_palindrome(num, size):
    if size == 0 or size == 1:
        return True

    fdig_factor = DEC_FACT ** (size - 1)
    fdig = num // fdig_factor
    ldig = num % DEC_FACT

    if fdig != ldig:
        return False

    new_num = (num - (fdig * fdig_factor)) // DEC_FACT
    return is_palindrome(new_num, size - 2)


def is_palindrome_helper(num):
    size = 0
    num_cp = num
    while num_cp:
        num_cp = num_cp // DEC_FACT
        size += 1

    return is_palindrome(num, size)



# Tests
assert is_palindrome_helper(121)
assert is_palindrome_helper(888)
assert not is_palindrome_helper(678)
assert not is_palindrome_helper(1678)
assert is_palindrome_helper(1661)

```



Question 203

This problem was asked by Uber.

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element in `O(log N)` time. You may assume the array does not contain duplicates.

For example, given `[5, 7, 10, 3, 4]`, return `3`.

Answer 203





```

def get_smallest(arr, start, end):
    mid = start + ((end - start) // 2)

    if arr[start] <= arr[mid]:
        if arr[end] < arr[mid]:
            return get_smallest(arr, mid + 1, end)
        else:
            return arr[start]
    elif arr[start] >= arr[mid]:
        if arr[end] > arr[mid]:
            return get_smallest(arr, start, end)
        else:
            return arr[end]


def get_smallest_helper(arr):
    smallest = get_smallest(arr, 0, len(arr) - 1)
    return smallest


# Tests
assert get_smallest_helper([5, 7, 10, 3, 4]) == 3
assert get_smallest_helper([4, 5, 7, 10, 3]) == 3
assert get_smallest_helper([3, 4, 5, 7, 10]) == 3

```



Question 204

This problem was asked by Amazon.

Given a complete binary tree, count the number of nodes in faster than `O(n)` time. Recall that a complete binary tree has every level filled except the last, and the nodes in the last level are filled starting from the left.

Answer 204





```

class Node:
    def __init__(self):
        self.left = None
        self.right = None


def count_nodes(root, lspine=0, rspine=0):
    if not root:
        return 0

    if not lspine:
        node = root
        while node:
            node = node.left
            lspine += 1
    if not rspine:
        node = root
        while node:
            node = node.right
            rspine += 1

    if lspine == rspine:
        return 2**lspine - 1

    return 1 + \
        count_nodes(root.left, lspine=lspine-1) + \
        count_nodes(root.right, rspine=rspine-1)


# Tests
a = Node()
b = Node()
c = Node()
a.left = b
a.right = c
assert count_nodes(a) == 3
d = Node()
b.left = d
assert count_nodes(a) == 4
e = Node()
b.right = e
assert count_nodes(a) == 5
f = Node()
c.left = f
assert count_nodes(a) == 6

```



Question 205

This problem was asked by IBM.

Given an integer, find the next permutation of it in absolute order. For example, given `48975`, the next permutation would be `49578`.

Answer 205





```

from bisect import bisect


def get_next_perm(num):
    num_str = str(num)

    # if the number is already maxed out, return it
    max_val_str = "".join(sorted(num_str, reverse=True))
    if max_val_str == num_str:
        return num

    # find the point n the number at which there is
    # a chance to increase the number by changing order
    right_nums = list()
    num_to_replace = None
    for i in range(len(num_str) - 2, -1, -1):
        right_nums.append(num_str[i + 1])
        if num_str[i] < num_str[i + 1]:
            num_to_replace = num_str[i]
            break

    # identify the replacement of the digit to be moved
    rep_index = bisect(right_nums, num_to_replace)
    replacement = right_nums[rep_index]

    # replace digit
    right_nums[rep_index] = num_to_replace
    leftover_nums = num_str[:i]

    # contruct new number
    final_str = "{}{}{}".format(leftover_nums, replacement,
                                "".join(sorted(right_nums)))

    return int(final_str)


# Tests
assert get_next_perm(98754) == 98754
assert get_next_perm(48975) == 49578
assert get_next_perm(48759) == 48795
assert get_next_perm(49875) == 54789
assert get_next_perm(408975) == 409578

```



Question 206

This problem was asked by Twitter.

A permutation can be specified by an array `P`, where `P[i]` represents the location of the element at `i` in the permutation. For example, `[2, 1, 0]` represents the permutation where elements at the index `0` and `2` are swapped.

Given an array and a permutation, apply the permutation to the array. For example, given the array `["a", "b", "c"]` and the permutation `[2, 1, 0]`, return `["c", "b", "a"]`.

Answer 206





```

def permute(arr, perms):
    evicted = dict()
    for i, (num, new_pos) in enumerate(zip(arr, perms)):
        if i in evicted:
            num = evicted[i]
            del evicted[i]

        if new_pos > i:
            evicted[new_pos] = arr[new_pos]

        arr[new_pos] = num

    return arr


# Tests
assert permute(['a', 'b', 'c'], [2, 1, 0]) == ['c', 'b', 'a']
assert permute(['a', 'b', 'c', 'd'], [2, 1, 0, 3]) == ['c', 'b', 'a', 'd']
assert permute(['a', 'b', 'c', 'd'], [3, 0, 1, 2]) == ['b', 'c', 'd', 'a']

```



Question 207

This problem was asked by Dropbox.

Given an undirected graph `G`, check whether it is bipartite. Recall that a graph is bipartite if its vertices can be divided into two independent sets, `U` and `V`, such that no edge connects vertices of the same set.

Answer 207





```

class Node:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return str(self.val)


class Graph:
    def __init__(self):
        self.nodes, self.edges = set(), dict()
        self.set_1, self.set_2 = set(), set()

    def is_bipartite(self):
        sorted_nodes = sorted(self.edges.items(),
                              key=lambda x: len(x[1]), reverse=True)
        for node, _ in sorted_nodes:
            if node in self.set_2:
                continue
            self.set_1.add(node)

            if self.edges[node]:
                for other_node in self.edges[node]:
                    self.set_2.add(other_node)

        for node in self.set_2:
            if self.edges[node]:
                for other_node in self.edges[node]:
                    if other_node in self.set_2:
                        return False

        return True


# Tests
g = Graph()
a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
g.nodes = set([a, b, c, d, e, f])

g.edges[a] = set([d])
g.edges[b] = set([d, e, f])
g.edges[c] = set([f])
g.edges[d] = set([a, b])
g.edges[e] = set([b])
g.edges[f] = set([b, c])
assert g.is_bipartite()

g.edges = dict()
g.nodes = set([a, b, c, d, e, f])
g.edges[a] = set([d])
g.edges[b] = set([d, e, f])
g.edges[c] = set([f])
g.edges[d] = set([a, b])
g.edges[e] = set([b, f])
g.edges[f] = set([b, c, e])
assert not g.is_bipartite()

```



Question 208

This problem was asked by LinkedIn.

Given a linked list of numbers and a pivot `k`, partition the linked list so that all nodes less than `k` come before nodes greater than or equal to `k`.

For example, given the linked list `5 -> 1 -> 8 -> 0 -> 3` and `k = 3`, the solution could be `1 -> 0 -> 5 -> 8 -> 3`.

Answer 208





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def partition(llist, k):
    head = llist
    prev, curr = head, head.next

    while curr:
        if curr.val < k:
            prev.next = curr.next
            curr.next = head
            head = curr
            curr = prev.next
        else:
            prev = curr
            curr = curr.next

    return head


# Tests
assert get_list(partition(get_nodes([5, 1, 8, 0, 3]), 3)) == [0, 1, 5, 8, 3]

```



Question 209

This problem was asked by YouTube.

Write a program that computes the length of the longest common subsequence of three given strings. For example, given "epidemiologist", "refrigeration", and "supercalifragilisticexpialodocious", it should return `5`, since the longest common subsequence is "eieio".

Answer 209





```

# This solution generalizes to n strings

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def get_lcs(strings, context, indices):
    lcs_len = len(context)
    for letter in ALPHABET:
        new_indices = list()
        for j, string in enumerate(strings):
            index = string.find(letter, indices[j] + 1)
            if index == -1:
                break
            new_indices.append(index)
        if len(new_indices) == 3:
            length_cs = get_lcs(strings, context + letter, new_indices)
            if length_cs > lcs_len:
                lcs_len = length_cs

    return lcs_len


def get_lcs_helper(strings):
    return get_lcs(strings, "", [-1]*len(strings))


# Tests
assert get_lcs_helper(["epidemiologist", "refrigeration",
                       "supercalifragilisticexpialodocious"]) == 5

```



Question 210

This problem was asked by Apple.

A Collatz sequence in mathematics can be defined as follows. Starting with any positive integer:
* If `n` is even, the next number in the sequence is `n / 2`
* If `n` is odd, the next number in the sequence is `3n + 1`
It is conjectured that every such sequence eventually reaches the number `1`. Test this conjecture.

Bonus: What input `n <= 1000000` gives the longest sequence?

Answer 210





```

import sys
from random import randint

cache = dict()


def get_collatz_seq(num, prev=list()):
    prev.append(num)

    if num in cache:
        return prev + cache[num]

    if num == 1:
        return prev

    if num % 2 == 0:
        num //= 2
    else:
        num = 3*num + 1

    return get_collatz_seq(num, prev)


# Tests
experiments = 10
for _ in range(experiments):
    num = randint(1, sys.maxsize)
    cs = get_collatz_seq(num)
    assert cs[-1] == 1


def get_longest_collatz(limit):
    longest_cs, length = None, -1
    for num in range(1, limit + 1):
        if num in cache:
            cs = cache[num]
        else:
            cs = get_collatz_seq(num)
            cache[num] = cs
        if len(cs) > length:
            length = len(cs)
            longest_cs = cs

    return longest_cs


print(len(get_longest_collatz(100)))

```



Question 211

This problem was asked by Microsoft.

Given a string and a pattern, find the starting indices of all occurrences of the pattern in the string. For example, given the string "abracadabra" and the pattern "abr", you should return `[0, 7]`.

Answer 211





```

def get_occurrences(string, pattern):
    sl, pl = len(string), len(pattern)
    occurrences = list()

    for i in range(sl - pl + 1):
        if string[i:i+pl] == pattern:
            occurrences.append(i)

    return occurrences


# Tests
assert get_occurrences("abracadabra", "abr") == [0, 7]
assert not get_occurrences("abracadabra", "xyz")
assert not get_occurrences("abr", "abracadabra")
assert get_occurrences("aaaa", "aa") == [0, 1, 2]

```



Question 212

This problem was asked by Dropbox.

Spreadsheets often use this alphabetical encoding for its columns: "A", "B", "C", ..., "AA", "AB", ..., "ZZ", "AAA", "AAB", ....

Given a column number, return its alphabetical column id. For example, given `1`, return "A". Given `27`, return "AA".

Answer 212





```

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_alpha_encoding(num):
    num_chars = 1
    min_range, max_range = 1, 26
    while num > max_range:
        num_chars += 1
        min_range = max_range
        max_range += len(alphabet) ** num_chars

    chars = list()
    for _ in range(num_chars):
        interval = ((max_range - min_range + 1) // len(alphabet))
        char_pos = 0
        prev, curr = min_range, min_range + interval
        while num >= curr:
            char_pos += 1
            prev = curr
            curr = prev + interval
        chars.append(alphabet[char_pos])
        num -= prev
        min_range, max_range = prev, curr

    return "".join(chars)


# Tests
assert get_alpha_encoding(1) == "A"
assert get_alpha_encoding(20) == "T"
assert get_alpha_encoding(27) == "AA"

```



Question 213

This problem was asked by Snapchat.

Given a string of digits, generate all possible valid IP address combinations.

IP addresses must follow the format `A.B.C.D`, where `A`, `B`, `C`, and `D` are numbers between `0` and `255`. Zero-prefixed numbers, such as `01` and `065`, are not allowed, except for `0` itself.

For example, given "2542540123", you should return `['254.25.40.123', '254.254.0.123']`.

Answer 213





```

def generate_valid_ips(string, curr, all_ips):
    if len(curr) > 4 or (len(curr) < 4 and not string):
        return
    elif len(curr) == 4 and not string:
        all_ips.add(".".join(curr))
        return

    def recurse(index):
        generate_valid_ips(string[index:], curr + [string[0:index]], all_ips)

    recurse(1)
    first = int(string[0])
    if first and len(string) > 1:
        recurse(2)
        if len(string) > 2 and first < 3:
            recurse(3)


def generate_valid_ip_helper(string):
    all_ips = set()
    generate_valid_ips(string, list(), all_ips)
    return all_ips


# Tests
assert generate_valid_ip_helper("2542540123") == \
    {'254.25.40.123', '254.254.0.123'}
assert generate_valid_ip_helper("0000") == \
    {'0.0.0.0'}
assert generate_valid_ip_helper("255255255255") == \
    {'255.255.255.255'}
assert generate_valid_ip_helper("100100110") == \
    {'100.10.0.110', '10.0.100.110', '100.100.11.0', '100.100.1.10'}

```



Question 214

This problem was asked by Stripe.

Given an integer `n`, return the length of the longest consecutive run of `1`s in its binary representation.

For example, given `156`, you should return `3`.

Answer 214





```

def get_lcos(num):
    current = longest = 0

    def reset_current():
        nonlocal current, longest
        if current > longest:
            longest = current
        current = 0

    while num:
        if num % 2:
            current += 1
        else:
            reset_current()
        num = num >> 1

    reset_current()

    return longest


# Tests
assert get_lcos(0) == 0
assert get_lcos(4) == 1
assert get_lcos(6) == 2
assert get_lcos(15) == 4
assert get_lcos(21) == 1
assert get_lcos(156) == 3

```



Question 215

This problem was asked by Yelp.

The horizontal distance of a binary tree node describes how far left or right the node will be when the tree is printed out.

More rigorously, we can define it as follows:
* The horizontal distance of the root is `0`.
* The horizontal distance of a left child is `hd(parent) - 1`.
* The horizontal distance of a right child is `hd(parent) + 1`.

For example, for the following tree, `hd(1) = -2`, and `hd(6) = 0`.

```
             5
          /     \
        3         7
      /  \      /   \
    1     4    6     9
   /                /
  0                8
```
  
The bottom view of a tree, then, consists of the lowest node at each horizontal distance. If there are two nodes at the same depth and horizontal distance, either is acceptable.

For this tree, for example, the bottom view could be `[0, 1, 3, 6, 8, 9]`.

Given the root to a binary tree, return its bottom view.

Answer 215





```

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.depth = None

    def __repr__(self):
        return "Node[val={}]".format(self.data)


def compute_bottom_view(root, depths, depth, width):
    root.depth = depth

    if width not in depths or depths[width].depth < depth:
        depths[width] = root

    if root.right:
        compute_bottom_view(root.right, depths, depth + 1, width + 1)
    if root.left:
        compute_bottom_view(root.left, depths, depth + 1, width - 1)


def get_bottom_view(root):
    depths = dict()
    compute_bottom_view(root, depths, 0, 0)

    sorted_items = sorted(depths.items(), key=lambda x: x[0])
    return [x[1].data for x in sorted_items]


# Tests
root = Node(5)
root.left = Node(3)
root.left.left = Node(1)
root.left.left.left = Node(0)
root.left.right = Node(4)
root.right = Node(7)
root.right.left = Node(6)
root.right.right = Node(9)
root.right.right.left = Node(8)
assert get_bottom_view(root) == [0, 1, 3, 6, 8, 9]

```



Question 216

This problem was asked by Facebook.

Given a number in Roman numeral format, convert it to decimal.

The values of Roman numerals are as follows:
```
{
    'M': 1000,
    'D': 500,
    'C': 100,
    'L': 50,
    'X': 10,
    'V': 5,
    'I': 1
}
```

In addition, note that the Roman numeral system uses subtractive notation for numbers such as `IV` and `XL`.

For the input `XIV`, for instance, you should return `14`.

Answer 216





```

values = {
    'M': 1000,
    'D': 500,
    'C': 100,
    'L': 50,
    'X': 10,
    'V': 5,
    'I': 1
}


def convert_roman_to_decimal(roman):
    if not roman:
        return 0

    pchar = roman[-1]
    decimal = values[pchar]
    for char in reversed(roman[:-1]):
        decimal += values[char] * (-1 if values[char] < values[pchar] else 1)
        pchar = char

    return decimal


# Tests
assert convert_roman_to_decimal("I") == 1
assert convert_roman_to_decimal("IV") == 4
assert convert_roman_to_decimal("XL") == 40
assert convert_roman_to_decimal("XIV") == 14

```



Question 217

This problem was asked by Oracle.

We say a number is sparse if there are no adjacent ones in its binary representation. For example, `21` (`10101`) is sparse, but `22` (`10110`) is not. For a given input `N`, find the smallest sparse number greater than or equal to `N`.

Do this in faster than `O(N log N)` time.

Answer 217





```

def get_next_sparse(num):
    str_bin = str(bin(num))[2:]

    new_str_bin = ""
    prev_digit = None
    flag = False
    for i, digit in enumerate(str_bin):
        if digit == '1' and prev_digit == '1':
            flag = True

        if flag:
            new_str_bin += '0' * (len(str_bin) - i)
            break
        else:
            new_str_bin += digit
        prev_digit = digit

    if flag:
        if new_str_bin[0] == '1':
            new_str_bin = '10' + new_str_bin[1:]
        else:
            new_str_bin = '1' + new_str_bin

    new_num = int(new_str_bin, base=2)

    return new_num


# Tests
assert get_next_sparse(21) == 21
assert get_next_sparse(25) == 32
assert get_next_sparse(255) == 256

```



Question 218

This problem was asked by Yahoo.

Write an algorithm that computes the reversal of a directed graph. For example, if a graph consists of `A -> B -> C`, it should become `A <- B <- C`.

Answer 218





```

class Node:
    def __init__(self, iden):
        self.iden = iden

    def __hash__(self):
        return hash(self.iden)

    def __eq__(self, other):
        return self.iden == other.iden

    def __repr__(self):
        return str(self.iden)


class Edge:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __hash__(self):
        return hash((self.src, self.tgt))

    def __eq__(self, other):
        return self.src == other.src and self.tgt == other.tgt

    def __repr__(self):
        return "{}->{}".format(self.src, self.tgt)

    def reverse(self):
        tmp_node = self.src
        self.src = self.tgt
        self.tgt = tmp_node


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.add(node)

    def add_edge(self, src_node, tgt_node):
        self.edges.add(Edge(src_node, tgt_node))

    def reverse_edges(self):
        self.edges = [Edge(x.tgt, x.src) for x in self.edges]

    def get_edges(self):
        return self.edges


# Tests
g = Graph()
a = Node('a')
b = Node('b')
c = Node('c')

g.add_node(a)
g.add_node(b)
g.add_node(c)
g.add_edge(a, b)
g.add_edge(b, c)
edges = g.get_edges()
assert Edge(a, b) in edges and Edge(b, c) in edges and len(edges) == 2

g.reverse_edges()
edges = g.get_edges()
assert Edge(b, a) in edges and Edge(c, b) in edges and len(edges) == 2

```



Question 219

This problem was asked by Salesforce.

Connect 4 is a game where opponents take turns dropping red or black discs into a `7 x 6` vertically suspended grid. The game ends either when one player creates a line of four consecutive discs of their color (horizontally, vertically, or diagonally), or when there are no more spots left in the grid.

Design and implement Connect 4.

Answer 219





```

from enum import Enum

ROWS = 6
COLS = 7


class Color(Enum):
    RED = 1
    BLACK = 2


class Board:
    def __init__(self):
        self.grid = list()
        for _ in range(COLS):
            col = list()
            for _ in range(ROWS):
                col.append(None)
            self.grid.append(col)
        self.occupancy = [0] * COLS


class IllegalMove(Exception):
    pass


def play_piece(board, played_column, played_color):
    if board.occupancy[played_column] == 6:
        raise IllegalMove("Illegal move in this column")

    played_row = board.occupancy[played_column]
    board.grid[played_column][played_row] = played_color
    board.occupancy[played_column] += 1

    # check vertical
    consecutive = 0
    if len(board.grid[played_column]) > 4:
        for color in board.grid[played_column]:
            if color == played_color:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive == 4:
                return True

    # check horizontal
    consecutive = 0
    for i in range(COLS):
        color = board.grid[i][played_row]
        if color == played_color:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive == 4:
            return True

    # check positive-slope diagonal
    consecutive = 0
    offset = min(played_column, played_row)
    col = played_column - offset
    row = played_row - offset
    while col < COLS and row < ROWS:
        color = board.grid[col][row]
        if color == played_color:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive == 4:
            return True

    # check negative-slope diagonal
    consecutive = 0
    col = played_column + offset
    row = played_row - offset
    while col > 0 and row < ROWS:
        color = board.grid[col][row]
        if color == played_color:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive == 4:
            return True

    return False


def play_game():

    board = Board()
    print("New board initialized")

    turn = 0
    players = [Color.RED, Color.BLACK]
    while True:
        player = players[turn % (len(players))]
        print("{}'s turn.".format(player))
        col_num = int(input("Enter a column number: "))

        try:
            won = play_piece(board, col_num, player)
        except IllegalMove as e:
            print(e)
            continue

        if won:
            print("{} wins".format(player))
            break

        turn += 1


play_game()

```



Question 220

This problem was asked by Square.

In front of you is a row of N coins, with values `v_1, v_2, ..., v_n`.

You are asked to play the following game. You and an opponent take turns choosing either the first or last coin from the row, removing it from the row, and receiving the value of the coin.

Write a program that returns the maximum amount of money you can win with certainty, if you move first, assuming your opponent plays optimally.

Answer 220





```

def get_max_possible(coins, amount=0, turn=True):
    if not coins:
        return amount

    if turn:
        alt_1 = get_max_possible(coins[1:], amount + coins[0], False)
        alt_2 = get_max_possible(coins[:-1], amount + coins[-1], False)
        return max(alt_1, alt_2)

    first, last = coins[0], coins[-1]
    if first > last:
        coins = coins[1:]
    else:
        coins = coins[:-1]

    return get_max_possible(coins, amount, True)


# Test
assert get_max_possible([1, 2, 3, 4, 5]) == 9

```



Question 221

This problem was asked by Zillow.

Let's define a "sevenish" number to be one which is either a power of `7`, or the sum of unique powers of `7`. The first few sevenish numbers are `1, 7, 8, 49`, and so on. Create an algorithm to find the `n`th sevenish number.

Answer 221





```

def get_nth_sevenish(n):
    if n < 1:
        raise Exception("Invalid value for 'n'")

    power = 0
    sevenish_nums = list()
    while len(sevenish_nums) < n:
        num = 7 ** power
        new_sevenish_nums = [num]
        for old in sevenish_nums:
            if len(sevenish_nums) + len(new_sevenish_nums) == n:
                return new_sevenish_nums[-1]
            new_sevenish_nums.append(num + old)

        sevenish_nums += new_sevenish_nums
        power += 1

    return sevenish_nums[-1]


# Tests
assert get_nth_sevenish(1) == 1
assert get_nth_sevenish(2) == 7
assert get_nth_sevenish(3) == 8
assert get_nth_sevenish(10) == 350

```



Question 222

This problem was asked by Quora.

Given an absolute pathname that may have `.` or `..` as part of it, return the shortest standardized path.

For example, given `/usr/bin/../bin/./scripts/../`, return `/usr/bin/`.

Answer 222





```

PATH_SEPARATOR = "/"


def shorten_path(path):
    stack = list()
    dirs = path.split(PATH_SEPARATOR)

    for dir_name in dirs:
        if dir_name == ".":
            continue
        elif dir_name == "..":
            stack.pop()
        else:
            stack.append(dir_name)

    spath = PATH_SEPARATOR.join(stack)
    return spath


# Tests
assert shorten_path("/usr/bin/../bin/./scripts/../") == "/usr/bin/"

```



Question 223

This problem was asked by Palantir.

Typically, an implementation of in-order traversal of a binary tree has `O(h)` space complexity, where `h` is the height of the tree. Write a program to compute the in-order traversal of a binary tree using `O(1)` space.

Answer 223





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return "{}=[l={}, r={}]".format(self.val, self.left, self.right)


def add_reverse_links(root, parent=None):
    root.parent = parent

    if root.left:
        add_reverse_links(root.left, root)
    if root.right:
        add_reverse_links(root.right, root)


def print_inorder(root):
    if root.left:
        print_inorder(root.left)

    print(root.val)

    if root.right:
        print_inorder(root.right)


# Tests
a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
g = Node('g')
h = Node('h')

a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
c.right = g
d.right = h
# print(a)

add_reverse_links(a)
assert h.parent == d
assert g.parent == c

print_inorder(a)

```



Question 224

This problem was asked by Amazon.

Given a sorted array, find the smallest positive integer that is not the sum of a subset of the array.

For example, for the input `[1, 2, 3, 10]`, you should return `7`.

Do this in `O(N)` time.

Answer 224





```

def findSmallest(arr):
    res = 1
    for num in arr:
        if num > res:
            break
        res += num
    return res


# Tests
assert findSmallest([1, 2, 3, 10]) == 7
assert findSmallest([1, 2, 10]) == 4
assert findSmallest([0, 10]) == 1

```



Question 225

This problem was asked by Bloomberg.

There are `N` prisoners standing in a circle, waiting to be executed. The executions are carried out starting with the `k`th person, and removing every successive `k`th person going clockwise until there is no one left.

Given `N` and `k`, write an algorithm to determine where a prisoner should stand in order to be the last survivor.

For example, if `N = 5` and `k = 2`, the order of executions would be `[2, 4, 1, 5, 3]`, so you should return `3`.

Bonus: Find an `O(log N)` solution if `k = 2`.

Answer 225





```

def last_exec(n, k):
    last_exec = None
    next_exec_index = 0

    prisoners = list(range(1, n + 1))

    while prisoners:
        next_exec_index = (next_exec_index + k - 1) % len(prisoners)
        last_exec = prisoners[next_exec_index]
        prisoners = prisoners[:next_exec_index] + \
            prisoners[next_exec_index + 1:]

    return last_exec


# Tests
assert last_exec(5, 2) == 3
assert last_exec(3, 2) == 3
assert last_exec(5, 3) == 4

```



Question 226

This problem was asked by Airbnb.

You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.

For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

Answer 226





```

def update_letter_order(sorted_words, letters):
    order = list()
    new_words = dict()
    prev_char = None
    for word in sorted_words:
        if word:
            char = word[0]
            if char != prev_char:
                order.append(char)
            if char not in new_words:
                new_words[char] = list()
            new_words[char].append(word[1:])
            prev_char = char

    for index, char in enumerate(order):
        letters[char] |= set(order[index + 1:])

    for char in new_words:
        update_letter_order(new_words[char], letters)


def find_path(letters, start, path, length):
    if len(path) == length:
        return path

    if not letters[start]:
        return None

    for next_start in letters[start]:
        new_path = find_path(letters, next_start, path + [next_start], length)
        if new_path:
            return new_path


def update_letter_order_helper(sorted_words):
    letters = dict()
    for word in sorted_words:
        for letter in word:
            if letter not in letters:
                letters[letter] = set()

    update_letter_order(sorted_words, letters)

    max_children = max([len(x) for x in letters.values()])
    potential_heads = [x for x in letters if len(letters[x]) == max_children]

    path = None
    for head in potential_heads:
        path = find_path(letters, head, path=[head], length=len(letters))
        if path:
            break

    return path


# Tests
assert update_letter_order_helper(
    ['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']) == ['x', 'z', 'w', 'y']

```



Question 227

This problem was asked by Facebook.

Boggle is a game played on a `4 x 4` grid of letters. The goal is to find as many words as possible that can be formed by a sequence of adjacent letters in the grid, using each cell at most once. Given a game board and a dictionary of valid words, implement a Boggle solver.

Answer 227





```

NUM_ROWS, NUM_COLS = 4, 4
WORD_END_CHAR = '0'


class Trie:
    def __init__(self):
        self.size = 0
        self.letter_map = dict()

    def __repr__(self):
        return str(self.letter_map)

    def add_word(self, word):
        if not word:
            self.letter_map[WORD_END_CHAR] = None
            return
        letter = word[0]

        sub_trie = None
        if letter in self.letter_map:
            sub_trie = self.letter_map[letter]
        else:
            sub_trie = Trie()
            self.letter_map[letter] = sub_trie

        self.size += 1
        sub_trie.add_word(word[1:])


class Coordinate():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return "C(x={};y={})".format(self.x, self.y)


def get_adjacent_coords(x, y):
    adj_coords = set()

    red_x, red_y = max(0, x - 1), max(0, y - 1)
    inc_x, inc_y = min(NUM_ROWS - 1, x + 1), min(NUM_COLS - 1, y + 1)

    adj_coords.add(Coordinate(x, red_y))
    adj_coords.add(Coordinate(x, inc_y))
    adj_coords.add(Coordinate(red_x, y))
    adj_coords.add(Coordinate(red_x, red_y))
    adj_coords.add(Coordinate(red_x, inc_y))
    adj_coords.add(Coordinate(inc_x, y))
    adj_coords.add(Coordinate(inc_x, red_y))
    adj_coords.add(Coordinate(inc_x, inc_y))

    return adj_coords


def search_for_words(coord, dtrie, seen, board, sol_words, current):
    letter = board[coord.x][coord.y]
    if letter not in dtrie.letter_map:
        # no possibility of creating a word
        return

    current += letter
    next_trie = dtrie.letter_map[letter]
    if WORD_END_CHAR in next_trie.letter_map:
        # a word can end here
        sol_words.add(current)
        # shouldn't return because a valid word
        # might be extended

    adj_coords = get_adjacent_coords(coord.x, coord.y)
    for adj_coord in adj_coords:
        if adj_coord in seen:
            continue

        seen_cp = seen.copy()
        seen_cp.add(adj_coord)
        search_for_words(adj_coord, next_trie, seen_cp,
                         board, sol_words, current)


def solve_boggle(board, dictionary):
    assert len(board) == NUM_ROWS
    assert len(board[0]) == NUM_COLS

    dtrie = Trie()
    for word in dictionary:
        dtrie.add_word(word)

    sol_words = set()
    for i in range(NUM_ROWS):
        for k in range(NUM_COLS):
            coord = Coordinate(i, k)
            search_for_words(coord, dtrie, {coord}, board, sol_words, "")

    return sol_words


# Tests
board = [
    ["A", "L", "B", "P"],
    ["C", "O", "E", "Y"],
    ["F", "C", "H", "P"],
    ["B", "A", "D", "A"]
]
words_in_board = {
    "AFOCAL", "CHAPEL", "CLOCHE", "DHOLE", "LOCHE", "CHOLA", "CHELA",
    "HOLEY", "FOCAL", "FOLEY", "COLEY", "COLBY", "COHAB", "COBLE", "DACHA",
    "BACHA", "BACCO", "BACCA", "BLECH", "PHOCA", "ALOHA", "ALEPH", "CHAPE",
    "BOCCA", "BOCCE", "BOCHE", "LECH", "PECH", "OCHE", "FOAL", "YECH", "OBEY",
    "YEBO", "LOCA", "LOBE", "LOCH", "HYPE", "HELO", "PELA", "HOLE", "COCA"}
words_not_in_board = {
    "DUMMY", "WORDS", "FOR", "TESTING"
}
dictionary = words_in_board | words_not_in_board

found_words = solve_boggle(board, dictionary)

assert found_words == words_in_board

```



Question 228

This problem was asked by Twitter.

Given a list of numbers, create an algorithm that arranges them in order to form the largest possible integer. For example, given `[10, 7, 76, 415]`, you should return `77641510`.

Answer 228





```

def get_largest(nums, prefix=""):
    num_dict = dict()
    for num in nums:
        str_num = str(num)
        fdig = str_num[0]
        if fdig not in num_dict:
            num_dict[fdig] = list()
        num_dict[fdig].append(str_num)

    sorted_arrs = sorted(num_dict.values(), key=lambda x: x[0], reverse=True)
    combined = list()
    for arr in sorted_arrs:
        if len(arr) == 1:
            combined.extend(arr)
            continue

        split_dict = dict()
        for num in arr:
            len_num = len(num)
            if len_num not in split_dict:
                split_dict[len_num] = list()
            split_dict[len_num].append(num)

        sorted_val_arrs = sorted(
            split_dict.values(), key=lambda x: len(x[0]))
        for val_arr in sorted_val_arrs:
            combined.extend(sorted(val_arr, reverse=True))

    return int(prefix.join(combined))


# Tests
assert get_largest([10, 7, 76, 415]) == 77641510

```



Question 229

This problem was asked by Flipkart.

Snakes and Ladders is a game played on a `10 x 10` board, the goal of which is get from square `1` to square `100`. On each turn players will roll a six-sided die and move forward a number of spaces equal to the result. If they land on a square that represents a snake or ladder, they will be transported ahead or behind, respectively, to a new square.

Find the smallest number of turns it takes to play snakes and ladders.

For convenience, here are the squares representing snakes and ladders, and their outcomes:

```
snakes = {16: 6, 48: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
```

Answer 229





```

import sys

SNAKES = {16: 6, 48: 26, 49: 11, 56: 53, 62: 19,
          64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
LADDERS = {1: 38, 4: 14, 9: 31, 21: 42,
           28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
SHORCUTS = {**SNAKES, **LADDERS}
CACHE = dict()


def get_num_turns(pos, played_turns):
    if played_turns > 100 or pos > 100:
        return sys.maxsize

    if pos == 100:
        return played_turns

    if pos in CACHE:
        return CACHE[pos]

    if pos in SHORCUTS:
        num_turns = get_num_turns(SHORCUTS[pos], played_turns)
        CACHE[pos] = num_turns
        return CACHE[pos]

    possible_num_turns = list()
    for i in range(1, 7):
        num_turns = get_num_turns(pos + i, played_turns + 1)
        CACHE[pos + i] = num_turns
        possible_num_turns.append(num_turns)

    return min(possible_num_turns)


# Tests
assert get_num_turns(0, 0) == 24

```



Question 230

This problem was asked by Goldman Sachs.

You are given `N` identical eggs and access to a building with `k` floors. Your task is to find the lowest floor that will cause an egg to break, if dropped from that floor. Once an egg breaks, it cannot be dropped again. If an egg breaks when dropped from the `x`th floor, you can assume it will also break when dropped from any floor greater than `x`.

Write an algorithm that finds the minimum number of trial drops it will take, in the worst case, to identify this floor.

For example, if `N = 1` and `k = 5`, we will need to try dropping the egg at every floor, beginning with the first, until we reach the fifth floor, so our solution will be `5`.

Answer 230





```

def get_min_drops(N, k):
    if N == 0 or N == 1 or k == 1:
        return N

    possibilities = list()
    for i in range(1, N + 1):
        possibilities.append(
            max(get_min_drops(i-1, k-1),
                get_min_drops(N-i, k))
        )

    return min(possibilities) + 1


# Tests
assert get_min_drops(20, 2) == 6
assert get_min_drops(15, 3) == 5

```



Question 231

This problem was asked by IBM.

Given a string with repeated characters, rearrange the string so that no two adjacent characters are the same. If this is not possible, return None.

For example, given "aaabbc", you could return "ababac". Given "aaab", return None.

Answer 231





```

from collections import Counter
from queue import Queue


def rearrange(string):
    c = Counter(string)
    sitems = sorted(c.items(), key=lambda x: x[1], reverse=True)

    strlen = len(string)
    if strlen % 2:
        if sitems[0][1] > (strlen // 2) + 1:
            return None
    else:
        if sitems[0][1] > (strlen // 2):
            return None

    q = Queue()
    for item in sitems:
        q.put(item)

    new_str = ""
    while not q.empty():
        item = q.get()
        new_str += item[0]
        item = (item[0], item[1] - 1)
        if item[1]:
            q.put(item)

    return new_str


# Tests
assert rearrange("aaabbc") == "abcaba"
assert rearrange("aaab") == None

```



Question 232

This problem was asked by Google.

Implement a PrefixMapSum class with the following methods:

`insert(key: str, value: int)`: Set a given key's value in the map. If the key already exists, overwrite the value.
`sum(prefix: str)`: Return the sum of all values of keys that begin with a given prefix.
For example, you should be able to run the following code:

```
mapsum.insert("columnar", 3)
assert mapsum.sum("col") == 3
```

```
mapsum.insert("column", 2)
assert mapsum.sum("col") == 5
```

Answer 232





```

class Trie:
    def __init__(self):
        self.size = 0
        self.value = 0
        self.letter_map = dict()

    def __repr__(self):
        return "{}: {}".format(self.letter_map, self.value)

    def add_word(self, word, value):
        if not word:
            self.value += value
            return
        letter = word[0]

        sub_trie = None
        if letter in self.letter_map:
            sub_trie = self.letter_map[letter]
        else:
            sub_trie = Trie()
            self.letter_map[letter] = sub_trie

        self.size += 1
        sub_trie.add_word(word[1:], value)
        self.value = sum([x.value for x in self.letter_map.values()])

    def find_prefix(self, word):
        if not word:
            return self.value

        letter = word[0]
        if letter in self.letter_map:
            sub_trie = self.letter_map[letter]
            return sub_trie.find_prefix(word[1:])

        return 0


class PrefixMapSum:

    def __init__(self, trie):
        self.trie = trie

    def insert(self, key, value):
        self.trie.add_word(key, value)

    def sum(self, prefix):
        return self.trie.find_prefix(prefix)


# Tests
pms = PrefixMapSum(Trie())
pms.insert("columnar", 3)
assert pms.sum("col") == 3
pms.insert("column", 2)
assert pms.sum("col") == 5

```



Question 233

This problem was asked by Apple.

Implement the function `fib(n)`, which returns the nth number in the Fibonacci sequence, using only `O(1)` space.

Answer 233





```

def get_fib(n):
    assert n > 0

    fib_a = 0
    fib_b = 1

    if n == 1:
        return fib_a
    elif n == 2:
        return fib_b

    fib_c = None
    for _ in range(n - 2):
        fib_c = fib_a + fib_b
        fib_a = fib_b
        fib_b = fib_c

    return fib_c


## Tests
assert get_fib(5) == 3
assert get_fib(2) == 1
assert get_fib(7) == 8

```



Question 234

This problem was asked by Microsoft.

Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight. Given an undirected graph with weighted edges, compute the maximum weight spanning tree.

Answer 234





```

class Node:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return hash(self.val)

    def __repr__(self):
        return str(self.val)


class Edge:
    def __init__(self, target, weight):
        self.target = target
        self.weight = weight

    def __hash__(self):
        return hash(self.target)

    def __repr__(self):
        return "-{}-> {}".format(self.weight, self.target)


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)
        self.edges[node] = set()

    def add_edge(self, source, target, weight):
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)

        self.edges[source].add(Edge(target, weight))
        self.edges[target].add(Edge(source, weight))


def get_max_span_helper(g, start, remaining, score):
    if not remaining:
        return score

    scores = list()
    for edge in g.edges[start]:
        if edge.target in remaining:
            rem_cp = remaining.copy()
            rem_cp.remove(edge.target)
            new_score = get_max_span_helper(
                g, edge.target, rem_cp, score + edge.weight)
            scores.append(new_score)

    return max(scores)


def get_max_span(g):
    remaining = g.nodes.copy()
    start_node = list(remaining)[0]
    remaining.remove(start_node)

    score = get_max_span_helper(g, start_node, remaining, 0)
    return score


# Tests
g = Graph()
a = Node('a')
b = Node('b')
c = Node('c')
g.add_edge(a, b, 1)
g.add_edge(a, c, 2)
g.add_edge(b, c, 3)


assert get_max_span(g) == 5

```



Question 235

This problem was asked by Facebook.

Given an array of numbers of length `N`, find both the minimum and maximum using less than `2 * (N - 2)` comparisons.

Answer 235





```

def find_min_max(nums):
    mini = maxi = nums[0]
    for num in nums[1:]:
        if num < mini:
            mini = num
            continue
        elif num > maxi:
            maxi = num

    return mini, maxi


# Tests
assert find_min_max([4, 3, 1, 2, 5]) == (1, 5)

```



Question 236

This problem was asked by Nvidia.

You are given a list of N points `(x1, y1), (x2, y2), ..., (xN, yN)` representing a polygon. You can assume these points are given in order; that is, you can construct the polygon by connecting point 1 to point 2, point 2 to point 3, and so on, finally looping around to connect point N to point 1.

Determine if a new point p lies inside this polygon. (If p is on the boundary of the polygon, you should return False).

Answer 236





```

## Transcribing a high-level solution I found

1. Draw a horizontal line to the right of each point and extend it to infinity
2. Count the number of times the line intersects with polygon edges.
3. A point is inside the polygon if either count of intersections is odd or point lies on an edge of polygon.  If none of the conditions is true, then point lies outside.

[Source](https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/)

```



Question 237

This problem was asked by Amazon.

A tree is symmetric if its data and shape remain unchanged when it is reflected about the root node. The following tree is an example:

```
        4
      / | \
    3   5   3
  /           \
9              9
```

Given a k-ary tree, determine whether it is symmetric.

Answer 237





```

class Node:
    def __init__(self, val):
        self.val = val
        self.children = list()

    def __repr__(self):
        return "{} -> {}".format(self.val, self.children)


def update_levels_dict(root, levels, lnum):
    if lnum not in levels:
        levels[lnum] = list()

    levels[lnum].append(root.val)
    for child in root.children:
        update_levels_dict(child, levels, lnum + 1)


def is_symmetric(tree):
    levels = dict()
    update_levels_dict(tree, levels, 0)

    for level in levels:
        arr = levels[level]
        if arr != arr[::-1]:
            return False

    return True


# Tests
e = Node(9)
f = Node(9)
d = Node(3)
d.children = [e]
c = Node(3)
c.children = [f]
b = Node(5)
a = Node(4)
a.children = [c, b, d]
assert is_symmetric(a)

c.val = 4
assert not is_symmetric(a)

```



Question 238

This problem was asked by MIT.

Blackjack is a two player card game whose rules are as follows:
* The player and then the dealer are each given two cards.
* The player can then "hit", or ask for arbitrarily many additional cards, so long as their total does not exceed 21.
* The dealer must then hit if their total is 16 or lower, otherwise pass.
* Finally, the two compare totals, and the one with the greatest sum not exceeding 21 is the winner.

For this problem, cards values are counted as follows: each card between 2 and 10 counts as their face value, face cards count as 10, and aces count as 1.

Given perfect knowledge of the sequence of cards in the deck, implement a blackjack solver that maximizes the player's score (that is, wins minus losses).

Answer 238





```

from random import random

card_distribution = dict()


def generate_card_distribution():
    for i in range(1, 11):
        if i not in card_distribution:
            card_distribution[i] = 0

        if i == 10:
            card_distribution[i] += 4
        else:
            card_distribution[i] += 1
    summed = sum(card_distribution.values())

    prev = 0
    for key in card_distribution:
        card_distribution[key] = prev + (card_distribution[key]/summed)
        prev = card_distribution[key]


def get_next_card_value():
    rand = random()
    lower, upper = 0, None
    for key, value in card_distribution.items():
        upper = value
        if rand >= lower and rand <= upper:
            return key
        lower = upper


def play_turn(scores, dealer):
    if dealer and scores[1] < 16:
        next_card = get_next_card_value()
        scores[1] += next_card
        return True if scores[1] > 21 else False

    if scores[0] < 21:
        next_card = get_next_card_value()
        if scores[0] + next_card < 22:
            scores[0] += next_card

    return False


def play_game():
    scores = [0, 0]
    won = False
    dealer = False

    while not won:
        won = play_turn(scores, dealer)
        dealer = not dealer
        if scores[1] < scores[0]:
            print("Player wins")
            break


# Tests
generate_card_distribution()
play_game()

```



Question 239

This problem was asked by Uber.

One way to unlock an Android phone is through a pattern of swipes across a 1-9 keypad.

For a pattern to be valid, it must satisfy the following:

All of its keys must be distinct.
It must not connect two keys by jumping over a third key, unless that key has already been used.
For example, `4 - 2 - 1 - 7` is a valid pattern, whereas `2 - 1 - 7` is not.

Find the total number of valid unlock patterns of length N, where `1 <= N <= 9`.

Answer 239





```

class Dialpad:
    def __init__(self):
        self.nodes = set(range(1, 10))
        self.edges = dict()
        self.edges[1] = {2, 4}
        self.edges[2] = {1, 3, 5}
        self.edges[3] = {2, 6}
        self.edges[4] = {1, 5, 7}
        self.edges[5] = {2, 4, 6, 8}
        self.edges[6] = {3, 5, 9}
        self.edges[7] = {4, 8}
        self.edges[8] = {5, 7, 9}
        self.edges[9] = {6, 8}


def count_code_helper(dp, code_len, curr, seen):
    if code_len == 0:
        return 1

    seen_cp = seen.copy()
    seen_cp.add(curr)

    nodes = dp.edges[curr]
    sub_count = 0
    for node in nodes:
        sub_count += count_code_helper(dp, code_len - 1, node, seen_cp)

    return sub_count


def count_codes(dp, code_len):
    if code_len == 1:
        return len(dp.nodes)

    count = 0
    for node in dp.nodes:
        count += count_code_helper(dp, code_len, node, set())

    return count


# Tests
dp = Dialpad()
assert count_codes(dp, 1) == 9
assert count_codes(dp, 2) == 68
assert count_codes(dp, 3) == 192

```



Question 240

This problem was asked by Spotify.

There are `N` couples sitting in a row of length `2 * N`. They are currently ordered randomly, but would like to rearrange themselves so that each couple's partners can sit side by side.
What is the minimum number of swaps necessary for this to happen?

Answer 240





```

I think, at the most, `n-1` moves should be sufficient. 

```



Question 241

This problem was asked by Palantir.

In academia, the h-index is a metric used to calculate the impact of a researcher's papers. It is calculated as follows:

A researcher has index `h` if at least `h` of her `N` papers have `h` citations each. If there are multiple `h` satisfying this formula, the maximum is chosen.

For example, suppose `N = 5`, and the respective citations of each paper are `[4, 3, 0, 1, 5]`. Then the h-index would be `3`, since the researcher has `3` papers with at least `3` citations.

Given a list of paper citations of a researcher, calculate their h-index.

Answer 241





```

def get_h_index(citations):
    citations.sort(reverse=True)

    for i, cit_count in enumerate(citations):
        if i >= cit_count:
            return i


# Tests
assert get_h_index([4, 3, 0, 1, 5]) == 3
assert get_h_index([4, 1, 0, 1, 1]) == 1
assert get_h_index([4, 4, 4, 5, 4]) == 4

```



Question 242

This problem was asked by Twitter.

You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour. Implement a data structure that efficiently supports the following:
* `update(hour: int, value: int)`: Increment the element at index hour by value.
* `query(start: int, end: int)`: Retrieve the number of subscribers that have signed up between start and end (inclusive).
You can assume that all values get cleared at the end of the day, and that you will not be asked for start and end values that wrap around midnight.

Answer 242





```

class HourlySubscribers:
    def __init__(self):
        self.subscribers = [0] * 24

    def update(self, hour, value):
        self.subscribers[hour] += value

    def query(self, start, end):
        return sum(self.subscribers[start:end])


# Tests
hs = HourlySubscribers()
hs.update(2, 50)
hs.update(5, 110)
assert hs.query(1, 7) == 160

```



Question 243

This problem was asked by Etsy.

Given an array of numbers `N` and an integer `k`, your task is to split `N` into `k` partitions such that the maximum sum of any partition is minimized. Return this sum.

For example, given `N = [5, 1, 2, 7, 3, 4]` and `k = 3`, you should return `8`, since the optimal partition is `[5, 1, 2], [7], [3, 4]`.

Answer 243





```

import sys


def split(arr, k):
    if k == 1:
        return ([arr], sum(arr))

    min_val = sys.maxsize
    min_cand = None
    for i in range(len(arr)):
        arr_1, sum_1 = ([arr[:i]], sum(arr[:i]))
        arr_2, sum_2 = split(arr[i:], k - 1)
        candidate = arr_1 + arr_2, max(sum_1, sum_2)
        if candidate[1] < min_val:
            min_val = candidate[1]
            min_cand = candidate

    return min_cand


def split_helper(arr, k):
    return split(arr, k)[1]


# Tests
assert split_helper([5, 1, 2, 7, 3, 4], 3) == 8

```



Question 244

This problem was asked by Square.

The Sieve of Eratosthenes is an algorithm used to generate all prime numbers smaller than N. The method is to take increasingly larger prime numbers, and mark their multiples as composite.

For example, to find all primes less than 100, we would first mark `[4, 6, 8, ...]` (multiples of two), then `[6, 9, 12, ...]` (multiples of three), and so on. Once we have done this for all primes less than `N`, the unmarked numbers that remain will be prime.

Implement this algorithm.

Bonus: Create a generator that produces primes indefinitely (that is, without taking `N` as an input).

Answer 244





```

import time


def get_next_prime(primes_seen):
    num = primes_seen[-1] + 1
    while True:
        if all([num % x for x in primes_seen]):
            time.sleep(0.1)
            yield num
        num += 1


def print_primes():
    first_prime = 2
    primes_seen = [first_prime]
    print(first_prime)
    for next_prime in get_next_prime(primes_seen):
        primes_seen.append(next_prime)
        print(next_prime)


print_primes()

```



Question 245

This problem was asked by Yelp.

You are given an array of integers, where each element represents the maximum number of steps that can be jumped going forward from that element. Write a function to return the minimum number of jumps you must take in order to get from the start to the end of the array.

For example, given `[6, 2, 4, 0, 5, 1, 1, 4, 2, 9]`, you should return `2`, as the optimal solution involves jumping from `6` to `5`, and then from `5` to `9`.

Answer 245





```

def get_min_jumps(arr):
    if len(arr) < 2:
        return 0

    start = arr[0]
    candidates = list()
    for i in range(1, min(start + 1, len(arr))):
        if arr[i] == 0:
            continue
        candidate = 1 + get_min_jumps(arr[i:])
        candidates.append(candidate)

    return min(candidates)


# Tests
assert get_min_jumps([6, 2, 4, 0, 5, 1, 1, 4, 2, 9]) == 2

```



Question 246

This problem was asked by Dropbox.

Given a list of words, determine whether the words can be chained to form a circle. A word `X` can be placed in front of another word `Y` in a circle if the last character of `X` is same as the first character of `Y`.

For example, the words `['chair', 'height', 'racket', 'touch', 'tunic']` can form the following circle: `chair -> racket -> touch -> height -> tunic -> chair`.

Answer 246





```

def can_form_circle(curr_char, remaining_words, seen_circle, word_dict):
    if not remaining_words and curr_char == seen_circle[0][0]:
        print(seen_circle + [seen_circle[0]])
        return True

    if curr_char not in word_dict or not word_dict[curr_char]:
        return False

    next_words = word_dict[curr_char].copy()
    for next_word in next_words:
        word_dict_cp = word_dict.copy()
        word_dict_cp[curr_char].remove(next_word)
        if can_form_circle(next_word[-1], remaining_words - {next_word},
                           seen_circle + [next_word], word_dict_cp):
            return True

    return False


def create_word_dict(words):
    word_dict = dict()
    for word in words:
        start_char = word[0]
        if not start_char in word_dict:
            word_dict[start_char] = set()
        word_dict[start_char].add(word)
    return word_dict


def circle_helper(words):
    words = set(words)
    word_dict = create_word_dict(words)
    for word in words:
        curr_char = word[-1]
        if can_form_circle(curr_char, words - {word}, [word], word_dict):
            return True

    return False


# Tests
assert circle_helper(['chair', 'height', 'racket', 'touch', 'tunic'])
assert not circle_helper(['height', 'racket', 'touch', 'tunic'])
assert circle_helper(['abc', 'cba'])

```



Question 247

This problem was asked by PayPal.

Given a binary tree, determine whether or not it is height-balanced. A height-balanced binary tree can be defined as one in which the heights of the two subtrees of any node never differ by more than one.

Answer 247





```

class Node:
    def __init__(self):
        self.left = None
        self.right = None


def get_height(root):
    if not root:
        return 0

    left_height = get_height(root.left)
    right_height = get_height(root.right)

    return max(left_height, right_height) + 1


def is_balanced(root):
    left_height = get_height(root.left)
    right_height = get_height(root.right)

    return True if abs(left_height - right_height) < 2 else False


# Tests
a = Node()
b = Node()
c = Node()
a.left = b
assert is_balanced(a)
a.right = c
assert is_balanced(a)
d = Node()
e = Node()
b.left = d
d.left = e
assert not is_balanced(a)

```



Question 248

This problem was asked by Nvidia.

Find the maximum of two numbers without using any if-else statements, branching, or direct comparisons.

Answer 248





```

def get_max(a, b):
    c = a-b
    k = (c >> 31) & 1
    return (a - k*c)


# Tests
assert get_max(5, 3) == 5
assert get_max(5, 10) == 10

```



Question 249

This problem was asked by Salesforce.

Given an array of integers, find the maximum XOR of any two elements.

Answer 249





```

import sys


def max_xor(arr):
    maxx = -sys.maxsize
    for i in range(len(arr) - 1):
        for k in range(i + 1, len(arr)):
            maxx = max(maxx, arr[i] ^ arr[k])

    return maxx


# Tests
assert max_xor([1, 2, 3, 4]) == 7

```



Question 250

This problem was asked by Google.

A cryptarithmetic puzzle is a mathematical game where the digits of some numbers are represented by letters. Each letter represents a unique digit.

For example, a puzzle of the form:

```
  SEND
+ MORE
--------
 MONEY
```
may have the solution:

`{'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}`

Given a three-word puzzle like the one above, create an algorithm that finds a solution.

Answer 250





```

def get_num_from_string(char_map, string):
    power = 0
    total = 0
    for char in string[::-1]:
        total += char_map[char] * (10 ** power)
        power += 1

    return total


def is_valid_map(exp1, exp2, res, char_map):
    num1 = get_num_from_string(char_map, exp1)
    num2 = get_num_from_string(char_map, exp2)
    num3 = get_num_from_string(char_map, res)
    return num1 + num2 == num3


def evaluate_char_maps(exp1, exp2, res, char_maps):
    for char_map in char_maps:
        if is_valid_map(exp1, exp2, res, char_map):
            return char_map


def assign_letters(chars_left, nums_left, restrictions, char_map=dict()):
    if not chars_left:
        return [char_map]

    curr_char = list(chars_left)[0]
    char_maps = list()
    for num in nums_left:
        if num in restrictions[curr_char]:
            continue

        char_map_cp = char_map.copy()
        char_map_cp[curr_char] = num

        child_char_maps = assign_letters(
            chars_left - set([curr_char]),
            nums_left - set([num]),
            restrictions,
            char_map_cp)
        char_maps.extend(child_char_maps)

    return char_maps


def decode(exp1, exp2, res):
    characters = set(exp1) | set(exp2) | set(res)
    assert len(characters) < 11
    nums = set(range(0, 10))

    restrictions = dict()
    for char in characters:
        restrictions[char] = set()
    for word in [exp1, exp2, res]:
        restrictions[word[0]].add(0)

    char_maps = assign_letters(characters, nums, restrictions)
    return evaluate_char_maps(exp1, exp2, res, char_maps)


# Tests
assert decode("SEND", "MORE", "MONEY") == {
    'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}

```



Question 251

This problem was asked by Amazon.

Given an array of a million integers between zero and a billion, out of order, how can you efficiently sort it? Assume that you cannot store an array of a billion elements in memory.

Answer 251





```

# Solution

* Stream the million numbers into a heap
* `O(log(n))` to insert each one

```



Question 252

This problem was asked by Palantir.

The ancient Egyptians used to express fractions as a sum of several terms where each numerator is one. For example, `4 / 13` can be represented as `1 / (4 + 1 / (18 + (1 / 468)))`.

Create an algorithm to turn an ordinary fraction `a / b`, where `a < b`, into an Egyptian fraction.

Answer 252





```

import math
from fractions import Fraction


def get_egypt_frac(fraction, prev_fracs=list()):
    if fraction.numerator == 1:
        prev_fracs.append(fraction)

        return prev_fracs

    egpyt_frac = Fraction(1, math.ceil(
        fraction.denominator / fraction.numerator))
    prev_fracs.append(egpyt_frac)

    new_frac = fraction - egpyt_frac
    return get_egypt_frac(new_frac, prev_fracs)


# Tests
assert get_egypt_frac(Fraction(4, 13)) == \
    [Fraction(1, 4), Fraction(1, 18), Fraction(1, 468)]

```



Question 253

This problem was asked by PayPal.

Given a string and a number of lines `k`, print the string in zigzag form. In zigzag, characters are printed out diagonally from top left to bottom right until reaching the kth line, then back up to top right, and so on.

For example, given the sentence `"thisisazigzag"` and `k = 4`, you should print:

```
t     a     g
 h   s z   a
  i i   i z
   s     g
```

Answer 253





```

def print_zigzag(string, k):

    string_dict = dict()
    for row in range(k):
        string_dict[row] = ""

    crow = 0
    direction = -1
    for i in range(len(string)):
        for row in range(k):
            if row == crow:
                string_dict[row] += string[i]
            else:
                string_dict[row] += " "

        if crow == k-1 or crow == 0:
            direction *= -1

        crow += direction

    final_string = "\n".join([x for x in string_dict.values()])

    print(final_string)


# Tests
print_zigzag("thisisazigzag", 4)
print_zigzag("thisisazigzag", 5)

```



Question 254

This problem was asked by Yahoo.

Recall that a full binary tree is one in which each node is either a leaf node, or has two children. Given a binary tree, convert it to a full one by removing nodes with only one child.

For example, given the following tree:

```
         a
      /     \
    b         c
  /            \
d                 e
  \             /   \
    f          g     h
```

You should convert it to:

Answer 254





```

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "{}=[{},{}]".format(self.val, self.left, self.right)

    def __hash__(self):
        return hash(self.val)


def recover_full(root):
    if not root.left and not root.right:
        return root
    elif root.left and root.right:
        root.left = recover_full(root.left)
        root.right = recover_full(root.right)
        return root
    elif root.left:
        return recover_full(root.left)
    elif root.right:
        return recover_full(root.right)


# Tests
a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")
e = Node("e")
f = Node("f")
g = Node("g")
h = Node("h")

a.left = b
a.right = c
b.left = d
d.right = f
c.right = e
e.left = g
e.right = h

print(a)
print(recover_full(a))

```



Question 255

This problem was asked by Microsoft.

The transitive closure of a graph is a measure of which vertices are reachable from other vertices. It can be represented as a matrix `M`, where `M[i][j] == 1` if there is a path between vertices `i` and `j`, and otherwise `0`.

For example, suppose we are given the following graph in adjacency list form:
```
graph = [
    [0, 1, 3],
    [1, 2],
    [2],
    [3]
]
```

The transitive closure of this graph would be:
```
[1, 1, 1, 1]
[0, 1, 1, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

Given a graph, find its transitive closure.

Answer 255





```

def update_transitive_closure(orig, node, adjacency_list, transitive_closure):
    if len(adjacency_list[node]) == 1:
        return

    for adj_node in adjacency_list[node]:
        if orig == adj_node or node == adj_node:
            continue
        transitive_closure[orig][adj_node] = 1
        update_transitive_closure(
            orig, adj_node, adjacency_list, transitive_closure)


def get_transitive_closure(adjacency_list):
    transitive_closure = \
        [[0 for _ in range(len(adjacency_list))]
         for _ in range(len(adjacency_list))]
    for i in range(len(adjacency_list)):
        transitive_closure[i][i] = 1

    for i in adjacency_list:
        update_transitive_closure(i, i, adjacency_list, transitive_closure)

    return transitive_closure


# Tests
adjacency_list = {
    0: [0, 1, 3],
    1: [1, 2],
    2: [2],
    3: [3]
}
assert get_transitive_closure(adjacency_list) == \
    [[1, 1, 1, 1],
     [0, 1, 1, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

```



Question 256

This problem was asked by Fitbit.

Given a linked list, rearrange the node values such that they appear in alternating `low -> high -> low -> high` ... form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.

Answer 256





```

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def rearrange(llist):
    if not llist.next:
        return llist

    arr = get_list(llist)
    arr.sort()

    for i in range(2, len(arr), 2):
        tmp = arr[i]
        arr[i] = arr[i-1]
        arr[i-1] = tmp

    return get_nodes(arr)


# Tests
assert get_list(rearrange(get_nodes([1, 2, 3, 4, 5]))) == [1, 3, 2, 5, 4]

```



Question 257

This problem was asked by WhatsApp.

Given an array of integers out of order, determine the bounds of the smallest window that must be sorted in order for the entire array to be sorted. For example, given `[3, 7, 5, 6, 9]`, you should return `(1, 3)`.

Answer 257





```

from heapq import heappush as hp


def get_sort_range(arr):
    if arr == sorted(arr):
        return ()

    options = list()
    for sort_start in range(len(arr) - 1):
        for sort_end in range(1, len(arr) + 1):
            a1 = arr[:sort_start]
            a2 = arr[sort_start:sort_end]
            a3 = arr[sort_end:]

            new_arr = a1 + sorted(a2) + a3
            if new_arr == sorted(new_arr):
                # options.append((sort_start, sort_end - 1))
                hp(options, (sort_end - sort_start, (sort_start, sort_end - 1)))

    return options[0][1]


# Test
assert get_sort_range([3, 5, 6, 7, 9]) == ()
assert get_sort_range([3, 7, 5, 6, 9]) == (1, 3)
assert get_sort_range([5, 4, 3, 2, 1]) == (0, 4)

```



Question 258

This problem was asked by Morgan Stanley.

In Ancient Greece, it was common to write text with the first line going left to right, the second line going right to left, and continuing to go back and forth. This style was called "boustrophedon".

Given a binary tree, write an algorithm to print the nodes in boustrophedon order.

For example, given the following tree:

```
       1
    /     \
  2         3
 / \       / \
4   5     6   7
```

You should return `[1, 3, 2, 4, 5, 6, 7]`.

Answer 258





```

class Node:
    def __init__(self, val):
        self.val = val
        self.ln = None
        self.rn = None

    def __repr__(self):
        return "Node=({}, ln={}, rn={})".format(
            self.val, self.ln, self.rn)


def get_bfs_alt(root, level, level_dict):
    if not root:
        return

    if level not in level_dict:
        level_dict[level] = list()
    level_dict[level].append(root.val)

    get_bfs_alt(root.ln, level + 1, level_dict)
    get_bfs_alt(root.rn, level + 1, level_dict)


def get_boustrophedon(root):
    level_dict = dict()
    get_bfs_alt(root, 0, level_dict)

    final_order = list()
    for i in range(len(level_dict)):
        final_order.extend(reversed(level_dict[i]) if i % 2 else level_dict[i])

    return final_order


# Tests
n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)
n6 = Node(6)
n7 = Node(7)

n2.ln = n4
n2.rn = n5
n3.ln = n6
n3.rn = n7
n1.ln = n2
n1.rn = n3

assert get_boustrophedon(n1) == [1, 3, 2, 4, 5, 6, 7]

```



Question 259

This problem was asked by Two Sigma.

Ghost is a two-person word game where players alternate appending letters to a word. The first person who spells out a word, or creates a prefix for which there is no possible continuation, loses. Here is a sample game:

```
Player 1: g
Player 2: h
Player 1: o
Player 2: s
Player 1: t [loses]
```

Given a dictionary of words, determine the letters the first player should start with, such that with optimal play they cannot lose.

For example, if the dictionary is `["cat", "calf", "dog", "bear"]`, the only winning start letter would be b.

Answer 259





```

def get_optimal_chars(words):
    candidates = dict()
    for word in words:
        fc = word[0]
        if fc not in candidates:
            candidates[fc] = set()
        candidates[fc].add(len(word))

    opt_chars = set()
    for char in candidates:
        if all((not (x % 2)) for x in candidates[char]):
            opt_chars.add(char)

    return opt_chars


# Tests
assert get_optimal_chars(["cat", "calf", "dog", "bear"]) == set(['b'])
assert get_optimal_chars(["cat", "calf", "dog", "bear", "ao"]) == set(['b', 'a'])

```



Question 260

This problem was asked by Pinterest.

The sequence `[0, 1, ..., N]` has been jumbled, and the only clue you have for its order is an array representing whether each number is larger or smaller than the last. Given this information, reconstruct an array that is consistent with it. For example, given `[None, +, +, -, +]`, you could return `[1, 2, 3, 0, 4]`.

Answer 260





```

def deduce_nums(arr):
    ln = len(arr)
    count_gt = sum([1 for x in arr if x == '+'])
    first = ln - count_gt - 1
    nums = [first]
    small, large = first - 1, first + 1
    for sym in arr[1:]:
        if sym == '+':
            nums.append(large)
            large += 1
        else:
            nums.append(small)
            small -= 1
    
    return nums


# Tests
assert deduce_nums([None, '+', '+', '-', '+']) == [1, 2, 3, 0, 4]

```



Question 261

This problem was asked by Amazon.

Huffman coding is a method of encoding characters based on their frequency. Each letter is assigned a variable-length binary string, such as `0101` or `111110`, where shorter lengths correspond to more common letters. To accomplish this, a binary tree is built such that the path from the root to any leaf uniquely maps to a character. When traversing the path, descending to a left child corresponds to a `0` in the prefix, while descending right corresponds to `1`.

Here is an example tree (note that only the leaf nodes have letters):

```
        *
      /   \
    *       *
   / \     / \
  *   a   t   *
 /             \
c               s
```
With this encoding, cats would be represented as `0000110111`.

Given a dictionary of character frequencies, build a Huffman tree, and use it to determine a mapping between characters and their encoded binary strings.

Answer 261





```

from queue import Queue


class Node:
    def __init__(self, char, count):
        self.ch = char
        self.ct = count
        self.lt = None
        self.rt = None

    def __repr__(self):
        return "{}=>[ct={},lt={},rt={}]".format(self.ch, self.ct, self.lt, self.rt)


def parse_queue_and_get_tree(nq):
    if nq.qsize() == 1:
        node = nq.get()
        return node

    n1 = nq.get()
    n2 = nq.get()

    par = Node(None, n1.ct + n2.ct)
    par.lt = n1
    par.rt = n2

    nq.put(par)

    return parse_queue_and_get_tree(nq)


def build_tree(words):
    ch_dict = dict()
    for word in words:
        for char in word:
            if not char in ch_dict:
                ch_dict[char] = 0
            ch_dict[char] += 1
    print(ch_dict)

    nodes = list()
    for char in ch_dict:
        nodes.append(Node(char, ch_dict[char]))
    nodes.sort(key=lambda x: x.ct)
    if not nodes:
        return Node(None, 0)

    nq = Queue()
    for node in nodes:
        nq.put(node)

    tree = parse_queue_and_get_tree(nq)
    return tree


def update_char_map(htree, char_map, hcode=""):
    if htree.ch:
        char_map[htree.ch] = hcode
        return

    update_char_map(htree.lt, char_map, hcode + "0")
    update_char_map(htree.rt, char_map, hcode + "1")


# Tests
htree = build_tree(["cats", "cars", "dogs"])
char_map = dict()
update_char_map(htree, char_map)
print(char_map)

htree = build_tree(["cat", "car", "dog"])
char_map = dict()
update_char_map(htree, char_map)
print(char_map)

```



Question 262

This problem was asked by Mozilla.

A bridge in a connected (undirected) graph is an edge that, if removed, causes the graph to become disconnected. Find all the bridges in a graph.

Answer 262





```

* For each edge
    * Disconnect it
    * Choose a random node and try to visit all other nodes
    * If not possible label the edge as a 'bridge'

```



Question 263

This problem was asked by Nest.

Create a basic sentence checker that takes in a stream of characters and determines whether they form valid sentences. If a sentence is valid, the program should print it out.

We can consider a sentence valid if it conforms to the following rules:
* The sentence must start with a capital letter, followed by a lowercase letter or a space.
* All other characters must be lowercase letters, separators `(,,;,:)` or terminal marks `(.,?,!,‽)`.
* There must be a single space between each word.
* The sentence must end with a terminal mark immediately following a word.

Answer 263





```

SEPARATORS = {',', ';', ':'}
TERM_MARKS = {'.', '?', '!'}


def is_valid(context, char, next_chars):
    curr_valid = True

    if not context and not char.istitle():
        return False

    if len(context) == 1:
        if char == ' ' or not char.istitle():
            pass
        else:
            return False

    if char in TERM_MARKS:
        return context[-1] not in (SEPARATORS | TERM_MARKS)

    if not next_chars:
        return char in TERM_MARKS and curr_valid

    return is_valid(context + char, next_chars[0], next_chars[1:]) if curr_valid else False


def is_valid_sentence(sentence):
    return is_valid("", sentence[0], sentence[1:])


# Test
assert is_valid_sentence("Valid sentence.")
assert not is_valid_sentence("Invalid sentence")
assert not is_valid_sentence("INvalid sentence.")
assert is_valid_sentence("A valid sentence.")

```



Question 264

This problem was asked by LinkedIn.

Given a set of characters `C` and an integer `k`, a De Bruijn sequence is a cyclic sequence in which every possible `k`-length string of characters in `C` occurs exactly once.

For example, suppose `C = {0, 1}` and `k = 3`. Then our sequence should contain the substrings `{'000', '001', '010', '011', '100', '101', '110', '111'}`, and one possible solution would be `00010111`.

Create an algorithm that finds a De Bruijn sequence.

Answer 264





```

# NOTE: The answer in the problem description is incorrect since
# it doesn't include '100' or '110'


def generate_combos(chars, k, context=""):
    if not k:
        return set([context])

    combos = set()
    for ch in chars:
        combos |= generate_combos(chars, k-1, context + ch)

    return combos


def get_debruijn_seq(chars, combos, context=""):
    if not combos:
        return set([context])

    dseqs = set()
    if not context:
        for cb in combos:
            child_dseqs = get_debruijn_seq(
                chars, combos - set([cb]), cb)
            dseqs |= child_dseqs

        return dseqs

    for ch in chars:
        new_cb = context[-2:] + ch
        if new_cb in combos:
            child_dseqs = get_debruijn_seq(
                chars, combos - set([new_cb]), context + ch)
            dseqs |= child_dseqs

    return dseqs


# Tests
c, k = {'0', '1'}, 3
combos = generate_combos(c, k)
dseqs = get_debruijn_seq(c, combos)
assert all([all([cb in ds for cb in combos]) for ds in dseqs])

```



Question 265

This problem was asked by Atlassian.

MegaCorp wants to give bonuses to its employees based on how many lines of codes they have written. They would like to give the smallest positive amount to each worker consistent with the constraint that if a developer has written more lines of code than their neighbor, they should receive more money.

Given an array representing a line of seats of employees at MegaCorp, determine how much each one should get paid.

For example, given `[10, 40, 200, 1000, 60, 30]`, you should return `[1, 2, 3, 4, 2, 1]`.

Answer 265





```

def get_segments(arr):
    asc = arr[1] > arr[0]
    prev = arr[0]
    start = 0
    segments = []
    for i, num in enumerate(arr[1:]):
        if (asc and num < prev) or (not asc and num > prev):
            segments.append((asc, i - start + 1))
            start = i + 1
            asc = not asc

        prev = num

    segments.append((asc, len(arr) - start))

    return segments


def get_bonuses(arr):
    if not arr:
        return []
    if len(arr) == 1:
        return [1]

    segments = get_segments(arr)
    bonuses = list()
    for segment in segments:
        asc, length = segment
        seg_bonuses = list(range(length))
        if not asc:
            seg_bonuses.reverse()
        bonuses.extend(seg_bonuses)

    bonuses = [x + 1 for x in bonuses]

    return bonuses


# Tests
assert get_bonuses([1000]) == [1]
assert get_bonuses([10, 40, 200, 1000, 60, 30]) == [1, 2, 3, 4, 2, 1]
assert get_bonuses([10, 40, 200, 1000, 900, 800, 30]) == [1, 2, 3, 4, 3, 2, 1]

```



Question 266

This problem was asked by Pivotal.

A step word is formed by taking a given word, adding a letter, and anagramming the result. For example, starting with the word "APPLE", you can add an "A" and anagram to get "APPEAL".

Given a dictionary of words and an input word, create a function that returns all valid step words.

Answer 266





```

ALPHA_SIZE = 26
APLHA_ASCII_OFFSET = 65


class WordCode:
    def __init__(self, word):
        self.word = word
        self.vec = [0 for _ in range(ALPHA_SIZE)]
        for ch in word:
            ind = ord(ch) - APLHA_ASCII_OFFSET
            self.vec[ind] += 1

    def __repr__(self):
        return "{}=>{}".format(self.word, self.vec)

    def __sub__(self, other):
        result = list()
        for i in range(ALPHA_SIZE):
            result.append(max(0, self.vec[i] - other.vec[i]))

        return result


def get_step_words(word, dictionary):
    step_words = set()
    wc = WordCode(word)
    for dword in dictionary:
        dwc = WordCode(dword)
        diff = dwc - wc
        if sum(diff) == 1:
            step_words.add(dword)

    return step_words


# Tests
assert get_step_words("APPLE", {"APPEAL"}) == {"APPEAL"}
assert get_step_words("APPLE", {"APPEAL", "APPLICT"}) == {"APPEAL"}
assert get_step_words("APPLE", {"APPEAL", "APPLICT", "APPLES"}) == {"APPEAL", "APPLES"}

```



Question 267

This problem was asked by Oracle.

You are presented with an 8 by 8 matrix representing the positions of pieces on a chess board. The only pieces on the board are the black king and various white pieces. Given this matrix, determine whether the king is in check.

For details on how each piece moves, see [here](https://en.wikipedia.org/wiki/Chess_piece#Moves_of_the_pieces).

For example, given the following matrix:

```
...K....
........
.B......
......P.
.......R
..N.....
........
.....Q..
```

You should return `True`, since the bishop is attacking the king diagonally.

Answer 267





```

from enum import Enum


class Type(Enum):
    K = 0
    P = 1
    N = 2
    Q = 3
    R = 4
    B = 5


class Piece:
    def __init__(self, typ, loc):
        self.typ = typ
        self.loc = loc

    def __repr__(self):
        return "Piece=[type={}, loc={}]".format(self.typ, self.loc)


class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Loc=[x={}, y={}]".format(self.x, self.y)


class Board:
    def __init__(self, matrix, pieces):
        self.matrix = matrix
        self.pieces = pieces


def read_board(board_str):
    rows = board_str.split("\n")
    pieces = list()
    king = None
    for i, row in enumerate(rows):
        for k, char in enumerate(row):
            if char == ".":
                continue

            pt = Type[char]
            loc = Location(i, k)
            piece = Piece(pt, loc)
            if pt == Type.K:
                king = piece
            else:
                pieces.append(piece)

    return king, pieces


def is_check_by_piece(kl, p):
    straight_attack = (p.loc.x == kl.x or p.loc.y == kl.y)

    dx = abs(p.loc.x - kl.x)
    dy = abs(p.loc.y - kl.y)
    diagonal_attack = (dx == dy)

    knight_attack = (dx * dy == 2)

    pawn_attack = (p.loc.y == kl.x - 1 or p.loc.y == kl.x + 1) and \
        p.loc.x == kl.x + 1

    if p.typ == Type.P:
        if pawn_attack:
            return True
    elif p.typ == Type.N:
        if knight_attack:
            return True
    elif p.typ == Type.R:
        if straight_attack:
            return True
    elif p.typ == Type.B:
        if diagonal_attack:
            return True
    elif p.typ == Type.Q:
        if straight_attack or diagonal_attack:
            return True


def in_check(board_str):
    king, pieces = read_board(board_str)

    for piece in pieces:
        if is_check_by_piece(king.loc, piece):
            print("In check by {}".format(piece))
            return True

    return False


# Tests
board_str = \
    "...K...." + "\n" + \
    "........" + "\n" + \
    ".B......" + "\n" + \
    "......P." + "\n" + \
    ".......R" + "\n" + \
    "..N....." + "\n" + \
    "........" + "\n" + \
    ".....Q.."
assert in_check(board_str)

```



Question 268

This problem was asked by Indeed.

Given a 32-bit positive integer `N`, determine whether it is a power of four in faster than `O(log N)` time.

Answer 268





```

def is_power_of_four(x):
    # https://stackoverflow.com/a/19611541/8650340

    return ((x & -x) & 0x55555554) == x


# Tests
assert is_power_of_four(4)
assert is_power_of_four(16)
assert is_power_of_four(64)
assert is_power_of_four(256)
assert not is_power_of_four(1)
assert not is_power_of_four(8)
assert not is_power_of_four(100)

```



Question 269

This problem was asked by Microsoft.

You are given an string representing the initial conditions of some dominoes. Each element can take one of three values:
* `L`, meaning the domino has just been pushed to the left,
* `R`, meaning the domino has just been pushed to the right, or
* `.`, meaning the domino is standing still.

Determine the orientation of each tile when the dominoes stop falling. Note that if a domino receives a force from the left and right side simultaneously, it will remain upright.

For example, given the string `.L.R....L`, you should return `LL.RRRLLL`.

Given the string `..R...L.L`, you should return `..RR.LLLL`.

Answer 269





```

def get_new_orientation_helper(dominos):
    changes = 0
    new_dominos = dominos.copy()

    for i in range(len(dominos)):
        if dominos[i] == 'L' and i > 0 and dominos[i-1] == '.' and dominos[i-2] != 'R':
            new_dominos[i-1] = 'L'
            changes += 1
        elif dominos[i] == 'R' and i < len(dominos) - 1 and dominos[i+1] == '.' and dominos[i+2] != 'L':
            new_dominos[i+1] = 'R'
            changes += 1

    return get_new_orientation_helper(new_dominos) if changes else dominos


def get_new_orientation(dominos):
    arr = list(dominos)
    arr = get_new_orientation_helper(arr)
    return "".join(arr)


# Tests
assert get_new_orientation(".L.R....L") == "LL.RRRLLL"
assert get_new_orientation("..R...L.L") == "..RR.LLLL"

```



Question 270

This problem was asked by Twitter.

A network consists of nodes labeled `0` to `N`. You are given a list of edges `(a, b, t)`, describing the time `t` it takes for a message to be sent from node `a` to node `b`. Whenever a node receives a message, it immediately passes the message on to a neighboring node, if possible.

Assuming all nodes are connected, determine how long it will take for every node to receive a message that begins at node `0`.

For example, given `N = 5`, and the following edges:

```
edges = [
    (0, 1, 5),
    (0, 2, 3),
    (0, 5, 4),
    (1, 3, 8),
    (2, 3, 1),
    (3, 5, 10),
    (3, 4, 5)
]
```

You should return `9`, because propagating the message from `0 -> 2 -> 3 -> 4` will take that much time.

Answer 270





```

def find_distance(target, edge_dict):
    if target == 0:
        return 0

    cand_target_distances = edge_dict[target]
    distances = list()
    for cand_tgt, cand_dist in cand_target_distances:
        dist = cand_dist + find_distance(cand_tgt, edge_dict)
        distances.append(dist)

    return min(distances)


def get_shortest_trip(edges, node_count):
    edge_dict = dict()
    for edge in edges:
        start, end, dist = edge
        if end not in edge_dict:
            edge_dict[end] = list()
        edge_dict[end].append((start, dist))

    distances = list()
    for target in set(range(1, node_count + 1)):
        dist = find_distance(target, edge_dict)
        distances.append(dist)

    return max(distances)


# Tests
edges = [
    (0, 1, 5),
    (0, 2, 3),
    (0, 5, 4),
    (1, 3, 8),
    (2, 3, 1),
    (3, 5, 10),
    (3, 4, 5)
]
assert get_shortest_trip(edges, 5) == 9

```



Question 271

This problem was asked by Netflix.

Given a sorted list of integers of length `N`, determine if an element `x` is in the list without performing any multiplication, division, or bit-shift operations.

Do this in `O(log N)` time.

Answer 271





```

import bisect


def does_element_exist(arr, x):
    pos = bisect.bisect(arr, x)
    return pos and arr[pos - 1] == x


# Tests
assert does_element_exist([1, 3, 5, 7, 9], 3)
assert not does_element_exist([1, 3, 5, 7, 9], 6)
assert does_element_exist([1, 3, 5, 7, 9], 1)
assert not does_element_exist([1, 3, 5, 7, 9], 0)
assert not does_element_exist([1, 3, 5, 7, 9], 10)

```



Question 272

This problem was asked by Spotify.

Write a function, `throw_dice(N, faces, total)`, that determines how many ways it is possible to throw `N` dice with some number of faces each to get a specific total.

For example, `throw_dice(3, 6, 7)` should equal `15`.

Answer 272





```

def perm_counter(num_dice, face_range, total):
    if num_dice < 1 or total < 1:
        return 0
    elif num_dice == 1 and total in face_range:
        return 1

    return sum([perm_counter(num_dice - 1, face_range, total - x) for x in face_range])


def throw_dice(num_dice, faces, total):
    return perm_counter(num_dice, range(1, faces + 1), total)


# Tests
assert throw_dice(3, 6, 7) == 15

```



Question 273

This problem was asked by Apple.

A fixed point in an array is an element whose value is equal to its index. Given a sorted array of distinct elements, return a fixed point, if one exists. Otherwise, return `False`.

For example, given `[-6, 0, 2, 40]`, you should return `2`. Given `[1, 5, 7, 8]`, you should return `False`.

Answer 273





```

def get_fixed_point(arr):
    for i, num in enumerate(arr):
        if i == num:
            return i

    return False


# Tests
assert get_fixed_point([-6, 0, 2, 40]) == 2
assert get_fixed_point([1, 5, 7, 8]) == False

```



Question 274

This problem was asked by Facebook.

Given a string consisting of parentheses, single digits, and positive and negative signs, convert the string into a mathematical expression to obtain the answer.

Don't use eval or a similar built-in parser.

For example, given `'-1 + (2 + 3)'`, you should return `4`.

Answer 274





```

def eval_string(expr_string):
    return eval(expr_string)


# Tests
assert eval_string("-1 + (2 + 3)") == 4

```



Question 275

This problem was asked by Epic.

The "look and say" sequence is defined as follows: beginning with the term `1`, each subsequent term visually describes the digits appearing in the previous term. The first few terms are as follows:

```
1
11
21
1211
111221
```

As an example, the fourth term is `1211`, since the third term consists of one `2` and one `1`.

Given an integer `N`, print the `Nth` term of this sequence.

Answer 275





```

def look_and_say(seq_count, num_str="1"):
    if seq_count == 1:
        return num_str

    tuples = [(0, num_str[0])]
    for char in num_str:
        prev_count, prev_char = tuples.pop()
        if char == prev_char:
            tuples.append((prev_count + 1, char))
        else:
            tuples.append((prev_count, prev_char))
            tuples.append((1, char))

    flat_list = [str(x) for tup in tuples for x in tup]
    new_num_str = "".join(flat_list)

    return look_and_say(seq_count - 1, new_num_str)


# Test
assert look_and_say(1) == "1"
assert look_and_say(5) == "111221"
assert look_and_say(6) == "312211"

```



Question 276

This problem was asked by Dropbox.

Implement an efficient string matching algorithm.

That is, given a string of length `N` and a pattern of length `k`, write a program that searches for the pattern in the string with less than `O(N * k)` worst-case time complexity.

If the pattern is found, return the start index of its location. If not, return `False`.

Answer 276





```

def contains_pattern(string, pattern):
    if not string or not pattern:
        return False

    slen, plen = len(string), len(pattern)
    if plen > len(string):
        return False

    hashed_strings = set()
    for i in range(slen - plen + 1):
        hashed_strings.add(string[i:i+plen])

    return pattern in hashed_strings


# Tests
assert contains_pattern("abcabcabcd", "abcd")
assert not contains_pattern("abcabcabc", "abcd")

```



Question 277

This problem was asked by Google.

UTF-8 is a character encoding that maps each symbol to one, two, three, or four bytes.

For example, the Euro sign, `€`, corresponds to the three bytes `11100010 10000010 10101100`. The rules for mapping characters are as follows:
* For a single-byte character, the first bit must be zero.
* For an `n`-byte character, the first byte starts with `n` ones and a zero. The other `n - 1` bytes all start with `10`.
Visually, this can be represented as follows.

```
 Bytes   |           Byte format
-----------------------------------------------
   1     | 0xxxxxxx
   2     | 110xxxxx 10xxxxxx
   3     | 1110xxxx 10xxxxxx 10xxxxxx
   4     | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

Write a program that takes in an array of integers representing byte values, and returns whether it is a valid UTF-8 encoding.

Answer 277





```

BLEN = 8
SHIFT_RES = {
    2: 6,
    3: 14,
    4: 30
}
TAIL_SHIFT = 6
TAIL_SHIFT_RES = 2


def is_valid_utf8(int_arr):
    ln = len(int_arr)
    if ln == 1:
        return int_arr[0] < 128

    first = int_arr[0]
    tail = int_arr[1:]

    sfirst = first >> (BLEN - ln - 1)
    if SHIFT_RES[ln] != sfirst:
        return False

    for num in tail:
        snum = num >> TAIL_SHIFT
        if snum != TAIL_SHIFT_RES:
            return False

    return True


# Tests
assert is_valid_utf8([226, 130, 172])
assert not is_valid_utf8([226, 194, 172])
assert not is_valid_utf8([226])
assert is_valid_utf8([100])
assert is_valid_utf8([194, 130])

```



Question 278

This problem was asked by Amazon.

Given an integer `N`, construct all possible binary search trees with `N` nodes.

Answer 278





```

from copy import deepcopy


class Node:
    def __init__(self, val):
        self.val = val
        self.l, self.r = None, None

    def __repr__(self):
        return "[{}=>(l={}, r={})]".format(self.val, self.l, self.r)


def get_trees(nodes):
    if not nodes:
        return []
    elif len(nodes) == 1:
        return deepcopy(nodes)

    trees = list()
    for ind, root in enumerate(nodes):
        lefts = get_trees(nodes[:ind]) if ind > 0 else list()
        rights = get_trees(nodes[ind + 1:]) if ind < len(nodes) - 1 else list()

        for left in lefts:
            for right in rights:
                root.l = deepcopy(left)
                root.r = deepcopy(right)
                trees.append(deepcopy(root))

    return trees


def create_trees(n):
    nodes = [Node(x) for x in range(n)]
    return get_trees(nodes)


# Tests
trees = create_trees(5)
print(trees)

```



Question 279

This problem was asked by Twitter.

A classroom consists of N students, whose friendships can be represented in an adjacency list. For example, the following descibes a situation where `0` is friends with `1` and `2`, `3` is friends with `6`, and so on.

```
{
    0: [1, 2],
    1: [0, 5],
    2: [0],
    3: [6],
    4: [],
    5: [1],
    6: [3]
}
```
Each student can be placed in a friend group, which can be defined as the transitive closure of that student's friendship relations. In other words, this is the smallest set such that no student in the group has any friends outside this group. For the example above, the friend groups would be `{0, 1, 2, 5}, {3, 6}, {4}`.

Given a friendship list such as the one above, determine the number of friend groups in the class.

Answer 279





```

import random


def populate_group(node, adj_list, group):
    group.add(node)

    adj_nodes = adj_list[node]
    if not adj_nodes:
        return

    for anode in adj_nodes:
        if anode not in group:
            populate_group(anode, adj_list, group)


def get_groups(nodes, adj_list, groups):
    num_nodes = len(nodes)
    while num_nodes:
        new_group = set()
        node = list(nodes)[0]
        populate_group(node, adj_list, new_group)
        groups.append(new_group)
        nodes -= new_group
        num_nodes = len(nodes)

    return groups


def get_num_groups(nodes, adj_list):
    groups = list()
    isolated = set()
    for node in nodes:
        if not adj_list[node]:
            isolated.add(node)

    for node in isolated:
        groups.append({node})
        nodes.remove(node)
        del adj_list[node]

    groups = get_groups(nodes, adj_list, groups)
    print(groups)
    return len(groups)


# Tests
adj_list = {
    0: [1, 2],
    1: [0, 5],
    2: [0],
    3: [6],
    4: [],
    5: [1],
    6: [3]
}
assert get_num_groups(set(range(7)), adj_list) == 3

```



Question 280

This problem was asked by Pandora.

Given an undirected graph, determine if it contains a cycle.

Answer 280





```

class Node:
    def __init__(self, val):
        self.val = val
        self.adj_nodes = set()

    def __hash__(self):
        return hash(self.val)

    def __repr__(self):
        return self.val


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.unseen_nodes = None

    def has_cycle_helper(self, node, path=list()):
        if node in self.unseen_nodes:
            self.unseen_nodes.remove(node)

        for adj_node in node.adj_nodes:
            if adj_node in path and adj_node != path[-1]:
                return True

            if self.unseen_nodes:
                return self.has_cycle_helper(adj_node, path + [node])

        return False

    def has_cycle(self):
        start_node = next(iter(self.nodes))
        self.unseen_nodes = self.nodes.copy()
        return self.has_cycle_helper(start_node)


# Tests
a = Node("a")
b = Node("b")
c = Node("c")

a.adj_nodes = {b}
b.adj_nodes = {c}
c.adj_nodes = {a}

g1 = Graph({a, b, c})
assert g1.has_cycle()

a.adj_nodes = {b, c}
b.adj_nodes = set()
c.adj_nodes = set()
g2 = Graph({a, b, c})
assert not g2.has_cycle()

```



Question 281

This problem was asked by LinkedIn.

A wall consists of several rows of bricks of various integer lengths and uniform height. Your goal is to find a vertical line going from the top to the bottom of the wall that cuts through the fewest number of bricks. If the line goes through the edge between two bricks, this does not count as a cut.

For example, suppose the input is as follows, where values in each row represent the lengths of bricks in that row:

```
[[3, 5, 1, 1],
 [2, 3, 3, 2],
 [5, 5],
 [4, 4, 2],
 [1, 3, 3, 3],
 [1, 1, 6, 1, 1]]
```

The best we can we do here is to draw a line after the eighth brick, which will only require cutting through the bricks in the third and fifth row.

Given an input consisting of brick lengths for each row such as the one above, return the fewest number of bricks that must be cut to create a vertical line.

Answer 281





```

from collections import Counter


def get_min_cuts(brick_wall):

    for i in range(len(brick_wall)):
        # Make wall of cumulative brick lengths
        prev_bricks_len = 0
        for k in range(len(brick_wall[i])):
            brick_wall[i][k] += prev_bricks_len
            prev_bricks_len = brick_wall[i][k]
        brick_wall[i] = brick_wall[i][:-1]

    # Find the most common edge to cut down through
    brick_counter = Counter()
    for row in brick_wall:
        brick_counter.update(row)

    most_common_bricks = brick_counter.most_common()
    bricks_avoided = most_common_bricks[0][1] if most_common_bricks else 0

    return len(brick_wall) - bricks_avoided


# Tests
brick_wall = [[3, 5, 1, 1],
              [2, 3, 3, 2],
              [5, 5],
              [4, 4, 2],
              [1, 3, 3, 3],
              [1, 1, 6, 1, 1]]
assert get_min_cuts(brick_wall) == 2

brick_wall = [[1]]
assert get_min_cuts(brick_wall) == 1

brick_wall = [[1],
              [1, 2]]
assert get_min_cuts(brick_wall) == 1

brick_wall = [[1, 2],
              [1, 2]]
assert get_min_cuts(brick_wall) == 0

```



Question 282

This problem was asked by Netflix.

Given an array of integers, determine whether it contains a Pythagorean triplet. Recall that a Pythogorean triplet `(a, b, c)` is defined by the equation `a^2 + b^2 = c^2`.

Answer 282





```

def contains_pytrip(arr):
    squared = [x * x for x in arr]
    set_of_squares = set(squared)
    for i in range(len(squared) - 1):
        for k in range(i + 1, len(squared) - 1):
            summed = squared[i] + squared[k]
            if summed in set_of_squares:
                return True

    return False


# Tests
assert contains_pytrip([3, 4, 5, 6, 7])
assert not contains_pytrip([3, 5, 6, 7])

```



Question 283

This problem was asked by Google.

A regular number in mathematics is defined as one which evenly divides some power of `60`. Equivalently, we can say that a regular number is one whose only prime divisors are `2`, `3`, and `5`.

These numbers have had many applications, from helping ancient Babylonians keep time to tuning instruments according to the diatonic scale.

Given an integer `N`, write a program that returns, in order, the first `N` regular numbers.

Answer 283





```

from heapq import heappop, heappush

start_factor_counts = {2: 2, 3: 1, 5: 1}


def get_val_from_count(factor_counts):
    total = 1
    for key in factor_counts:
        total *= key * factor_counts[key]
    return total


def populate_heap(n, heap, regular_nums):
    if len(regular_nums) >= n:
        return

    lowest_val, lowest_factors = heappop(heap)
    regular_nums.add(lowest_val)
    for key in lowest_factors:
        lf_copy = lowest_factors.copy()
        lf_copy[key] += 1
        heappush(heap, (lowest_val * key, lf_copy))

    populate_heap(n, heap, regular_nums)


def get_n_regular(n, factor_counts=dict()):
    factor_counts = start_factor_counts

    heap, regular_nums = list(), set()
    heappush(heap, (get_val_from_count(factor_counts), factor_counts))
    populate_heap(n, heap, regular_nums)

    return sorted(regular_nums)


# Tests
assert get_n_regular(10) == [60, 120, 180, 240, 300, 360, 480, 540, 600, 720]

```



Question 284

This problem was asked by Yext.

Two nodes in a binary tree can be called cousins if they are on the same level of the tree but have different parents. For example, in the following diagram `4` and `6` are cousins.

```
    1
   / \
  2   3
 / \   \
4   5   6
```

Given a binary tree and a particular node, find all cousins of that node.

Answer 284





```

class Node:
    def __init__(self, val):
        self.val = val
        self.lev = None
        self.l = None
        self.r = None
        self.p = None

    def __repr__(self):
        return "{} = (l={}, r={})".format(self.val, self.l, self.r)

    def __hash__(self):
        return hash(self.val)


def populate_level_map(node, level_map, parent=None, level=0):
    if not node:
        return

    node.p = parent
    node.lev = level

    if level not in level_map:
        level_map[level] = set()
    level_map[level].add(node)

    populate_level_map(node.l, level_map, node, level + 1)
    populate_level_map(node.r, level_map, node, level + 1)


def get_cousins(root, node):
    level_map = dict()
    populate_level_map(root, level_map)

    cousins = set([x for x in level_map[node.lev] if x.p != node.p])
    return cousins


# Tests
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)
f = Node(6)
a.l, a.r = b, c
b.l, b.r = d, e
c.r = f

assert get_cousins(a, d) == {f}
assert get_cousins(a, f) == {d, e}
assert get_cousins(a, a) == set()
assert get_cousins(a, c) == set()

```



Question 285

This problem was asked by Mailchimp.

You are given an array representing the heights of neighboring buildings on a city street, from east to west. The city assessor would like you to write an algorithm that returns how many of these buildings have a view of the setting sun, in order to properly value the street.

For example, given the array `[3, 7, 8, 3, 6, 1]`, you should return `3`, since the top floors of the buildings with heights `8`, `6`, and `1` all have an unobstructed view to the west.

Can you do this using just one forward pass through the array?

Answer 285





```

def get_sunset_bldgs(buildings):
    sbs = list()

    for height in buildings:
        if sbs and sbs[-1] < height:
            sbs.pop()
        sbs.append(height)

    return sbs


# Tests
assert get_sunset_bldgs([3, 7, 8, 3, 6, 1]) == [8, 6, 1]

```



Question 286

This problem was asked by VMware.

The skyline of a city is composed of several buildings of various widths and heights, possibly overlapping one another when viewed from a distance. We can represent the buildings using an array of `(left, right, height)` tuples, which tell us where on an imaginary `x`-axis a building begins and ends, and how tall it is. The skyline itself can be described by a list of `(x, height)` tuples, giving the locations at which the height visible to a distant observer changes, and each new height.

Given an array of buildings as described above, create a function that returns the skyline.

For example, suppose the input consists of the buildings `[(0, 15, 3), (4, 11, 5), (19, 23, 4)]`. In aggregate, these buildings would create a skyline that looks like the one below.

```
     ______  
    |      |        ___
 ___|      |___    |   | 
|   |   B  |   |   | C |
| A |      | A |   |   |
|   |      |   |   |   |
------------------------
```

As a result, your function should return `[(0, 3), (4, 5), (11, 3), (15, 0), (19, 4), (23, 0)]`.

Answer 286





```

import sys


def get_max_width(buildings):
    leftmost, rightmost = sys.maxsize, -sys.maxsize
    for start, end, _ in buildings:
        if start < leftmost:
            leftmost = start
        if end > rightmost:
            rightmost = end

    return rightmost - leftmost + 1


def get_skyline(buildings):
    skyline_width = get_max_width(buildings)
    fill_arr = [0 for _ in range(skyline_width)]

    for start, end, height in buildings:
        for col in range(start, end):
            fill_arr[col] = max(fill_arr[col], height)

    skyline = list()
    prev_height = None
    for col, col_height in enumerate(fill_arr):
        if not skyline or prev_height != col_height:
            skyline.append((col, col_height))
        prev_height = col_height

    return skyline


# Tests
assert get_skyline([(0, 15, 3), (4, 11, 5), (19, 23, 4)]) == \
    [(0, 3), (4, 5), (11, 3), (15, 0), (19, 4), (23, 0)]

```



Question 287

This problem was asked by Quora.

You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top `k` pairs of websites with the greatest similarity.

For example, suppose `k = 1`, and the list of tuples is:

```
[('a', 1), ('a', 3), ('a', 5),
 ('b', 2), ('b', 6),
 ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
 ('d', 4), ('d', 5), ('d', 6), ('d', 7),
 ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
```

Then a reasonable similarity metric would most likely conclude that `a` and `e` are the most similar, so your program should return `[('a', 'e')]`.

Answer 287





```

from heapq import heappush


def get_similarity_score(users_a, users_b):
    union = users_a | users_b
    intersect = users_a & users_b

    return len(intersect) / len(union)


def get_similar_websites(visits, k=1):
    website_users = dict()
    for website, user in visits:
        if website not in website_users:
            website_users[website] = set()
        website_users[website].add(user)

    websites = list(website_users.keys())

    most_similar = list()
    for i in range(len(websites) - 1):
        for j in range(i + 1, len(websites)):
            web_a, web_b = websites[i], websites[j]
            sim_score = get_similarity_score(website_users[web_a], website_users[web_b])
            heappush(most_similar, (-sim_score, (web_a, web_b)))

    most_similar = [y for x, y in most_similar]

    return most_similar[:k]


# Tests
visits = [
    ("a", 1),
    ("a", 3),
    ("a", 5),
    ("b", 2),
    ("b", 6),
    ("c", 1),
    ("c", 2),
    ("c", 3),
    ("c", 4),
    ("c", 5),
    ("d", 4),
    ("d", 5),
    ("d", 6),
    ("d", 7),
    ("e", 1),
    ("e", 3),
    ("e", 5),
    ("e", 6),
]
assert get_similar_websites(visits, 1) == [("a", "e")]
assert get_similar_websites(visits, 3) == [("a", "e"), ("a", "c"), ("b", "d")]

```



Question 288

This problem was asked by Salesforce.

The number `6174` is known as Kaprekar's contant, after the mathematician who discovered an associated property: for all four-digit numbers with at least two distinct digits, repeatedly applying a simple procedure eventually results in this value. The procedure is as follows:

For a given input `x`, create two new numbers that consist of the digits in `x` in ascending and descending order.
Subtract the smaller number from the larger number.
For example, this algorithm terminates in three steps when starting from `1234`:

```
4321 - 1234 = 3087
8730 - 0378 = 8352
8532 - 2358 = 6174
```

Write a function that returns how many steps this will take for a given input `N`.

Answer 288





```

KAP_CONST = 6174


def apply_kproc(num, steps=0):
    if num == KAP_CONST:
        return steps

    digits = str(num)
    assert len(set(digits)) > 2

    asc_num = "".join(sorted(digits))
    dsc_num = "".join(sorted(digits, reverse=True))

    diff = int(dsc_num) - int(asc_num)
    return apply_kproc(diff, steps + 1)


# Tests
assert apply_kproc(KAP_CONST) == 0
assert apply_kproc(1234) == 3

```



Question 289

This problem was asked by Google.

The game of Nim is played as follows. Starting with three heaps, each containing a variable number of items, two players take turns removing one or more items from a single pile. The player who eventually is forced to take the last stone loses. For example, if the initial heap sizes are 3, 4, and 5, a game could be played as shown below:

| A   | B   | C   |
| --- | --- | --- |
| 3   | 4   | 5   |
| 3   | 1   | 5   |
| 3   | 1   | 3   |
| 0   | 1   | 3   |
| 0   | 1   | 0   |
| 0   | 0   | 0   |

In other words, to start, the first player takes three items from pile `B`. The second player responds by removing two stones from pile `C`. The game continues in this way until player one takes last stone and loses.

Given a list of non-zero starting values `[a, b, c]`, and assuming optimal play, determine whether the first player has a forced win.

Answer 289





```

# Source: https://en.wikipedia.org/wiki/Nim#Mathematical_theory


def has_forced_win(heaps):
    x = heaps[0]
    for heap in heaps[1:]:
        x ^= heap

    for heap in heaps:
        xa = heap ^ x
        if xa < heap:
            return True

    return False


# Tests
assert has_forced_win((3, 4, 5))

```



Question 290

This problem was asked by Facebook.

On a mysterious island there are creatures known as Quxes which come in three colors: red, green, and blue. One power of the Qux is that if two of them are standing next to each other, they can transform into a single creature of the third color.

Given N Quxes standing in a line, determine the smallest number of them remaining after any possible sequence of such transformations.

For example, given the input `['R', 'G', 'B', 'G', 'B']`, it is possible to end up with a single Qux through the following steps:

```
        Arrangement       |   Change
----------------------------------------
['R', 'G', 'B', 'G', 'B'] | (R, G) -> B
['B', 'B', 'G', 'B']      | (B, G) -> R
['B', 'R', 'B']           | (R, B) -> G
['B', 'G']                | (B, G) -> R
['R']                     |
```

Answer 290





```

COLORS = {"R", "G", "B"}


def get_odd_man(col_a, col_b):
    return list(COLORS - set([col_a, col_b]))[0]


def minimize(quixes):
    stack = list()
    for quix in quixes:
        if not stack or stack[-1] == quix:
            stack.append(quix)
            continue

        new = get_odd_man(quix, stack[-1])
        stack.pop()
        stack.append(new)
        while len(stack) > 1 and stack[-1] != stack[-2]:
            a, b = stack.pop(), stack.pop()
            stack.append(get_odd_man(a, b))

    return stack


# Tests
assert minimize(["R", "G", "B", "G", "B"]) == ["R"]

```



Question 291

This problem was asked by Glassdoor.

An imminent hurricane threatens the coastal town of Codeville. If at most two people can fit in a rescue boat, and the maximum weight limit for a given boat is `k`, determine how many boats will be needed to save everyone.

For example, given a population with weights `[100, 200, 150, 80]` and a boat limit of `200`, the smallest number of boats required will be three.

Answer 291





```

BOAT_LIMIT = 200


def get_min_boats_helper(people, boats_used):
    if len(people) < 2:
        return boats_used + len(people)

    first = people[0]
    remaining = people[1:]
    if first == BOAT_LIMIT:
        return get_min_boats_helper(remaining, boats_used + 1)

    allowed = BOAT_LIMIT - first
    second_index = len(remaining) - 1
    while allowed >= people[second_index]:
        second_index -= 1

    if second_index == len(remaining):
        return get_min_boats_helper(remaining, boats_used + 1)

    return get_min_boats_helper(remaining[:second_index] + remaining[second_index + 1:], boats_used + 1)


def get_min_boats(people):
    return get_min_boats_helper(sorted(people, reverse=True), 0)


# Tests
assert get_min_boats([100, 200, 150, 80]) == 3

```



Question 292

This problem was asked by Twitter.

A teacher must divide a class of students into two teams to play dodgeball. Unfortunately, not all the kids get along, and several refuse to be put on the same team as that of their enemies.

Given an adjacency list of students and their enemies, write an algorithm that finds a satisfactory pair of teams, or returns `False` if none exists.

For example, given the following enemy graph you should return the teams `{0, 1, 4, 5}` and `{2, 3}`.
```
students = {
    0: [3],
    1: [2],
    2: [1, 4],
    3: [0, 4, 5],
    4: [2, 3],
    5: [3]
}
```

On the other hand, given the input below, you should return `False`.
```
students = {
    0: [3],
    1: [2],
    2: [1, 3, 4],
    3: [0, 2, 4, 5],
    4: [2, 3],
    5: [3]
}
```

Answer 292





```

class Group:
    def __init__(self):
        self.members = set()
        self.enemies = set()

    def __repr__(self):
        return str(self.members)

    def add_student(self, student, enemies):
        self.members.add(student)
        self.enemies |= set(enemies)


def get_groups(enemy_map):
    students = enemy_map.keys()
    first, second = Group(), Group()
    for student in students:
        if not first.members:
            first.add_student(student, enemy_map[student])
        elif student not in first.enemies:
            first.add_student(student, enemy_map[student])
        elif not second.members:
            second.add_student(student, enemy_map[student])
        elif student not in second.enemies:
            second.add_student(student, enemy_map[student])

    if len(first.members) + len(second.members) == len(students):
        return first.members, second.members

    return False


# Tests
enemy_map = {
    0: [3],
    1: [2],
    2: [1, 4],
    3: [0, 4, 5],
    4: [2, 3],
    5: [3]
}
assert get_groups(enemy_map) == ({0, 1, 4, 5}, {2, 3})

enemy_map = {
    0: [3],
    1: [2],
    2: [1, 3, 4],
    3: [0, 2, 4, 5],
    4: [2, 3],
    5: [3]
}
assert get_groups(enemy_map) == False

```



Question 293

This problem was asked by Uber.

You have N stones in a row, and would like to create from them a pyramid. This pyramid should be constructed such that the height of each stone increases by one until reaching the tallest stone, after which the heights decrease by one. In addition, the start and end stones of the pyramid should each be one stone high.

You can change the height of any stone by paying a cost of `1` unit to lower its height by `1`, as many times as necessary. Given this information, determine the lowest cost method to produce this pyramid.

For example, given the stones `[1, 1, 3, 3, 2, 1]`, the optimal solution is to pay 2 to create `[0, 1, 2, 3, 2, 1]`.

Answer 293





```

from typing import List


def construct_pyramid(length: int):
    assert length % 2

    peak = (length//2) + 1
    start = [x for x in range(1, peak)]
    pyramid = start + [peak] + list(reversed(start))

    return pyramid


def get_pyramid(stones: List[int]):
    len_stones = len(stones)
    len_pyr = len_stones if len_stones % 2 else len_stones - 1

    while len_pyr > 0:
        max_pyr = construct_pyramid(len_pyr)

        for offset in (0, len_stones - len_pyr):
            valid = True
            for pyr_index, pyr_num in enumerate(max_pyr):
                stone_index = pyr_index + offset
                if pyr_num > stones[stone_index]:
                    valid = False
                    break

            if valid:
                return ([0] * offset) + max_pyr + ([0] * (len_stones - offset - len_pyr))

        len_pyr -= 2

    return []


# Tests
assert get_pyramid([1, 1, 3, 3, 2, 1]) == [0, 1, 2, 3, 2, 1]
assert get_pyramid([1, 1, 1, 1, 1]) == [1, 0, 0, 0, 0]
assert get_pyramid([1, 1, 1, 5, 1]) == [0, 0, 1, 2, 1]

```



Question 294

This problem was asked by Square.

A competitive runner would like to create a route that starts and ends at his house, with the condition that the route goes entirely uphill at first, and then entirely downhill.

Given a dictionary of places of the form `{location: elevation}`, and a dictionary mapping paths between some of these locations to their corresponding distances, find the length of the shortest route satisfying the condition above. Assume the runner's home is location `0`.

For example, suppose you are given the following input:
```
elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
paths = {
    (0, 1): 10,
    (0, 2): 8,
    (0, 3): 15,
    (1, 3): 12,
    (2, 4): 10,
    (3, 4): 5,
    (3, 0): 17,
    (4, 0): 10
}
```

In this case, the shortest valid path would be `0 -> 2 -> 4 -> 0`, with a distance of `28`.

Answer 294





```

import sys


def get_shortest_path(
        target, elevations, path_map, path_so_far,
        elevations_so_far, distance, switches):

    if 0 == target and path_so_far:
        return path_so_far, distance if switches < 2 else sys.maxsize

    min_dist, min_path = sys.maxsize, None
    for src, dist in path_map[target]:
        if src == target:
            continue

        new_switches = switches + 1 \
            if elevations_so_far and elevations[src] > elevations_so_far[0] \
            else switches

        new_path_so_far, new_dist = get_shortest_path(
            src, elevations, path_map, [src] + path_so_far,
            [elevations[target]] + elevations_so_far, distance + dist, new_switches)

        if new_dist < min_dist:
            min_dist = new_dist
            min_path = new_path_so_far

    return min_path, min_dist


def get_shortest_path_helper(elevations, paths):
    path_map = dict()
    for (src, tgt), dist in paths.items():
        if tgt not in path_map:
            path_map[tgt] = list()
        path_map[tgt].append((src, dist))

    shortest_path, _ = get_shortest_path(
        0, elevations, path_map, list(), list(), 0, 0)

    return shortest_path


# Tests
elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
paths = {
    (0, 1): 10,
    (0, 2): 8,
    (0, 3): 15,
    (1, 3): 12,
    (2, 4): 10,
    (3, 4): 5,
    (3, 0): 17,
    (4, 0): 10,
}
assert get_shortest_path_helper(elevations, paths) == [0, 2, 4]

```



Question 295

This problem was asked by Stitch Fix.

Pascal's triangle is a triangular array of integers constructed with the following formula:

The first row consists of the number 1.
For each subsequent row, each element is the sum of the numbers directly above it, on either side.
For example, here are the first few rows:
```
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```

Given an input `k`, return the `k`th row of Pascal's triangle.

Bonus: Can you do this using only `O(k)` space?

Answer 295





```

def get_pastri_row(k, row=None):
    assert k and k > 0

    if not row:
        row = [0 for _ in range(k)]
        row[0] = 1

    if k == 1:
        return row

    row = get_pastri_row(k - 1, row)
    for i in range(len(row) - 1, 0, -1):
        row[i] += row[i - 1]

    return row


# Tests
assert get_pastri_row(1) == [1]
assert get_pastri_row(3) == [1, 2, 1]
assert get_pastri_row(5) == [1, 4, 6, 4, 1]

```



Question 296

This problem was asked by Etsy.

Given a sorted array, convert it into a height-balanced binary search tree.

Answer 296





```

class Node:
    def __init__(self, val):
        self.val = val
        self.l = None
        self.r = None

    def __repr__(self):
        return "{}=[l={}, r={}]".format(self.val, self.l, self.r)


def get_hbal_tree(arr):
    if not arr:
        return None

    mid = len(arr) // 2
    node = Node(arr[mid])
    node.l = get_hbal_tree(arr[:mid])
    node.r = get_hbal_tree(arr[mid + 1 :])

    return node


# Tests
assert get_hbal_tree([1, 2, 3, 4, 5]).val == 3
assert get_hbal_tree([1, 2, 3, 4, 5, 6]).val == 4

```



Question 297

This problem was asked by Amazon.

At a popular bar, each customer has a set of favorite drinks, and will happily accept any drink among this set. For example, in the following situation, customer 0 will be satisfied with drinks `0`, `1`, `3`, or `6`.

```
preferences = {
    0: [0, 1, 3, 6],
    1: [1, 4, 7],
    2: [2, 4, 7, 5],
    3: [3, 2, 5],
    4: [5, 8]
}
```

A lazy bartender working at this bar is trying to reduce his effort by limiting the drink recipes he must memorize. Given a dictionary input such as the one above, return the fewest number of drinks he must learn in order to satisfy all customers.

For the input above, the answer would be `2`, as drinks `1` and `5` will satisfy everyone.

Answer 297





```

from typing import Dict, List


def minimize_drinks(drinks, remaining_drinks, remaining_customers, cust_by_drink):
    min_option = drinks

    if not remaining_customers:
        return drinks - remaining_drinks

    for drink in remaining_drinks:
        option = minimize_drinks(
            drinks, remaining_drinks - {drink},
            remaining_customers - cust_by_drink[drink], cust_by_drink)
        if len(option) < len(min_option):
            min_option = option

    return min_option


def get_min_drinks(preferences: Dict[int, List[int]]):
    cust_by_drink = dict()
    for cust in preferences:
        for drink in preferences[cust]:
            if drink not in cust_by_drink:
                cust_by_drink[drink] = set()
            cust_by_drink[drink].add(cust)

    remaining_drinks = set(cust_by_drink.keys())
    remaining_customers = set(preferences.keys())
    min_drinks = minimize_drinks(set(cust_by_drink.keys()), remaining_drinks,
                                 remaining_customers, cust_by_drink)
    return min_drinks


# Tests
preferences = {
    0: [0, 1, 3, 6],
    1: [1, 4, 7],
    2: [2, 4, 7, 5],
    3: [3, 2, 5],
    4: [5, 8]
}
assert get_min_drinks(preferences) == {1, 5}

```



Question 298

This problem was asked by Google.

A girl is walking along an apple orchard with a bag in each hand. She likes to pick apples from each tree as she goes along, but is meticulous about not putting different kinds of apples in the same bag.

Given an input describing the types of apples she will pass on her path, in order, determine the length of the longest portion of her path that consists of just two types of apple trees.

For example, given the input `[2, 1, 2, 3, 3, 1, 3, 5]`, the longest portion will involve types `1` and `3`, with a length of four.

Answer 298





```

from typing import List, Dict
from copy import deepcopy


class AppleSet:
    def __init__(self):
        self.apple_types = dict()

    def __repr__(self):
        return str(self.apple_types)

    def add_apple(self, atype: int):
        if atype not in self.apple_types:
            self.apple_types[atype] = 0
        self.apple_types[atype] += 1

    def remove_apple(self, atype: int):
        self.apple_types[atype] -= 1
        if self.apple_types[atype] == 0:
            del self.apple_types[atype]

    def size(self):
        return len(self.apple_types)

    def total(self):
        return sum(x for x in self.apple_types.values())


def get_min_set(apple_set: AppleSet, apples: List[int]):
    if apple_set.size() == 2:
        return apple_set.total()
    if not apples:
        return 0

    first, last = apples[0], apples[-1]

    apple_set_1 = deepcopy(apple_set)
    apple_set_1.remove_apple(first)
    alt_1 = get_min_set(apple_set_1, apples[1:])

    apple_set_2 = deepcopy(apple_set)
    apple_set_2.remove_apple(last)
    alt_2 = get_min_set(apple_set_2, apples[:-1])

    return max(alt_1, alt_2)


def get_longest_portion(apples: List[int]):
    apple_set = AppleSet()
    for atype in apples:
        apple_set.add_apple(atype)

    return get_min_set(apple_set, apples)


# Tests
assert get_longest_portion([2, 1, 2, 3, 3, 1, 3, 5]) == 4
assert get_longest_portion([2, 1, 2, 2, 2, 1, 2, 1]) == 8
assert get_longest_portion([1, 2, 3, 4]) == 2

```



Question 299

This problem was asked by Samsung.

A group of houses is connected to the main water plant by means of a set of pipes. A house can either be connected by a set of pipes extending directly to the plant, or indirectly by a pipe to a nearby house which is otherwise connected.

For example, here is a possible configuration, where A, B, and C are houses, and arrows represent pipes:
`A <--> B <--> C <--> plant`

Each pipe has an associated cost, which the utility company would like to minimize. Given an undirected graph of pipe connections, return the lowest cost configuration of pipes such that each house has access to water.

In the following setup, for example, we can remove all but the pipes from plant to A, plant to B, and B to C, for a total cost of 16.

```python
pipes = {
    'plant': {'A': 1, 'B': 5, 'C': 20},
    'A': {'C': 15},
    'B': {'C': 10},
    'C': {}
}
```

Answer 299





```

def connect_houses(pipe_costs, houses, total_cost):
    if not houses:
        return total_cost

    costs = list()
    for start_node in (pipe_costs.keys() - houses):
        for adj_node in (pipe_costs[start_node].keys() & houses):
            cost = connect_houses(pipe_costs, houses - {adj_node},
                                  total_cost + pipe_costs[start_node][adj_node])
            costs.append(cost)

    return min(costs)


def get_min_cost(pipe_costs):
    houses = pipe_costs.keys() - {'plant'}
    return connect_houses(pipe_costs, houses, 0)


# Tests
pipe_costs = {
    'plant': {'A': 1, 'B': 5, 'C': 20},
    'A': {'C': 15},
    'B': {'C': 10},
    'C': {}
}
assert get_min_cost(pipe_costs) == 16

```



Question 300

This problem was asked by Uber.

On election day, a voting machine writes data in the form `(voter_id, candidate_id)` to a text file. Write a program that reads this file as a stream and returns the top 3 candidates at any given time. If you find a voter voting more than once, report this as fraud.

Answer 300





```

from heapq import heappush, heapify


class VoterFraudException(Exception):
    pass


class Candidate:
    def __init__(self, name: int):
        self.name = name
        self.vote_count = 0

    def __eq__(self, other):
        return self.vote_count == other.vote_count

    def __lt__(self, other):
        return self.vote_count < other.vote_count

    def __gt__(self, other):
        return self.vote_count > other.vote_count

    def __repr__(self):
        return "Candidate: {}, Votes: {}".format(self.name, self.vote_count)


class VotingMachine:
    def __init__(self):
        self.votes = dict()
        self.leaderboard = list()
        self.voters = set()

    def cast_vote(self, candidate: int, voter: int):
        if voter in self.voters:
            raise VoterFraudException(
                "Fraud committed by voter {}".format(voter))
        self.voters.add(voter)

        if not candidate in self.votes:
            cand_obj = Candidate(candidate)
            heappush(self.leaderboard, cand_obj)
            self.votes[candidate] = cand_obj
        cand_obj = self.votes[candidate]
        cand_obj.vote_count += 1

    def get_top_three(self):
        heapify(self.leaderboard)
        return [x.name for x in self.leaderboard[-3:]]


def process_votes(vm, file_contents):
    for (candidate, voter) in file_contents:
        vm.cast_vote(candidate, voter)


# Tests
file_contents = [
    (0, 0),
    (0, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 6)
]
vm = VotingMachine()
process_votes(vm, file_contents)
assert vm.get_top_three() == [2, 0, 1]
process_votes(vm, [(2, 0)])

```



Question 301

This problem was asked by Triplebyte.

Implement a data structure which carries out the following operations without resizing the underlying array:
- `add(value)`: Add a value to the set of values.
- `check(value)`: Check whether a value is in the set.

The check method may return occasional false positives (in other words, incorrectly identifying an element as part of the set), but should always correctly identify a true element.

Answer 301





```

from hashlib import md5, sha256
from binascii import unhexlify


class BloomFilter:
    def __init__(self):
        self.vector = 0

    def get_hash(self, value):
        return int.from_bytes(
            unhexlify(md5(value.encode("UTF-8")).hexdigest()),
            byteorder='little')

    def add(self, value):
        hashed = self.get_hash(value)
        self.vector |= hashed

    def check(self, value):
        hashed = self.get_hash(value)
        for a, b in zip(bin(hashed)[2:], bin(self.vector)[2:]):
            if bool(int(a)) and not bool(int(b)):
                return False
        return True


# Tests
bf = BloomFilter()
bf.add("test1")
bf.add("test2")
assert bf.check("test1")
assert not bf.check("test3")

```



Question 302

This problem was asked by Uber.

You are given a 2-d matrix where each cell consists of either `/`, `\`, or an empty space. Write an algorithm that determines into how many regions the slashes divide the space.

For example, suppose the input for a three-by-six grid is the following:
```
\    /
 \  /
  \/
```

Considering the edges of the matrix as boundaries, this divides the grid into three triangles, so you should return `3`.

Answer 302





```

from typing import Set
from random import sample


class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash("{}-{}".format(self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return "[x={}, y={}]".format(self.x, self.y)


def explore_region(start: Coord, empty_spaces: Set, nrows: int, ncols: int):
    if start not in empty_spaces:
        return

    empty_spaces.remove(start)
    if start.x > 0:
        explore_region(Coord(start.x - 1, start.y), empty_spaces, nrows, ncols)
    if start.x < nrows - 1:
        explore_region(Coord(start.x + 1, start.y), empty_spaces, nrows, ncols)
    if start.y > 0:
        explore_region(Coord(start.x, start.y - 1), empty_spaces, nrows, ncols)
    if start.y < ncols - 1:
        explore_region(Coord(start.x, start.y + 1), empty_spaces, nrows, ncols)


def get_region_count(text: str):
    matrix = text.splitlines()
    nrows, ncols = len(matrix), len(matrix[0])
    for i in range(nrows):
        matrix[i] = [x for x in matrix[i]]

    empty_spaces = set()
    for row in range(nrows):
        for col in range(ncols):
            if matrix[row][col] == " ":
                empty_spaces.add(Coord(row, col))

    regions = 0
    while empty_spaces:
        start = sample(empty_spaces, 1)[0]
        explore_region(start, empty_spaces, nrows, ncols)
        regions += 1

    return regions


# Tests
matrix = \
    "\\    /\n" + \
    " \\  / \n" + \
    "  \\/  "
assert get_region_count(matrix) == 3

matrix = \
    "     /\n" + \
    " \\  / \n" + \
    "  \\/  "
assert get_region_count(matrix) == 2

matrix = \
    "     /\n" + \
    " \\  / \n" + \
    "  \\   "
assert get_region_count(matrix) == 1

```



Question 303

This problem was asked by Microsoft.

Given a clock time in `hh:mm` format, determine, to the nearest degree, the angle between the hour and the minute hands.

Bonus: When, during the course of a day, will the angle be zero?

Answer 303





```

from math import isclose

FLOAT_EQUALITY_TOLERANCE = 0.5


def get_angle_for_hour(hour: int, minute: int):
    minute_offset = minute / 12
    hour_angle = (hour * 30) + minute_offset
    return hour_angle


def get_angle_for_minute(minute: int):
    return minute * 6


def get_angle(hhmm_time: str):
    hour, minute = map(int, hhmm_time.split(":"))
    hour %= 12
    ha = get_angle_for_hour(hour, minute)
    ma = get_angle_for_minute(minute)

    angle = abs(ha - ma)
    return angle if angle < 180 else 360 - angle


# Tests
assert isclose(get_angle("12:20"), 118, abs_tol=FLOAT_EQUALITY_TOLERANCE)
assert isclose(get_angle("12:00"), 0, abs_tol=FLOAT_EQUALITY_TOLERANCE)
assert isclose(get_angle("6:30"), 3, abs_tol=FLOAT_EQUALITY_TOLERANCE)
assert isclose(get_angle("3:45"), 176, abs_tol=FLOAT_EQUALITY_TOLERANCE)

```



Question 304

This problem was asked by Two Sigma.

A knight is placed on a given square on an `8 x 8` chessboard. It is then moved randomly several times, where each move is a standard knight move. If the knight jumps off the board at any point, however, it is not allowed to jump back on.

After `k` moves, what is the probability that the knight remains on the board?

Answer 304





```

from typing import Set, List


BOARD_SIZE = 8
POSSIBLE_MOVES = 8
CACHE = dict()
KNIGHT_RANGE = [-2, -1, 1, 2]


class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        return "Pos[x={},y={}]".format(self.x, self.y)

    def is_valid(self) -> bool:
        return self.x > 0 and self.x <= BOARD_SIZE and self.y > 0 and self.y <= BOARD_SIZE


def get_valid_moves(start: Position) -> Set[Position]:
    if start in CACHE:
        return CACHE[start]

    candidates = set()
    for hor in KNIGHT_RANGE:
        for ver in KNIGHT_RANGE:
            if abs(hor) + abs(ver) == 3:
                candidates.add(Position(start.x + hor, start.y + ver))
    val_candidates = [pos for pos in candidates if pos.is_valid()]

    CACHE[start] = val_candidates

    return val_candidates


def explore(start: Position, turns: int, counts: List) -> None:
    if not turns:
        return

    valid_moves = get_valid_moves(start)
    counts[0] += POSSIBLE_MOVES
    counts[1] += len(valid_moves)

    for next_pos in valid_moves:
        explore(next_pos, turns - 1, counts)


def get_prob_remain(start: Position, turns: int) -> float:
    global CACHE
    CACHE = dict()

    counts = [0, 0]
    explore(start, turns, counts)

    return counts[1] / counts[0]


# Tests
assert get_prob_remain(Position(4, 4), 1) == 1.0
assert get_prob_remain(Position(1, 1), 3) < 0.66

```



Question 305

This problem was asked by Amazon.

Given a linked list, remove all consecutive nodes that sum to zero. Print out the remaining nodes.

For example, suppose you are given the input `3 -> 4 -> -7 -> 5 -> -6 -> 6`. In this case, you should first remove `3 -> 4 -> -7`, then `-6 -> 6`, leaving only `5`.

Answer 305





```

class Node:
    def __init__(self, x):
        self.val = x
        self.cum = 0
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{},{} ->".format(node.val, node.cum)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def add_cum_sum(head):
    node = head
    cum_sum = 0
    while node:
        node.cum = node.val + cum_sum
        cum_sum = node.cum
        node = node.next


def remove_zero_sum(head):
    add_cum_sum(head)
    dummy_head = Node(None)
    dummy_head.next = head

    seen_totals = dict()
    node = dummy_head
    index = 0
    while node:
        if node.cum in seen_totals:
            seen_totals[node.cum].next = node.next
        seen_totals[node.cum] = node
        index += 1
        node = node.next

    return dummy_head.next


# Tests
llist = get_nodes([3, 4, -7, 5, -6, 6])
assert get_list(remove_zero_sum(llist)) == [5]

```



Question 306

This problem was asked by Palantir.

You are given a list of N numbers, in which each number is located at most k places away from its sorted position. For example, if `k = 1`, a given element at index `4` might end up at indices `3`, `4`, or `5`.

Come up with an algorithm that sorts this list in `O(N log k)` time.

Answer 306





```

def sort_k(arr, k):
    arr = sorted(arr[:k]) + arr[k:]
    for i in range(k, len(arr)):
        start, end = i-k+1, i+1
        p, n = arr[:start], arr[end:]
        sub = sorted(arr[start:end])
        arr = p + sub + n
    return arr


# Test
assert sort_k([1, 0, 2, 4, 3], 2) == [0, 1, 2, 3, 4]

```



Question 307

This problem was asked by Oracle.

Given a binary search tree, find the floor and ceiling of a given integer. The floor is the highest element in the tree less than or equal to an integer, while the ceiling is the lowest element in the tree greater than or equal to an integer.

If either value does not exist, return None.

Answer 307





```

import bisect


class Node:
    def __init__(self, val):
        self.val = val
        self.l = None
        self.r = None


def get_arr(root):
    if not root:
        return list()

    return get_arr(root.l) + [root.val] + get_arr(root.r)


def get_fc(root, val):
    arr = get_arr(root)
    ind = bisect.bisect(arr, val)
    if ind == 0:
        return None, arr[0]
    elif ind == len(arr):
        return arr[-1], None
    elif val == arr[ind-1]:
        return val, val
    else:
        return arr[ind-1], arr[ind]


# Tests
a = Node(4)
b = Node(2)
c = Node(1)
d = Node(3)
b.l = c
b.r = d
e = Node(6)
a.l = b
a.r = e

assert get_fc(a, 2) == (2, 2)
assert get_fc(a, 7) == (6, None)
assert get_fc(a, -1) == (None, 1)
assert get_fc(a, 5) == (4, 6)

```



Question 308

This problem was asked by Quantcast.

You are presented with an array representing a Boolean expression. The elements are of two kinds:
- `T` and `F`, representing the values `True` and `False`.
- `&`, `|`, and `^`, representing the bitwise operators for `AND`, `OR`, and `XOR`.

Determine the number of ways to group the array elements using parentheses so that the entire expression evaluates to `True`.

For example, suppose the input is `['F', '|', 'T', '&', 'T']`. In this case, there are two acceptable groupings: `(F | T) & T` and `F | (T & T)`.

Answer 308





```

SYMBOLS = {'|', '&', '^'}


class Boolean:
    def __init__(self, exp, val, fe, se):
        self.exp = exp
        self.val = val
        self.fe = fe
        self.se = se


def evaluator(arr):
    expr = "".join(arr)
    if len(arr) == 1 or len(arr) == 3:
        return [Boolean(expr, eval(expr), arr[0], arr[2] if len(arr) > 2 else None)]

    groupings = list()
    for i in range(len(arr) // 2):
        pivot = i*2 + 1
        first = arr[:pivot]
        second = arr[pivot + 1:]

        for fe in evaluator(first):
            for se in evaluator(second):
                new_exp = str(fe.val) + arr[pivot] + str(se.val)
                groupings.append(Boolean(
                    new_exp, eval(new_exp), fe, se))

    return groupings


def get_groupings(arr):
    if not arr:
        return []

    for ind in range(len(arr)):
        if arr[ind] == 'F':
            arr[ind] = 'False'
        elif arr[ind] == 'T':
            arr[ind] = 'True'
    groupings = evaluator(arr)
    return groupings


# Tests
assert len(get_groupings(['F', '|', 'T', '&', 'T'])) == 2
assert len(get_groupings(['F', '|', 'T', '&', 'T', '^', 'F'])) == 5

```



Question 309

This problem was asked by Walmart Labs.

There are `M` people sitting in a row of `N` seats, where `M < N`. Your task is to redistribute people such that there are no gaps between any of them, while keeping overall movement to a minimum.

For example, suppose you are faced with an input of `[0, 1, 1, 0, 1, 0, 0, 0, 1]`, where `0` represents an empty seat and `1` represents a person. In this case, one solution would be to place the person on the right in the fourth seat. We can consider the cost of a solution to be the sum of the absolute distance each person must move, so that the cost here would be `5`.

Given an input such as the one above, return the lowest possible cost of moving people to remove all gaps.

Answer 309





```

import sys
from itertools import permutations


def get_people_indices(arr):
    return set([x for x, y in enumerate(arr) if y == 1])


def get_min_dist(vacant_spots, available_people):
    min_dist = sys.maxsize
    perms = list(permutations(range(len(vacant_spots))))
    for perm in perms:
        dist = 0
        for i in range(len(vacant_spots)):
            k = perm[i]
            dist += abs(vacant_spots[i] - available_people[k])
        min_dist = min(min_dist, dist)
    return min_dist


def get_lowest_cost(arr):
    num_people = sum(arr)
    ovr_people_indices = set([x for x, y in enumerate(arr) if y == 1])

    lowest_cost = sys.maxsize
    for offset in range(len(arr) - num_people + 1):
        subarr = arr[offset:offset + num_people]
        all_indices = set([offset + x for x in range(num_people)])
        people_indices = set([offset + x for x in get_people_indices(subarr)])

        vacant_indices = list(all_indices - people_indices)
        occupied_ovr_indices = list(ovr_people_indices - people_indices)

        lowest_cost = min(lowest_cost, get_min_dist(
            vacant_indices, occupied_ovr_indices))

    return lowest_cost


# Tests
assert get_lowest_cost([0, 1, 1, 0, 1, 0, 0, 0, 1]) == 5

```



Question 310

This problem was asked by Pivotal.

Write an algorithm that finds the total number of set bits in all integers between `1` and `N`.

Answer 310





```

def get_set_bits(num):
    if not num:
        return 0

    max_pow, max_pow_of_two = 0, 1
    while max_pow_of_two - 1 <= num:
        max_pow_of_two *= 2
        max_pow += 1
    max_pow_of_two //= 2
    max_pow -= 1

    remainder = num - (max_pow_of_two - 1)
    set_bits = ((max_pow * max_pow_of_two) // 2)

    set_bits = set_bits + get_set_bits(remainder)

    return set_bits


# Tests
assert get_set_bits(0) == 0
assert get_set_bits(1) == 1
assert get_set_bits(2) == 2
assert get_set_bits(3) == 4
assert get_set_bits(4) == 5

```



Question 311

This problem was asked by Sumo Logic.

Given an unsorted array, in which all elements are distinct, find a "peak" element in `O(log N)` time.

An element is considered a peak if it is greater than both its left and right neighbors. It is guaranteed that the first and last elements are lower than all others.

Answer 311





```

def find_peak(arr):
    if not arr:
        return None

    mid = len(arr) // 2

    if mid > 0 and arr[mid] > arr[mid - 1] and \
            mid < len(arr) and arr[mid] > arr[mid + 1]:
        return arr[mid]

    if mid > 0 and arr[mid] > arr[mid - 1]:
        return find_peak(arr[:mid])

    return find_peak(arr[mid + 1:])


# Tests
assert find_peak([0, 2, 4, 5, 3, 1]) == 5

```



Question 312

This problem was asked by Wayfair.

You are given a `2 x N` board, and instructed to completely cover the board with the following shapes:
- Dominoes, or `2 x 1` rectangles.
- Trominoes, or L-shapes.

For example, if `N = 4`, here is one possible configuration, where A is a domino, and B and C are trominoes.

```
A B B C
A B C C
```

Given an integer N, determine in how many ways this task is possible.

Answer 312





```

def get_arrangement_count(free_spaces):
    if not free_spaces:
        return 1
    elif free_spaces < 2:
        return 0

    arrangements = 0
    if free_spaces >= 3:
        arrangements += (2 + get_arrangement_count(free_spaces - 3))
    arrangements += (2 + get_arrangement_count(free_spaces - 2))

    return arrangements


def count_arragements(columns):
    return get_arrangement_count(columns * 2)


# Tests
assert count_arragements(4) == 32

```



Question 313

This problem was asked by Citrix.

You are given a circular lock with three wheels, each of which display the numbers `0` through `9` in order. Each of these wheels rotate clockwise and counterclockwise.

In addition, the lock has a certain number of "dead ends", meaning that if you turn the wheels to one of these combinations, the lock becomes stuck in that state and cannot be opened.

Let us consider a "move" to be a rotation of a single wheel by one digit, in either direction. Given a lock initially set to `000`, a target combination, and a list of dead ends, write a function that returns the minimum number of moves required to reach the target state, or `None` if this is impossible.

Answer 313





```

from typing import Set


class Combo:
    def __init__(self, key_1: int, key_2: int, key_3: int):
        self.key_1 = key_1 if key_1 > -1 else key_1 + 10
        self.key_2 = key_2 if key_1 > -1 else key_1 + 10
        self.key_3 = key_3 if key_1 > -1 else key_1 + 10

    def __hash__(self):
        return hash((self.key_1, self.key_2, self.key_3))

    def __eq__(self, other):
        return \
            self.key_1 == other.key_1 and \
            self.key_2 == other.key_2 and \
            self.key_3 == other.key_3

    def __repr__(self):
        return "{}-{}-{}".format(self.key_1, self.key_2, self.key_3)


def get_moves(target: Combo, deadends: Set[Combo],
              start: Combo = Combo(0, 0, 0)):
    if start == target:
        return 0
    elif start in deadends:
        return None

    if start.key_1 != target.key_1:
        k1_moves = list()
        k1_diff = abs(start.key_1 - target.key_1)
        k1_new_start = Combo(target.key_1, start.key_2, start.key_3)
        k1_moves = [
            k1_diff + get_moves(target, deadends, k1_new_start),
            (10 - k1_diff) + get_moves(target, deadends, k1_new_start)
        ]
        k1_moves = [x for x in k1_moves if x]
        if k1_moves:
            return min(k1_moves)

    if start.key_2 != target.key_2:
        k2_moves = list()
        k2_diff = abs(start.key_1 - target.key_1)
        k2_new_start = Combo(start.key_1, target.key_2, start.key_3)
        k2_moves = [
            k2_diff + get_moves(target, deadends, k2_new_start),
            (10 - k2_diff) + get_moves(target, deadends, k2_new_start)
        ]
        k2_moves = [x for x in k2_moves if x]
        if k2_moves:
            return min(k2_moves)

    if start.key_2 != target.key_3:
        k3_moves = list()
        k3_diff = abs(start.key_1 - target.key_1)
        k3_new_start = Combo(start.key_1, start.key_2, target.key_3)
        k3_moves = [
            k3_diff + get_moves(target, deadends, k3_new_start),
            (10 - k3_diff) + get_moves(target, deadends, k3_new_start)
        ]
        k3_moves = [x for x in k3_moves if x]
        if k3_moves:
            return min(k3_moves)

    return None


# Tests
assert get_moves(target=Combo(3, 4, 5), deadends={Combo(6, 6, 6)}) == 13

```



Question 314

This problem was asked by Spotify.

You are the technical director of WSPT radio, serving listeners nationwide. For simplicity's sake we can consider each listener to live along a horizontal line stretching from `0` (west) to `1000` (east).

Given a list of `N` listeners, and a list of `M` radio towers, each placed at various locations along this line, determine what the minimum broadcast range would have to be in order for each listener's home to be covered.

For example, suppose `listeners = [1, 5, 11, 20]`, and `towers = [4, 8, 15]`. In this case the minimum range would be `5`, since that would be required for the tower at position `15` to reach the listener at position `20`.

Answer 314





```

from typing import List, Set


def get_closest_tower_dist(start: int, end: int, towers: Set[int], dist_so_far: int):
    if start in towers or end in towers:
        return dist_so_far

    return get_closest_tower_dist(start - 1, end + 1, towers, dist_so_far + 1)


def get_max_range(listeners: List[int], towers: List[int]):
    max_dist = 0
    for listener in listeners:
        closest_dist = get_closest_tower_dist(
            listener, listener, set(towers), 0)
        max_dist = max(max_dist, closest_dist)

    return max_dist


# Tests
assert get_max_range([1, 5, 11, 20], [4, 8, 15]) == 5

```



Question 315

This problem was asked by Google.

In linear algebra, a Toeplitz matrix is one in which the elements on any given diagonal from top left to bottom right are identical.

Here is an example:
```
1 2 3 4 8
5 1 2 3 4
4 5 1 2 3
7 4 5 1 2
```

Write a program to determine whether a given input is a Toeplitz matrix.

Answer 315





```

def check_diagonal(start, matrix, val, rows, cols):
    if start[0] == rows or start[1] == cols:
        return True

    if matrix[start[0]][start[1]] == val:
        return check_diagonal((start[0] + 1, start[1] + 1), matrix, val, rows, cols)

    return False


def is_toeplitz(matrix):
    rows, cols = len(matrix), len(matrix[0])
    for ind in range(rows):
        val = matrix[ind][0]
        if not check_diagonal((ind + 1, 1), matrix, val, rows, cols):
            return False

    for ind in range(1, cols):
        val = matrix[0][ind]
        if not check_diagonal((1, ind + 1), matrix, val, rows, cols):
            return False

    return True


# Tests
matrix = [[1, 2, 3, 4, 8], [5, 1, 2, 3, 4], [4, 5, 1, 2, 3], [7, 4, 5, 1, 2]]
assert is_toeplitz(matrix)
matrix = [[1, 2, 3, 0, 8], [5, 1, 2, 3, 4], [4, 5, 1, 2, 3], [7, 4, 5, 1, 2]]
assert not is_toeplitz(matrix)

```



Question 316

This problem was asked by Snapchat.

You are given an array of length `N`, where each element `i` represents the number of ways we can produce `i` units of change. For example, `[1, 0, 1, 1, 2]` would indicate that there is only one way to make `0`, `2`, or `3` units, and two ways of making `4` units.

Given such an array, determine the denominations that must be in use. In the case above, for example, there must be coins with value `2`, `3`, and `4`.

Answer 316





```

from typing import List


def get_ways_to_produce(num: int, factors: List[int]):
    if not num or num in factors:
        return 1

    ways = 0
    for i, factor in enumerate(factors):
        if num % factor == 0:
            ways += get_ways_to_produce(num // factor, factors[i:])

    return ways


def get_denominators(ways_to_produce: List[int]):
    factors = [i for i, num in enumerate(ways_to_produce)
               if (num == 1 and i > 0)]

    for i, num in enumerate(ways_to_produce):
        if get_ways_to_produce(i, factors) == num - 1:
            factors.append(i)

    return factors


# Tests
assert get_denominators([1, 0, 1, 1, 2]) == [2, 3, 4]

```



Question 317

This problem was asked by Yahoo.

Write a function that returns the bitwise `AND` of all integers between `M` and `N`, inclusive.

Answer 317





```

# if there are 2 consecutive numbers, the least significant
# bit will be 0 once, which means the result of an AND on
# the last bit will be zero
from math import log2, ceil


def bitwise_and_slow(start, end):
    result = end
    for num in range(start, end):
        result &= num

    return result


def bitwise_and(start, end):
    diff = end - start + 1
    power_diff = ceil(log2(diff))
    power_end = ceil(log2(end))

    result = start & (2**power_end - 2**power_diff)
    return result


# Tests
assert bitwise_and_slow(5, 6) == 4
assert bitwise_and(5, 6) == 4
assert bitwise_and_slow(126, 127) == 126
assert bitwise_and(126, 127) == 126
assert bitwise_and_slow(129, 215) == 128
assert bitwise_and(193, 215) == 192

```



Question 318

This problem was asked by Apple.

You are going on a road trip, and would like to create a suitable music playlist. The trip will require `N` songs, though you only have `M` songs downloaded, where `M < N`. A valid playlist should select each song at least once, and guarantee a buffer of `B` songs between repeats.

Given `N`, `M`, and `B`, determine the number of valid playlists.

Answer 318





```

import random
from copy import deepcopy


class Node:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return self.val == other.val

    def __repr__(self):
        return str(self.val)


class LRUCache:
    def __init__(self, size):
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = size
        self.recent_nodes = dict()

    def use(self, val):
        if val in self.recent_nodes:
            used_node = self.recent_nodes[val]
            used_node.prev = used_node.next
        elif len(self.recent_nodes) == self.size:
            used_node = Node(val)
            del self.recent_nodes[self.head.next.val]
            self.head.next = self.head.next.next
        else:
            used_node = Node(val)

        before_tail = self.tail.prev
        before_tail.next = used_node
        used_node.next = self.tail
        used_node.prev = before_tail
        self.tail.prev = used_node
        self.recent_nodes[val] = used_node


def count_playlists(song_ids, cache, plays_left):
    if plays_left == 0:
        return 1

    total = 0
    for song_id in song_ids:
        if song_id in cache.recent_nodes:
            continue
        new_cache = deepcopy(cache)
        new_cache.use(song_id)
        total += count_playlists(song_ids, new_cache, plays_left - 1)

    return total


def get_valid_playlists(plays, songs, buffer):
    song_ids = set(range(songs))
    lru_cache = LRUCache(buffer)

    total = count_playlists(song_ids, lru_cache, plays)
    return total


# Tests
assert get_valid_playlists(6, 4, 2) > get_valid_playlists(6, 4, 3)

```



Question 319

This problem was asked by Airbnb.

An 8-puzzle is a game played on a `3 x 3` board of tiles, with the ninth tile missing. The remaining tiles are labeled `1` through `8` but shuffled randomly. Tiles may slide horizontally or vertically into an empty space, but may not be removed from the board.

Design a class to represent the board, and find a series of steps to bring the board to the state `[[1, 2, 3], [4, 5, 6], [7, 8, None]]`.

Answer 319





```

FINAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, None]
]


class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)


def get_adj_nums(empty_pos, num_rows, num_cols):
    adj_nums = set()
    if empty_pos.x > 0:
        if empty_pos.y > 0:
            adj_nums.add(Coord(empty_pos.x - 1, empty_pos.y - 1))
        if empty_pos.y < num_cols - 1:
            adj_nums.add(Coord(empty_pos.x - 1, empty_pos.y + 1))
    if empty_pos.y < num_rows - 1:
        if empty_pos.y > 0:
            adj_nums.add(Coord(empty_pos.x + 1, empty_pos.y - 1))
        if empty_pos.y < num_cols - 1:
            adj_nums.add(Coord(empty_pos.x + 1, empty_pos.y + 1))

    return adj_nums


def play(grid, empty_pos, num_rows, num_cols):
    if grid == FINAL_STATE:
        return

    adj_nums = get_adj_nums(empty_pos, num_rows, num_cols)
    for adj_num in adj_nums:
        new_grid = grid.copy()
        new_grid[empty_pos.x][empty_pos.y] = grid[adj_num.x][adj_num.y]
        new_grid[adj_num.x][adj_num.y] = None
        return play(new_grid, adj_num, num_rows, num_cols)


def win_game(grid):
    empty_pos = None
    for x, row in enumerate(grid):
        for y, val in enumerate(row):
            if not val:
                empty_pos = Coord(x, y)

    play(grid, empty_pos, len(start), len(start[0]))


# Tests
start = [
    [1, 2, 3],
    [4, 5, None],
    [7, 8, 6]
]
assert win_game(start)

```



Question 320

This problem was asked by Amazon.

Given a string, find the length of the smallest window that contains every distinct character. Characters may appear more than once in the window.

For example, given "jiujitsu", you should return 5, corresponding to the final five letters.

Answer 320





```

from sys import maxsize


def check_string(string, start, end, unique_chars):
    substr = string[start:end]
    if start == end or len(set(substr)) < unique_chars:
        return maxsize

    can_1 = check_string(string, start, end - 1, unique_chars)
    can_2 = check_string(string, start + 1, end, unique_chars)

    return min(len(substr), min(can_1, can_2))


def get_longest_distinct_window(string):
    if not string:
        return 0

    return check_string(string, 0, len(string), len(set(string)))


# Tests
assert get_longest_distinct_window("jiujitsu") == 5
assert get_longest_distinct_window("jiujiuuts") == 6

```



Question 321

This problem was asked by PagerDuty.

Given a positive integer `N`, find the smallest number of steps it will take to reach `1`.

There are two kinds of permitted steps:
- You may decrement `N` to `N - 1`.
- If `a * b = N`, you may decrement `N` to the larger of `a` and `b`.

For example, given `100`, you can reach `1` in five steps with the following route: `100 -> 10 -> 9 -> 3 -> 2 -> 1`.

Answer 321





```

def get_closest_factors(num):
    a, b = 1, num
    pa, pb = 1, 1
    while b > a:
        if num % a == 0:
            pa, pb = a, num // a
        b = num / a
        a += 1

    return (pa, pb)


def reduce(num):
    if num == 1:
        return [1]

    # permitted step 1
    next_steps = reduce(num - 1)

    # permitted step 2
    _, large_factor = get_closest_factors(num)
    if large_factor < num:
        # only consider this option if the number is not a prime
        alt_2 = reduce(large_factor)
        if len(next_steps) > len(alt_2):
            # if it's a better option that decrementing by 1, use it
            next_steps = alt_2

    return [num] + next_steps


# Tests
assert reduce(100) == [100, 10, 9, 3, 2, 1]
assert reduce(50) == [50, 10, 9, 3, 2, 1]
assert reduce(64) == [64, 8, 4, 2, 1]
assert reduce(31) == [31, 30, 6, 3, 2, 1]

```



Question 322

This problem was asked by Flipkart.

Starting from `0` on a number line, you would like to make a series of jumps that lead to the integer `N`.

On the `i`th jump, you may move exactly `i` places to the left or right.

Find a path with the fewest number of jumps required to get from `0` to `N`.

Answer 322





```

def jump_to_target(num):
    abs_num = abs(num)
    if abs_num < 2:
        return abs_num

    point, new_point = None, 0
    jump = 1
    while new_point <= abs_num:
        point = new_point
        new_point += jump
        jump += 1
    jump -= 2

    return (2 * (abs_num - point)) + jump


# Tests
assert jump_to_target(0) == 0
assert jump_to_target(1) == 1
assert jump_to_target(2) == 3
assert jump_to_target(3) == 2
assert jump_to_target(-3) == 2
assert jump_to_target(10) == 4
assert jump_to_target(11) == 6

```



Question 323

This problem was asked by Dropbox.

Create an algorithm to efficiently compute the approximate median of a list of numbers.

More precisely, given an unordered list of `N` numbers, find an element whose rank is between `N / 4` and `3 * N / 4`, with a high level of certainty, in less than `O(N)` time.

Answer 323





```

from heapq import heappush
from random import randint


def approx_median(arr):
    if not arr:
        return None

    sarr = list()
    for _ in range(len(arr)//2):
        rand = randint(0, len(arr) - 1)
    heappush(sarr, arr[rand])

    return sarr[len(sarr)//2]


# Tests
print(approx_median([3, 4, 3, 2, 4, 3, 1, 4, 3,
                     4, 2, 3, 4, 3, 0, 4, 0, 0, 1, 1, 0, 1, 2]))

```



Question 324

This problem was asked by Amazon.

Consider the following scenario: there are `N` mice and `N` holes placed at integer points along a line. Given this, find a method that maps mice to holes such that the largest number of steps any mouse takes is minimized.

Each move consists of moving one mouse one unit to the left or right, and only one mouse can fit inside each hole.

For example, suppose the mice are positioned at `[1, 4, 9, 15]`, and the holes are located at `[10, -5, 0, 16]`. In this case, the best pairing would require us to send the mouse at `1` to the hole at `-5`, so our function should return `6`.

Answer 324





```

import sys


def get_min_steps(mice, holes, largest_step=-sys.maxsize):
    if not mice:
        return largest_step

    mouse = mice[0]
    min_steps = list()
    for hole in holes:
        diff = abs(mouse - hole)
        min_steps.append(
            get_min_steps(mice[1:], holes - {hole}, max(largest_step, diff))
        )

    return min(min_steps)


# Tests
assert get_min_steps(mice=[1, 4, 9, 15], holes={10, -5, 0, 16}) == 6

```



Question 325

This problem was asked by Jane Street.

The United States uses the imperial system of weights and measures, which means that there are many different, seemingly arbitrary units to measure distance. There are 12 inches in a foot, 3 feet in a yard, 22 yards in a chain, and so on.

Create a data structure that can efficiently convert a certain quantity of one unit to the correct amount of any other unit. You should also allow for additional units to be added to the system.

Answer 325





```

class UnitConverter:
    def __init__(self):
        self.conv_unit = None
        self.units = dict()

    def add_unit(self, unit_tuple, quantities):
        (known_unit, new_unit) = unit_tuple
        if not self.conv_unit:
            self.conv_unit = known_unit
            self.units[known_unit] = 1

        assert known_unit in self.units or unit_tuple[1] in self.units
        self.units[new_unit] = \
            (quantities[1] / quantities[0]) * \
            self.units[known_unit]

    def convert(self, source_unit, source_quantity, target_unit):
        assert source_unit in self.units and target_unit in self.units
        source_conv = source_quantity / self.units[source_unit]
        return round(source_conv * self.units[target_unit], 2)


# Tests
uc = UnitConverter()
uc.add_unit(("inch", "foot"), (12, 1))
uc.add_unit(("foot", "yard"), (3, 1))
uc.add_unit(("yard", "chain"), (22, 1))
assert uc.convert("inch", 24, "foot") == 2.0
assert uc.convert("inch", 36, "yard") == 1
assert uc.convert("inch", 48, "yard") == 1.33
assert uc.convert("foot", 4, "yard") == 1.33
assert uc.convert("chain", 2, "inch") == 1584.0
assert uc.convert("chain", 3, "foot") == 198.0

```



Question 326

This problem was asked by Netflix.

A Cartesian tree with sequence `S` is a binary tree defined by the following two properties:

It is heap-ordered, so that each parent value is strictly less than that of its children.
An in-order traversal of the tree produces nodes with values that correspond exactly to `S`.
For example, given the sequence `[3, 2, 6, 1, 9]`, the resulting Cartesian tree would be:

```
      1
    /   \   
  2       9
 / \
3   6
```

Given a sequence S, construct the corresponding Cartesian tree.

Answer 326





```

from typing import List


class Node:
    def __init__(self, val: int):
        self.val = val
        self.l = None
        self.r = None

    def __repr__(self):
        return "{}=[l->{}, r->{}]".format(self.val, self.l, self.r)


def make_cartree(arr: List[int], last: Node, root: Node):
    if not arr:
        return root

    node = Node(arr[0])
    if not last:
        return make_cartree(arr[1:], node, node)

    if last.val > node.val:
        node.l = last
        return make_cartree(arr[1:], node, node)

    last.r = node
    return make_cartree(arr[1:], last, last)


# Tests
cartree = make_cartree([3, 2, 6, 1, 9], None, None)
assert str(cartree) == \
    "1=[l->2=[l->3=[l->None, r->None], " + \
    "r->6=[l->None, r->None]], " + \
    "r->9=[l->None, r->None]]"

```



Question 327

This problem was asked by Salesforce.

Write a program to merge two binary trees. Each node in the new tree should hold a value equal to the sum of the values of the corresponding nodes of the input trees.

If only one input tree has a node in a given position, the corresponding node in the new tree should match that input node.

Answer 327





```

class Node:
    def __init__(self, val):
        self.val = val
        self.l = None
        self.r = None

    def __repr__(self):
        return "{}".format(self.val)


def merge(t1, t2, final_prev, left):
    if not t1 and not t2:
        return

    if t1 and t2:
        final_node = Node(t1.val + t2.val)
        if left:
            final_prev.l = final_node
        else:
            final_prev.r = final_node
        merge(t1.l, t2.l, final_node, True)
        merge(t1.r, t2.r, final_node, False)
        return

    only_node = t1 if t1 else t2
    if left:
        final_prev.l = only_node
    else:
        final_prev.r = only_node


# Tests
root_1 = Node(1)
root_1.l = Node(2)
root_1.r = Node(3)
root_1.l.l = Node(4)

root_2 = Node(2)

final_root = Node(0)
merge(root_1, root_2, final_root, True)
assert final_root.l.val == 3

```



Question 328

This problem was asked by Facebook.

In chess, the Elo rating system is used to calculate player strengths based on game results.

A simplified description of the Elo system is as follows. Every player begins at the same score. For each subsequent game, the loser transfers some points to the winner, where the amount of points transferred depends on how unlikely the win is. For example, a 1200-ranked player should gain much more points for beating a 2000-ranked player than for beating a 1300-ranked player.

Implement this system.

Answer 328





```

class EloRatings:
    START_RATING = 1000

    def __init__(self):
        self.ratings = dict()

    def add_player(self, name):
        self.ratings[name] = EloRatings.START_RATING

    def add_result(self, p1, p2, winner):
        if p1 not in self.ratings:
            self.add_player(p1)
        if p2 not in self.ratings:
            self.add_player(p2)

        if not winner:
            if self.ratings[p1] == self.ratings[p2]:
                return

            diff = self.ratings[p1] // 20 \
                if self.ratings[p1] > self.ratings[p2] \
                else -self.ratings[p2] // 20

            self.ratings[p1] -= diff
            self.ratings[p2] += diff
        else:
            loser = p2 if winner == p1 else p1
            diff = self.ratings[loser] // 10
            self.ratings[loser] -= diff
            self.ratings[winner] += diff


# Tests
elo = EloRatings()
elo.add_player("a")
elo.add_player("b")
elo.add_player("c")
elo.add_result("a", "b", "a")
elo.add_result("a", "b", "b")
elo.add_result("a", "b", None)

```



Question 329

This problem was asked by Amazon.

The stable marriage problem is defined as follows:

Suppose you have `N` men and `N` women, and each person has ranked their prospective opposite-sex partners in order of preference.

For example, if `N = 3`, the input could be something like this:

```
guy_preferences = {
    'andrew': ['caroline', 'abigail', 'betty'],
    'bill': ['caroline', 'betty', 'abigail'],
    'chester': ['betty', 'caroline', 'abigail'],
}
gal_preferences = {
    'abigail': ['andrew', 'bill', 'chester'],
    'betty': ['bill', 'andrew', 'chester'],
    'caroline': ['bill', 'chester', 'andrew']
}
```

Write an algorithm that pairs the men and women together in such a way that no two people of opposite sex would both rather be with each other than with their current partners.

Answer 329





```

def find_matches(guy_pref, gal_pref):
    if not guy_pref:
        return []

    matches = list()
    taken_guys, taken_gals = set(), set()
    for guy in guy_preferences:
        gal = guy_pref[guy][0]
        pref_guy = gal_pref[gal][0]
        if pref_guy == guy:
            matches.append((guy, gal))
            taken_guys.add(guy)
            taken_gals.add(gal)

    if not matches:
        for guy in guy_preferences:
            gal = guy_pref[guy][0]
            matches.append((guy, gal))
        return matches

    for (guy, gal) in matches:
        del guy_pref[guy]
        del gal_pref[gal]

        for rguy in guy_pref:
            guy_pref[rguy] = [x for x in guy_pref[rguy] if x not in taken_gals]
        for rgal in gal_pref:
            gal_pref[rgal] = [x for x in gal_pref[rgal] if x not in taken_guys]

    return matches + find_matches(guy_pref, gal_pref)


# Tests
guy_preferences = {
    'andrew': ['caroline', 'abigail', 'betty'],
    'bill': ['caroline', 'betty', 'abigail'],
    'chester': ['betty', 'caroline', 'abigail'],
}
gal_preferences = {
    'abigail': ['andrew', 'bill', 'chester'],
    'betty': ['bill', 'andrew', 'chester'],
    'caroline': ['bill', 'chester', 'andrew']
}
assert find_matches(guy_preferences, gal_preferences) == \
    [('bill', 'caroline'), ('andrew', 'abigail'), ('chester', 'betty')]

```



Question 330

This problem was asked by Dropbox.

A Boolean formula can be said to be satisfiable if there is a way to assign truth values to each variable such that the entire formula evaluates to true.

For example, suppose we have the following formula, where the symbol `¬` is used to denote negation:

```
(¬c OR b) AND (b OR c) AND (¬b OR c) AND (¬c OR ¬a)
```

One way to satisfy this formula would be to let `a = False`, `b = True`, and `c = True`.

This type of formula, with AND statements joining tuples containing exactly one OR, is known as 2-CNF.

Given a 2-CNF formula, find a way to assign truth values to satisfy it, or return `False` if this is impossible.

Answer 330





```

NOT_SYMBOL = "¬"
OR_SYMBOL = "OR"
AND_SYMBOL = "AND"


def get_solution(string, symbols):
    power = len(symbols)
    count_sol = 2 ** power

    sol = None
    for solution in range(count_sol):
        bin_rep = format(solution, '0' + str(power) + 'b')

        new_str = string[:]
        for i in range(power):
            val = str(bool(int(bin_rep[i])))
            new_str = new_str.replace(symbols[i], val)
            new_str = new_str.replace(NOT_SYMBOL, "not ")
            new_str = new_str.replace(OR_SYMBOL, "or")
            new_str = new_str.replace(AND_SYMBOL, "and ")

        if eval(new_str) == True:
            sol = bin_rep
            break

    if not sol:
        return None

    solution_map = dict()
    for i in range(power):
        solution_map[symbols[i]] = bool(int(sol[i]))
    return solution_map


# Tests
assert get_solution(
    "(¬c OR b) AND (b OR c) AND (¬b OR c) AND (¬c OR ¬a)",
    ['a', 'b', 'c']
) == {'a': False, 'b': True, 'c': True}

```



Question 331

This problem was asked by LinkedIn.

You are given a string consisting of the letters `x` and `y`, such as `xyxxxyxyy`. In addition, you have an operation called flip, which changes a single `x` to `y` or vice versa.

Determine how many times you would need to apply this operation to ensure that all `x`'s come before all `y`'s. In the preceding example, it suffices to flip the second and sixth characters, so you should return `2`.

Answer 331





```

def get_flip_count(string):
    strlen = len(string)

    last_x_pos, first_y_pos = strlen, -1
    for i in range(strlen):
        if string[i] == 'y':
            first_y_pos = i
            break
    for i in range(strlen):
        index = strlen - i - 1
        if string[index] == 'x':
            last_x_pos = index
            break

    x_count, y_count = 0, 0
    for i in range(last_x_pos):
        if string[i] == 'y':
            y_count += 1
    for i in range(first_y_pos + 1, strlen):
        if string[i] == 'x':
            x_count += 1

    return min(x_count, y_count)


# Tests
assert get_flip_count("xyxxxyxyy") == 2

```



Question 332

This problem was asked by Jane Street.

Given integers `M` and `N`, write a program that counts how many positive integer pairs `(a, b)` satisfy the following conditions:

```
a + b = M
a XOR b = N
```

Answer 332





```

def get_variables(m, n):
    candidates = list()
    for a in range(m // 2 + 1):
        b = m - a
        if a ^ b == n:
            candidates.append((a, b))

    return candidates


# Tests
assert get_variables(100, 4) == [(48, 52)]

```



Question 333

This problem was asked by Pinterest.

At a party, there is a single person who everyone knows, but who does not know anyone in return (the "celebrity"). To help figure out who this is, you have access to an `O(1)` method called `knows(a, b)`, which returns `True` if person `a` knows person `b`, else `False`.

Given a list of `N` people and the above operation, find a way to identify the celebrity in `O(N)` time.

Answer 333





```

import random


def knows(known, a, b):
    return b in known[a]


def get_celeb(known):
    celeb_candidates = set(known.keys())

    while celeb_candidates:
        sample = next(iter(celeb_candidates))
        celeb_candidates.remove(sample)
        count = len(celeb_candidates)
        for other in celeb_candidates:
            if not knows(known, sample, other):
                count -= 1
        if count == 0:
            return sample


# Tests
known = {
    'a': {'b', 'c'},
    'b': set(),
    'c': {'b'}
}
assert knows(known, 'a', 'b')
assert not knows(known, 'b', 'a')
assert knows(known, 'c', 'b')
assert get_celeb(known) == 'b'

```



Question 334

This problem was asked by Twitter.

The `24` game is played as follows. You are given a list of four integers, each between `1` and `9`, in a fixed order. By placing the operators `+`, `-`, `*`, and `/` between the numbers, and grouping them with parentheses, determine whether it is possible to reach the value `24`.

For example, given the input `[5, 2, 7, 8]`, you should return True, since `(5 * 2 - 7) * 8 = 24`.

Write a function that plays the `24` game.

Answer 334





```

OPERATORS = {'+', '-', '*', '/'}
TARGET = 24


def possible(arr):
    if len(arr) == 1:
        return arr[0] == TARGET

    new_possibilities = list()
    for si in range(len(arr) - 1):
        for operator in OPERATORS:
            num_1 = arr[si]
            num_2 = arr[si + 1]
            try:
                possibility = \
                    arr[:si] + \
                    [eval("{}{}{}".format(num_1, operator, num_2))] + \
                    arr[si + 2:]
                new_possibilities.append(possibility)
            except Exception:
                pass

    return any([possible(x) for x in new_possibilities])


# Tests
assert possible([5, 2, 7, 8])
assert not possible([10, 10, 10, 10])

```



Question 335

This problem was asked by Google.

PageRank is an algorithm used by Google to rank the importance of different websites. While there have been changes over the years, the central idea is to assign each site a score based on the importance of other pages that link to that page.

More mathematically, suppose there are `N` sites, and each site `i` has a certain count `Ci` of outgoing links. Then the score for a particular site `Sj` is defined as :

```
score(Sj) = (1 - d) / N + d * (score(Sx) / Cx+ score(Sy) / Cy+ ... + score(Sz) / Cz))
```

Here, `Sx, Sy, ..., Sz` denote the scores of all the other sites that have outgoing links to `Sj`, and `d` is a damping factor, usually set to around `0.85`, used to model the probability that a user will stop searching.

Given a directed graph of links between various websites, write a program that calculates each site's page rank.

Answer 335





```

from typing import Set, Dict

DAMPING = 0.85


class Graph:
    def __init__(self):
        self.edges = dict()

    def __repr__(self):
        return str(self.edges)

    def add_edge(self, src, tgt):
        if not src in self.edges:
            self.edges[src] = set()
        if not tgt in self.edges:
            self.edges[tgt] = set()

        self.edges[src].add(tgt)


def calculate_score(node: str, g: Graph, page_scores: Dict):
    agg_score = 0
    for other in g.edges:
        if node in g.edges[other]:
            agg_score += page_scores[other] / len(g.edges[other])

    score = ((1 - DAMPING) / len(g.edges)) + (DAMPING * agg_score)
    return score


def rank_pages(g: Graph):
    page_scores = dict()
    start_prob = 1 / len(g.edges)
    for node in g.edges.keys():
        page_scores[node] = start_prob

    for node in g.edges.keys():
        page_scores[node] = calculate_score(node, g, page_scores)

    return page_scores


# Tests
g = Graph()
g.add_edge('a', 'b')
g.add_edge('a', 'c')
g.add_edge('b', 'c')
print(rank_pages(g))

```



Question 336

This problem was asked by Microsoft.

Write a program to determine how many distinct ways there are to create a max heap from a list of `N` given integers.

For example, if `N = 3`, and our integers are `[1, 2, 3]`, there are two ways, shown below.

```
  3      3
 / \    / \
1   2  2   1
```

Answer 336





```

class Node:
    def __init__(self, val):
        self.val = val
        self.l, self.r = None, None


def get_distinct_ways(node):
    if node and node.l and node.r:
        return 2 * get_distinct_ways(node.l) * get_distinct_ways(node.r)

    return 1


# Tests
a = Node(3)
b = Node(2)
c = Node(1)
a.l = b
a.r = c
assert get_distinct_ways(a) == 2

```



Question 337

This problem was asked by Apple.

Given a linked list, uniformly shuffle the nodes. What if we want to prioritize space over time?

Answer 337





```

from random import shuffle


class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        string = "["
        node = self
        while node:
            string += "{} ->".format(node.val)
            node = node.next
        string += "None]"
        return string


def get_nodes(values):
    next_node = None
    for value in values[::-1]:
        node = Node(value)
        node.next = next_node
        next_node = node

    return next_node


def get_list(head):
    node = head
    nodes = list()
    while node:
        nodes.append(node.val)
        node = node.next
    return nodes


def unishuffle(llist):
    length = 0
    node = ll
    node_list = list()
    while node:
        node_list.append(node)
        node = node.next
        length += 1

    shuffle(node_list)

    dummy = Node(None)
    for node in node_list:
        node.next = dummy.next
        dummy.next = node

    return dummy.next


# Tests
ll = get_nodes([1, 2, 3, 4, 5])
print(unishuffle(ll))

```



Question 338

This problem was asked by Facebook.

Given an integer `n`, find the next biggest integer with the same number of `1`-bits on. For example, given the number `6` (`0110` in binary), return `9` (`1001`).

Answer 338





```

def get_ones(num: int):
    binary = str(bin(num))
    count = 0
    for ch in binary:
        if ch == '1':
            count += 1

    return count


def get_next(num: int):
    inc = 1
    base_count = get_ones(num)
    while True:
        next_num = num + inc
        new_count = get_ones(next_num)
        if base_count == new_count:
            return next_num
        inc += 1


# Tests
assert get_next(6) == 9

```



Question 339

This problem was asked by Microsoft.

Given an array of numbers and a number `k`, determine if there are three entries in the array which add up to the specified number `k`. For example, given `[20, 303, 3, 4, 25]` and `k = 49`, return true as `20 + 4 + 25 = 49`.

Answer 339





```

def get_twos_sum(result, arr):
    i, k = 0, len(arr) - 1
    while i < k:
        a, b = arr[i], arr[k]
        res = a + b
        if res == result:
            return (a, b)
        elif res < result:
            i += 1
        else:
            k -= 1


def get_threes_sum(result, arr):
    arr.sort()
    for i in range(len(arr)):
        c = arr[i]
        if c > result:
            continue
        twos = get_twos_sum(result - c, arr[:i] + arr[i+1:])
        if twos:
            return True

    return get_twos_sum(result, arr)


# Tests
assert get_threes_sum(49, [20, 303, 3, 4, 25])
assert not get_threes_sum(50, [20, 303, 3, 4, 25])

```



Question 340

This problem was asked by Google.

Given a set of points `(x, y)` on a 2D cartesian plane, find the two closest points. For example, given the points `[(1, 1), (-1, -1), (3, 4), (6, 1), (-1, -6), (-4, -3)]`, return `[(-1, -1), (1, 1)]`.

Answer 340





```

import sys


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "[x={},y={}]".format(self.x, self.y)


def get_distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def get_closest(point_tuples):
    points = [Point(x, y) for (x, y) in point_tuples]
    min_dist, min_dist_pts = sys.maxsize, None
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            dist = get_distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                min_dist_pts = ((points[i].x, points[i].y),
                                (points[j].x, points[j].y))

    return min_dist_pts


# Tests
assert get_closest([(1, 1), (-1, -1), (3, 4), (6, 1), (-1, -6), (-4, -3)]) == \
    ((1, 1), (-1, -1))

```



Question 341

This problem was asked by Google.

You are given an N by N matrix of random letters and a dictionary of words. Find the maximum number of words that can be packed on the board from the given dictionary.

A word is considered to be able to be packed on the board if:
- It can be found in the dictionary
- It can be constructed from untaken letters by other words found so far on the board
- The letters are adjacent to each other (vertically and horizontally, not diagonally).
- Each tile can be visited only once by any word.

For example, given the following dictionary:
```
{ 'eat', 'rain', 'in', 'rat' }
```

and matrix:
```
[['e', 'a', 'n'],
 ['t', 't', 'i'],
 ['a', 'r', 'a']]
```

Your function should return 3, since we can make the words 'eat', 'in', and 'rat' without them touching each other. We could have alternatively made 'eat' and 'rain', but that would be incorrect since that's only 2 words.

Answer 341





```

import itertools


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return "[Cell(x={},y={})]".format(self.x, self.y)


def get_adj_options(matrix, size, row, col, seen):
    adj_options = list()
    if row > 0:
        adj_options.append(Cell(row - 1, col))
    if row < size - 1:
        adj_options.append(Cell(row + 1, col))
    if col > 0:
        adj_options.append(Cell(row, col - 1))
    if col < size - 1:
        adj_options.append(Cell(row, col + 1))

    adj_options = [x for x in adj_options if x not in seen]
    return adj_options


def consume_word(start, word, matrix, size, seen_cells):
    # print("Consuming. start:{}, word:{}, seen:{}".format(start, word, seen_cells))
    if not word:
        return [(True, seen_cells)]
    if matrix[start.x][start.y] != word[0]:
        return [(False, set())]

    seen = seen_cells | {start}
    adj_cells = get_adj_options(matrix, size, start.x, start.y, seen)

    results = list()
    for adj_cell in adj_cells:
        result = consume_word(adj_cell, word[1:], matrix, size, seen)
        results.extend(result)

    return results


def get_max_packed(matrix, dictionary, size, seen, packed):
    cell_occ = dict()
    for word in dictionary:
        cell_occ[word] = list()
        for i in range(size):
            for k in range(size):
                possibilities = consume_word(
                    Cell(i, k), word, matrix, size, set())
                consumed = [y for (x, y) in possibilities if x]
                if consumed:
                    cell_occ[word].extend(consumed)

    max_perm_length = len(dictionary)
    while max_perm_length > 0:
        all_seen = set()
        perms = itertools.combinations(dictionary, max_perm_length)
        for perm in perms:
            count_words = 0
            for word in perm:
                independent, val = False, None
                for poss in cell_occ[word]:
                    if poss & all_seen:
                        continue
                    independent, val = True, poss
                if independent:
                    all_seen |= val
                    count_words += 1
            if count_words == max_perm_length:
                return max_perm_length

        max_perm_length -= 1

    return 1


def get_max_packed_helper(matrix, dictionary):
    print(get_max_packed(matrix, dictionary, len(matrix), set(), set()))


# Tests
matrix = \
    [
        ['e', 'a', 'n'],
        ['t', 't', 'i'],
        ['a', 'r', 'a']
    ]
dictionary = {'eat', 'rain', 'in', 'rat'}
get_max_packed_helper(matrix, dictionary)

```



Question 342

This problem was asked by Stripe.

`reduce` (also known as `fold`) is a function that takes in an array, a combining function, and an initial value and builds up a result by calling the combining function on each element of the array, left to right. For example, we can write `sum()` in terms of reduce:

```python
def add(a, b):
    return a + b
```

```python
def sum(lst):
    return reduce(lst, add, 0)
```

This should call add on the initial value with the first element of the array, and then the result of that with the second element of the array, and so on until we reach the end, when we return the sum of the array.

Implement your own version of reduce.

Answer 342





```

def reduce(lst, f, init):
    res = init
    for x in lst:
        res = f(res, x)
    return res


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def custom_sum(lst):
    return reduce(lst, add, 0)


def custom_prod(lst):
    return reduce(lst, multiply, 1)


# Tests
assert custom_sum([1, 2, 3, 4]) == 10
assert custom_prod([1, 2, 3, 4]) == 24

```



Question 343

This problem was asked by Google.

Given a binary search tree and a range `[a, b]` (inclusive), return the sum of the elements of the binary search tree within the range.

For example, given the following tree:
```
    5
   / \
  3   8
 / \ / \
2  4 6  10
```

and the range `[4, 9]`, return `23 (5 + 4 + 6 + 8)`.

Answer 343





```

class Node:
    def __init__(self, val):
        self.val = val
        self.l = None
        self.r = None


def sum_range(node, lo, hi):
    if not node:
        return 0
    elif node.val < lo:
        return sum_range(node.r, lo, hi)
    elif node.val > hi:
        return sum_range(node.l, lo, hi)

    return node.val + sum_range(node.l, lo, hi) + sum_range(node.r, lo, hi)


# Tests
a = Node(5)
b = Node(3)
c = Node(8)
d = Node(2)
e = Node(4)
f = Node(6)
g = Node(10)
a.l = b
a.r = c
b.l = d
b.r = e
c.l = f
c.r = g
assert sum_range(a, 4, 9) == 23

```



Question 344

This problem was asked by Adobe.

You are given a tree with an even number of nodes. Consider each connection between a parent and child node to be an "edge". You would like to remove some of these edges, such that the disconnected subtrees that remain each have an even number of nodes.

For example, suppose your input was the following tree:
```
   1
  / \ 
 2   3
    / \ 
   4   5
 / | \
6  7  8
```

In this case, removing the edge `(3, 4)` satisfies our requirement.

Write a function that returns the maximum number of edges you can remove while still satisfying this requirement.

Answer 344





```

class Node:
    def __init__(self, val):
        self.val = val
        self.ch = set()
        self.par = None

    def size(self):
        if not self.ch:
            return 1

        return 1 + sum([x.size() for x in self.ch])


def split(node):
    results = list()
    for child in node.ch:
        if child.size() % 2 == 0:
            new_children = [x for x in node.ch if x != child]
            node.ch = new_children
            return [node, child]
        else:
            results.extend(split(child))
    return results


def segment(nodes):
    new_nodes = list()
    count = len(nodes)
    for node in nodes:
        new_nodes = split(node)

    if len(new_nodes) == count:
        return count

    return segment(new_nodes)


# Tests
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)
f = Node(6)
g = Node(7)
h = Node(8)
d.ch = [f, g, h]
c.ch = [d, e]
a.ch = [b, c]
assert segment([a]) == 2

```



Question 345

This problem was asked by Google.

You are given a set of synonyms, such as `(big, large)` and `(eat, consume)`. Using this set, determine if two sentences with the same number of words are equivalent.

For example, the following two sentences are equivalent:
- "He wants to eat food."
- "He wants to consume food."

Note that the synonyms `(a, b)` and `(a, c)` do not necessarily imply `(b, c)`: consider the case of `(coach, bus)` and `(coach, teacher)`.

Follow-up: what if we can assume that `(a, b)` and `(a, c)` do in fact imply `(b, c)`?

Answer 345





```

def get_synonyms(pairs):
    synonyms = dict()
    for pair in pairs:
        default = min(pair)
        synonyms[pair[0]] = default
        synonyms[pair[1]] = default

    return synonyms


def are_equal(s1, s2, synonyms):
    words_1, words_2 = s1.split(), s2.split()
    if len(words_1) != len(words_2):
        return False

    def lookup(word):
        return synonyms[word] if word in synonyms else word

    for (w1, w2) in zip(words_1, words_2):
        a1 = lookup(w1)
        a2 = lookup(w2)
        if a1 != a2:
            return False

    return True


# Tests
synonyms = get_synonyms([("big", "large"), ("eat", "consume")])
assert are_equal("He wants to eat food.",
                 "He wants to consume food.",
                 synonyms)
assert not are_equal("He wants to large food.",
                     "He wants to consume food.",
                     synonyms)

```



Question 346

This problem was asked by Airbnb.

You are given a huge list of airline ticket prices between different cities around the world on a given day. These are all direct flights. Each element in the list has the format `(source_city, destination, price)`.

Consider a user who is willing to take up to `k` connections from their origin city `A` to their destination `B`. Find the cheapest fare possible for this journey and print the itinerary for that journey.

For example, our traveler wants to go from JFK to LAX with up to 3 connections, and our input flights are as follows:

```
[
    ('JFK', 'ATL', 150),
    ('ATL', 'SFO', 400),
    ('ORD', 'LAX', 200),
    ('LAX', 'DFW', 80),
    ('JFK', 'HKG', 800),
    ('ATL', 'ORD', 90),
    ('JFK', 'LAX', 500),
]
```

Due to some improbably low flight prices, the cheapest itinerary would be JFK -> ATL -> ORD -> LAX, costing $440.

Answer 346





```

import sys


def get_city_map(flights):
    city_map = dict()
    for src, dst, fare in flights:
        if dst not in city_map:
            city_map[dst] = list()
        city_map[dst].append((src, fare))

    return city_map


def get_cheapest_fare(src, tgt, max_stops, city_map, total=0, stops=0):
    if stops > max_stops:
        return sys.maxsize

    if src == tgt:
        return total

    new_tgt_fares = city_map[tgt]
    possibilities = list()
    for new_tgt, fare in new_tgt_fares:
        poss = get_cheapest_fare(
            src, new_tgt, max_stops, city_map, total + fare, stops + 1)
        possibilities.append(poss)

    return min(possibilities)


# Tests
flights = [
    ('JFK', 'ATL', 150),
    ('ATL', 'SFO', 400),
    ('ORD', 'LAX', 200),
    ('LAX', 'DFW', 80),
    ('JFK', 'HKG', 800),
    ('ATL', 'ORD', 90),
    ('JFK', 'LAX', 500),
]
city_map = get_city_map(flights)
assert get_cheapest_fare("JFK", "LAX", 3, city_map) == 440
assert get_cheapest_fare("JFK", "LAX", 0, city_map) == sys.maxsize

```



Question 347

This problem was asked by Yahoo.

You are given a string of length `N` and a parameter `k`. The string can be manipulated by taking one of the first `k` letters and moving it to the end.

Write a program to determine the lexicographically smallest string that can be created after an unlimited number of moves.

For example, suppose we are given the string `daily` and `k = 1`. The best we can create in this case is `ailyd`.

Answer 347





```

def lex_mod(string, k):
    if k > 1:
        return "".join(sorted(string))

    min_ch = min(string)
    joined = string + string
    for i, ch in enumerate(joined):
        if ch == min_ch:
            return joined[i:i + len(string)]


# Tests
assert lex_mod("daily", 2) == "adily"
assert lex_mod("daily", 1) == "ailyd"

```



Question 348

This problem was asked by Zillow.

A ternary search tree is a trie-like data structure where each node may have up to three children. Here is an example which represents the words `code`, `cob`, `be`, `ax`, `war`, and `we`.
```
       c
    /  |  \
   b   o   w
 / |   |   |
a  e   d   a
|    / |   | \ 
x   b  e   r  e
```

The tree is structured according to the following rules:
- left child nodes link to words lexicographically earlier than the parent prefix
- right child nodes link to words lexicographically later than the parent prefix
- middle child nodes continue the current word

For instance, since code is the first word inserted in the tree, and `cob` lexicographically precedes `cod`, `cob` is represented as a left child extending from `cod`.

Implement insertion and search functions for a ternary search tree.

Answer 348





```

class TernaryTree:
    def __init__(self):
        self.ch = None
        self.l = None
        self.m = None
        self.r = None

    def __repr__(self):
        return "{}->[{}-{}-{}]".format(
            self.ch if self.ch else "",
            self.m if self.m else "",
            self.l if self.l else "",
            self.r if self.r else "")

    @staticmethod
    def add_tree(loc, word):
        if not loc:
            loc = TernaryTree()
        loc.add_word(word)
        return loc

    def add_word(self, word):
        if not word:
            return
        fch = word[0]
        rem_word = word[1:]

        if not self.ch:
            self.ch = fch
            self.m = TernaryTree.add_tree(self.m, rem_word)
        elif self.ch == fch:
            self.m = TernaryTree.add_tree(self.m, rem_word)
        elif self.ch < fch:
            self.l = TernaryTree.add_tree(self.l, word)
        else:
            self.r = TernaryTree.add_tree(self.l, word)

    def search_word(self, word):
        fch = word[0]
        rem_word = word[1:]

        if fch == self.ch:
            if not rem_word and not self.m.ch:
                return True
            return self.m.search_word(rem_word)
        elif fch < self.ch:
            if not self.l:
                return False
            return self.l.search_word(word)
        else:
            if not self.r:
                return False
            return self.r.search_word(word)


# Tests
tt = TernaryTree()
tt.add_word("code")
tt.add_word("cob")
tt.add_word("be")
tt.add_word("ax")
tt.add_word("war")
tt.add_word("we")

assert tt.search_word("code")
assert not tt.search_word("cow")

```



Question 349

This problem was asked by Grammarly.

Soundex is an algorithm used to categorize phonetically, such that two names that sound alike but are spelled differently have the same representation.

Soundex maps every name to a string consisting of one letter and three numbers, like `M460`.

One version of the algorithm is as follows:
- Remove consecutive consonants with the same sound (for example, change `ck -> c`).
- Keep the first letter. The remaining steps only apply to the rest of the string.
- Remove all vowels, including `y`, `w`, and `h`.
- Replace all consonants with the following digits:
    ```
    b, f, p, v -> 1
    c, g, j, k, q, s, x, z -> 2
    d, t -> 3
    l -> 4
    m, n -> 5
    r -> 6
    ```

If you don't have three numbers yet, append zeros until you do. Keep the first three numbers.
Using this scheme, `Jackson` and `Jaxen` both map to `J250`.

Implement Soundex.

Answer 349





```

cons_repl = [
    (["b", "f", "p", "v"], 1),
    (["c", "g", "j", "k", "q", "s", "x", "z"], 2),
    (["d", "t"], 3),
    (["m", "n"], 5),
    (["l"], 4),
    (["r"], 6),
]
sim_cons = [
    {"c", "k", "s"}
]
vowel_like = {"a", "e", "i", "o", "u", "y", "w", "h"}


def build_cons_repl_map(cons_repl):
    cons_repl_map = dict()
    for chars, num in cons_repl:
        for char in chars:
            cons_repl_map[char] = num
    return cons_repl_map


def build_cons_like_map(sim_cons):
    cons_like_map = dict()
    for group in sim_cons:
        elect = min(group)
        for char in group:
            cons_like_map[char] = elect

    return cons_like_map


def soundexify(word, cons_repl_map, cons_like_map):
    word = word.lower()

    # deduplicate similar sounding consonants
    w1 = ""
    for char in word:
        if char in cons_like_map:
            char = cons_like_map[char]
        if w1 and char == w1[-1]:
            continue
        w1 += char

    # remove vowel like characters
    w2 = ""
    for char in w1[1:]:
        if char not in vowel_like:
            w2 += char
    w3 = ""

    # replace consonants with numbers
    for char in w2:
        w3 += str(cons_repl_map[char])

    # massage into final format
    w3 += "000"
    w3 = w3[:3]
    final = w1[0] + w3

    return final


# Test
cons_repl_map = build_cons_repl_map(cons_repl)
cons_like_map = build_cons_like_map(sim_cons)
c1 = soundexify("Jackson", cons_repl_map, cons_like_map)
c2 = soundexify("Jaxen", cons_repl_map, cons_like_map)
assert c1 == c2
assert c1 == "j250"

```



Question 350

This problem was asked by Uber.

Write a program that determines the smallest number of perfect squares that sum up to `N`.

Here are a few examples:
- Given `N = 4`, return `1` `(4)`
- Given `N = 17`, return `2` `(16 + 1)`
- Given `N = 18`, return `2` `(9 + 9)`

Answer 350





```

import sys


def get_sum_sq(target, squares):
    if target == 0:
        return 0
    elif not squares:
        return sys.maxsize

    original_tgt = target
    biggest_sq = squares.pop()
    tally = 0
    while target >= biggest_sq:
        tally += 1
        target -= biggest_sq

    if tally:
        return min(
            tally + get_sum_sq(target, squares.copy()),
            get_sum_sq(original_tgt, squares.copy())
        )
    else:
        return get_sum_sq(original_tgt, squares.copy())


def get_min_squares(target):
    num, sq = 1, 1
    squares = list()
    while sq <= target:
        squares.append(sq)
        num += 1
        sq = num * num

    return get_sum_sq(target, squares)


# Tests
assert get_min_squares(4) == 1
assert get_min_squares(17) == 2
assert get_min_squares(18) == 2

```



Question 351

This problem was asked by Quora.

Word sense disambiguation is the problem of determining which sense a word takes on in a particular setting, if that word has multiple meanings. For example, in the sentence "I went to get money from the bank", bank probably means the place where people deposit money, not the land beside a river or lake.

Suppose you are given a list of meanings for several words, formatted like so:
```
{
    "word_1": ["meaning one", "meaning two", ...],
    ...
    "word_n": ["meaning one", "meaning two", ...]
}
```

Given a sentence, most of whose words are contained in the meaning list above, create an algorithm that determines the likely sense of each possibly ambiguous word.

Answer 351





```

- Problem definition is incomplete
- We also need a source for which word appears in which context, to be able to infer this in the actual sentences.
- Once we have a set of strongly correlated words with each word-sense, we can search the context of a word in the target sentence.
- If there is a high overlap of those words with the already correlated words for a particular word sense, we can guess that that is the answer


```



Question 352

This problem was asked by Palantir.

A typical American-style crossword puzzle grid is an `N x N` matrix with black and white squares, which obeys the following rules:
- Every white square must be part of an "across" word and a "down" word.
- No word can be fewer than three letters long.
- Every white square must be reachable from every other white square.

The grid is rotationally symmetric (for example, the colors of the top left and bottom right squares must match).
Write a program to determine whether a given matrix qualifies as a crossword grid.

Answer 352





```

# Skipping implementation since no unit tests are provided.

## Steps
- For a white square, if the previous row/column was black/before the start of the grid
  - Check if next 2 squares to the right are white (if column). Ditto for the next 2 squares below (if row). If not for either case, the crossword is invalid.
- Enumerate set of white squres.
- Start at the first white square and do a BFS until there are no more adjacent white squares to be seen. 
  - If the set difference of all white squares and seen white squares from this process if non-empty, then the crossword is invalid.

```



Question 353

This problem was asked by Square.

You are given a histogram consisting of rectangles of different heights. These heights are represented in an input list, such that `[1, 3, 2, 5]` corresponds to the following diagram:

```
      x
      x  
  x   x
  x x x
x x x x
```

Determine the area of the largest rectangle that can be formed only from the bars of the histogram. For the diagram above, for example, this would be six, representing the `2 x 3` area at the bottom right.

Answer 353





```

def get_max_hist_area(arr, start, end):
    if start == end:
        return 0

    curr_area = (end - start) * min(arr[start:end])
    opt_1 = get_max_hist_area(arr, start, end - 1)
    opt_2 = get_max_hist_area(arr, start + 1, end)

    return max(curr_area, opt_1, opt_2)


def get_max_hist_area_helper(arr):
    return get_max_hist_area(arr, 0, len(arr))


# Tests
assert get_max_hist_area_helper([1, 3, 2, 5]) == 6

```



Question 354

This problem was asked by Google.

Design a system to crawl and copy all of Wikipedia using a distributed network of machines.

More specifically, suppose your server has access to a set of client machines. Your client machines can execute code you have written to access Wikipedia pages, download and parse their data, and write the results to a database.

Some questions you may want to consider as part of your solution are:
- How will you reach as many pages as possible?
- How can you keep track of pages that have already been visited?
- How will you deal with your client machines being blacklisted?
- How can you update your database when Wikipedia pages are added or updated?

Answer 354





```

# Wikipedia Crawler

## System
- Use the main page to generate a queue of links that is to be processed by the server.
- Each link is put into a queue that is processed on the server.
- The processing involves just monitoring and sending each link to a client.
- The client downloads and parses the page, stores them in the DB and adds new URLs to the queue if the last update date is greater that the date the item was stored in a distributed cache.

## Questions
- How will you reach as many pages as possible? 
  - Parse all URLs on each page and add them back to the queue.
- How can you keep track of pages that have already been visited?
  - Distributed cache
- How will you deal with your client machines being blacklisted?
  - Maybe EC2 instance provisioning once `x` amount of requests get an error code response.
- How can you update your database when Wikipedia pages are added or updated?
  - Handled by the current mechanism of distributed cache checking.

```



Question 355

This problem was asked by Airbnb.

You are given an array `X` of floating-point numbers `x1, x2, ... xn`. These can be rounded up or down to create a corresponding array `Y` of integers `y1, y2, ... yn`.

Write an algorithm that finds an appropriate `Y` array with the following properties:
- The rounded sums of both arrays should be equal.
- The absolute pairwise difference between elements is minimized. In other words, `|x1- y1| + |x2- y2| + ... + |xn- yn|` should be as small as possible.

For example, suppose your input is `[1.3, 2.3, 4.4]`. In this case you cannot do better than `[1, 2, 5]`, which has an absolute difference of `|1.3 - 1| + |2.3 - 2| + |4.4 - 5| = 1`.

Answer 355





```

import sys
import math


def round_list(fl_arr, int_arr, rsum, diff):
    if not fl_arr:
        return (diff, int_arr) if sum(int_arr) == rsum else (sys.maxsize, list())

    num = fl_arr[0]

    op_1 = int(math.ceil(num))
    diff_1, int_arr_1 = round_list(
        fl_arr[1:], int_arr + [op_1], rsum, diff + abs(op_1 - num))

    op_2 = int(math.floor(num))
    diff_2, int_arr_2 = round_list(
        fl_arr[1:], int_arr + [op_2], rsum, diff + abs(op_2 - num))

    return (diff_1, int_arr_1) if diff_1 < diff_2 else (diff_2, int_arr_2)


def round_list_helper(arr):
    rounded_sum = int(round(sum(arr), 0))
    return round_list(arr, list(), rounded_sum, 0)[1]


# Tests
assert round_list_helper([1.3, 2.3, 4.4]) == [1, 2, 5]

```



Question 356

This problem was asked by Netflix.

Implement a queue using a set of fixed-length arrays.

The queue should support `enqueue`, `dequeue`, and `get_size` operations.

Answer 356





```

# Not sure how a set of arrays provides advantages over a single array
# Regardless, their indices can be combined to act as a single large array


class Queue:
    def __init__(self, arr_size):
        self.flarr = [None for _ in range(arr_size)]
        self.start, self.end = 0, 0
        self.size = 0
        self.max_size = arr_size

    def enqueue(self, val):
        if self.size == self.max_size:
            # No more space
            return

        new_end = (self.end + 1) % self.max_size
        self.flarr[self.end] = val
        self.end = new_end
        self.size += 1

    def dequeue(self):
        if self.size == 0:
            # Nothing to dequeue
            return None

        new_start = (self.start + 1) % self.max_size
        val = self.flarr[self.start]
        self.flarr[self.start] = None
        self.start = new_start
        self.size -= 1

        return val

    def get_size(self):
        return self.size


# Tests
q = Queue(5)
assert not q.dequeue()

q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
assert q.get_size() == 3
assert q.flarr == [1, 2, 3, None, None]

assert q.dequeue() == 1
assert q.dequeue() == 2

q.enqueue(4)
q.enqueue(5)
assert q.get_size() == 3
assert q.flarr == [None, None, 3, 4, 5]

q.enqueue(6)
q.enqueue(7)
assert q.flarr == [6, 7, 3, 4, 5]

q.enqueue(8)
# no new value added
assert q.flarr == [6, 7, 3, 4, 5]
assert q.dequeue() == 3
assert q.dequeue() == 4
assert q.dequeue() == 5
assert q.dequeue() == 6

```



Question 357

This problem was asked by LinkedIn.

You are given a binary tree in a peculiar string representation. Each node is written in the form `(lr)`, where `l` corresponds to the left child and `r` corresponds to the right child.

If either `l` or `r` is null, it will be represented as a zero. Otherwise, it will be represented by a new `(lr)` pair.

Here are a few examples:
- A root node with no children: `(00)`
- A root node with two children: `((00)(00))`
- An unbalanced tree with three consecutive left children: `((((00)0)0)0)`

Given this representation, determine the depth of the tree.

Answer 357





```

def get_continuous_count(string, char):
    count, max_count = 1, 1
    for i in range(1, len(string)):
        if string[i] == string[i-1]:
            count += 1
        else:
            count = 0
        max_count = max(max_count, count)
    return max_count - 1


def get_tree_depth(string):
    c1 = get_continuous_count(string, "(")
    c2 = get_continuous_count(string, ")")
    return max(c1, c2)


# Tests
assert get_tree_depth("(00)") == 0
assert get_tree_depth("((00)(00))") == 1
assert get_tree_depth("((((00)0)0)0)") == 3

```



Question 358

This problem was asked by Dropbox.

Create a data structure that performs all the following operations in `O(1)` time:
- `plus`: Add a key with value 1. If the key already exists, increment its value by one.
- `minus`: Decrement the value of a key. If the key's value is currently 1, remove it.
- `get_max`: Return a key with the highest value.
- `get_min`: Return a key with the lowest value.

Answer 358





```

import sys


class ValueSet:
    def __init__(self):
        self.keys = set()
        self.prev = None
        self.next = None


class MagicStruct:
    def __init__(self):
        self.keys = dict()
        self.values = dict()
        self.values[0] = ValueSet()
        self.values[sys.maxsize] = ValueSet()
        self.values[0].next = self.values[sys.maxsize]
        self.values[sys.maxsize].prev = self.values[0]
        self.vhead = self.values[0]
        self.vtail = self.values[sys.maxsize]

    def plus(self, k):
        vs = None
        if k not in self.keys:
            self.keys[k] = 0
            vs = ValueSet()
            prev_vs = self.vhead
            next_vs = self.vhead.next

        v = self.keys[k] + 1
        self.keys[k] = v

        if v in self.values:
            vs = self.values[v]
            prev_vs = self.values[v].prev
        vs.keys.add(k)
        next_vs = prev_vs.next

        vs.prev = prev_vs
        vs.next = next_vs

        if v-1 in self.values:
            old_vs = self.values[v-1]
            if not old_vs.keys:
                oprev = old_vs.prev
                onext = old_vs.next
                oprev.next = onext
                onext.prev = oprev

    def minus(self, k):
        if k not in self.keys:
            return

        vs = self.keys[k]
        prev_vs = vs.prev
        next_vs = vs.next
        v = self.keys[k] - 1

        if not v:
            prev_vs.next = next_vs
            next_vs.prev = prev_vs

    def get_max(self):
        pass

    def get_min(self):
        pass


# Tests
ms = MagicStruct()
ms.plus("a")
ms.plus("b")
ms.plus("a")
ms.plus("a")
print(ms.keys, ms.values)
ms.minus("a")
print(ms.keys, ms.values)
ms.minus("b")
print(ms.keys, ms.values)
print(ms.get_max())

```



Question 359

This problem was asked by Slack.

You are given a string formed by concatenating several words corresponding to the integers zero through nine and then anagramming.

For example, the input could be 'niesevehrtfeev', which is an anagram of 'threefiveseven'. Note that there can be multiple instances of each integer.

Given this string, return the original integers in sorted order. In the example above, this would be `357`.

Answer 359





```

WORD_MAP = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def get_char_count_dict(string):
    letter_dict = dict()
    for char in string:
        if char not in letter_dict:
            letter_dict[char] = 0
        letter_dict[char] += 1

    return letter_dict


def use_digit(letter_dict, word_dict, digit):
    for char in word_dict:
        if char not in letter_dict or word_dict[char] > letter_dict[char]:
            return letter_dict, 0

    for char in word_dict:
        letter_dict[char] -= word_dict[char]

    letter_dict, uses = use_digit(letter_dict, word_dict, digit)
    return letter_dict, uses + 1


def get_sorted_nums(string):
    letter_dict = get_char_count_dict(string)

    result = 0
    for i in range(10):
        word = WORD_MAP[i]
        word_dict = get_char_count_dict(word)
        letter_dict, uses = use_digit(letter_dict, word_dict, i)

        while uses > 0:
            result = result * 10 + i
            uses -= 1

    return result


# Tests
assert get_sorted_nums("niesevehrtfeev") == 357
assert get_sorted_nums("nienienn") == 99
assert get_sorted_nums("enieniennon") == 199

```



Question 360

This problem was asked by Spotify.

You have access to ranked lists of songs for various users. Each song is represented as an integer, and more preferred songs appear earlier in each list. For example, the list `[4, 1, 7]` indicates that a user likes song `4` the best, followed by songs `1` and `7`.

Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone's priorities.

For example, suppose your input is `{[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]}`. In this case a satisfactory playlist could be `[2, 1, 6, 7, 3, 9, 5]`.

Answer 360





```

from heapq import heappush, heappop


def interleave_playlist(ranked_listings):
    scores = dict()
    for listing in ranked_listings:
        num_songs = len(listing)
        total_points = ((num_songs + 1) * num_songs) // 2
        for rank, song in enumerate(listing):
            if song not in scores:
                scores[song] = 0
            scores[song] += total_points / (rank + 1)

    sorted_scored_tuples = list()
    for song, score in scores.items():
        heappush(sorted_scored_tuples, (score, song))

    interleaved = list()
    while sorted_scored_tuples:
        _, song = heappop(sorted_scored_tuples)
        interleaved.append(song)

    return interleaved[::-1]


# Tests
ranked_listings = [
    [1, 7, 3],
    [2, 1, 6, 7, 9],
    [3, 9, 5]
]
assert interleave_playlist(ranked_listings) == [2, 1, 3, 7, 9, 6, 5]

```



Question 361

This problem was asked by Facebook.

Mastermind is a two-player game in which the first player attempts to guess the secret code of the second. In this version, the code may be any six-digit number with all distinct digits.

Each turn the first player guesses some number, and the second player responds by saying how many digits in this number correctly matched their location in the secret code. For example, if the secret code were `123456`, then a guess of `175286` would score two, since `1` and `6` were correctly placed.

Write an algorithm which, given a sequence of guesses and their scores, determines whether there exists some secret code that could have produced them.

For example, for the following scores you should return `True`, since they correspond to the secret code `123456`:
`{175286: 2, 293416: 3, 654321: 0}`

However, it is impossible for any key to result in the following scores, so in this case you should return `False`:
`{123456: 4, 345678: 4, 567890: 4}`

Answer 361





```

def does_code_match_guess(code, guess, matches):
    count = 0
    for c_char, g_char in zip(code, guess):
        if c_char == g_char:
            count += 1

    return count == matches


def is_valid_code(guess_scores):
    for i in range(1000000):
        code = str(i)
        code = ("0" * (6 - len(code))) + code

        success = True
        for guess, matches in guess_scores.items():
            success = success & does_code_match_guess(
                code, str(guess), matches)

        if success:
            return True

    return False


# Tests
assert is_valid_code({175286: 2, 293416: 3, 654321: 0})
assert not is_valid_code({123456: 4, 345678: 4, 567890: 4})

```



Question 362

This problem was asked by Twitter.

A strobogrammatic number is a positive number that appears the same after being rotated `180` degrees. For example, `16891` is strobogrammatic.

Create a program that finds all strobogrammatic numbers with N digits.

Answer 362





```

def get_strob_numbers(num_digits):
    if not num_digits:
        return [""]
    elif num_digits == 1:
        return ["0", "1", "8"]

    smaller_strob_numbers = get_strob_numbers(num_digits - 2)
    strob_numbers = list()
    for x in smaller_strob_numbers:
        strob_numbers.extend([
            "1" + x + "1",
            "6" + x + "9",
            "9" + x + "6",
            "8" + x + "8",
        ])

    return strob_numbers


# Tests
assert get_strob_numbers(1) == ['0', '1', '8']
assert get_strob_numbers(2) == ['11', '69', '96', '88']
assert get_strob_numbers(3) == ['101', '609', '906', '808', '111', '619',
                                '916', '818', '181', '689', '986', '888']

```



Question 363

Write a function, add_subtract, which alternately adds and subtracts curried arguments. Here are some sample operations:

```
add_subtract(7) -> 7
add_subtract(1)(2)(3) -> 1 + 2 - 3 -> 0
add_subtract(-5)(10)(3)(9) -> -5 + 10 - 3 + 9 -> 11
```

Answer 363





```

def add(x, y=1):
    return lambda z: add(x+(y*z), (-1*y)) if z else x


# Tests
assert add(7)(None) == 7
assert add(1)(2)(3)(None) == 0
assert add(-5)(10)(3)(9)(None) == 11

```



Question 364

This problem was asked by Facebook.

Describe an algorithm to compute the longest increasing subsequence of an array of numbers in `O(n log n)` time.

Answer 364





```

def get_longest_inc_subsq(arr):
    longest = []
    start = 0
    for i in range(len(arr)):
        if arr[i] < arr[i - 1]:
            start = i
        end = i + 1

        if end - start > len(longest):
            longest = arr[start:end]

    return longest


# Tests
assert get_longest_inc_subsq([1, 2, 3, 5, 4]) == [1, 2, 3, 5]
assert get_longest_inc_subsq([5, 4, 3, 2, 1]) == [5]

```



Question 365

This problem was asked by Google.

A quack is a data structure combining properties of both stacks and queues. It can be viewed as a list of elements written left to right such that three operations are possible:
- `push(x)`: add a new item `x` to the left end of the list
- `pop()`: remove and return the item on the left end of the list
- `pull()`: remove the item on the right end of the list.

Answer 365





```

# no idea how to implement it using 3 stacks
# the time complexity constraints can be satisfied
# using a deque instead

from collections import deque

# Tests
dq = deque()
dq.appendleft(1)
dq.appendleft(2)
dq.appendleft(3)
dq.appendleft(4)
dq.appendleft(5)
assert dq.pop() == 1
assert dq.popleft() == 5

```
