# Code Puzzles 

## Reverse a Linked List 

``` python
Curr=head
while curr:
    Next = curr.next
    Curr.next = prev
    Prev = curr
    Curr = next
return prev (new head)
```

## Longest Palindrome 

``` python
def longest_palindrome(s: str):
    if not s:
       return ""
    longest = ""
    for i in range(len(s)):
           # odd case, like "aba"
         tmp = helper(s, i, i)
         if len(tmp) > len(longest):
              # update result
              longest = tmp

         # even case, like "abba"
         tmp = helper(s, i, i+1)
         if len(tmp) > len(longest):
              longest = tmp
   return longest

def helper(s: str, l: int, r: int):
    while l >= 0 and r < len(s) and s[l] == s[r]:
       l -= 1 # decrement the left
       r += 1 # increment the right
    return s[l+1:r]

```

## BFS (not recursive) 

``` python
# Visit adjacent unvisited vertex. 
# - Mark it as visited. Display it. Insert it in a queue.
# If no adjacent vertex, pop vertex off queue
# Repeat Rule 1 and Rule 2 until the queue is empty. 

def bfs(graph, current_node):
    visited = []
    queue = [current_node]

    while queue:
        s = queue.pop(0)
        print(s)
        for neighbor in graph[s]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
bfs(graph, 'A')
```

## DFS (not recursive) 

``` python
# add unvisited nodes to stack
def dfs(graph, start_vertex):
    visited = set()
    traversal = []
    stack = [start_vertex]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            traversal.append(vertex)
            stack.extend(reversed(graph[vertex])) # add in same order as visited
    return traversal

```

## Check if binary trees are equal: 

``` python
def are_identical(root1, root2):
  if root1 == None and root2 == None:
    return True

  if root1 != None and root2 != None:
    return (root1.data == root2.data and
              are_identical(root1.left, root2.left) and
              are_identical(root1.right, root2.right))

  return False
```

<!--
 ## Etc 

``` python
```
 -->
