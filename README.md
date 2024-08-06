# Young Tableaux

ヤング図形、ヤングタブロー、歪ヤング図形などを扱うためのライブラリです。

## example

```python
from young_tableaux import SkewTableaux, YoungDiagram, YoungTableaux

# Create a Young diagram
D = YoungDiagram([3,2,2])
print(D)

# calculate the number of standard Young tableaux
print(D.number_of_standard_tableaux()) 

# Create a Young tableaux
T = YoungTableaux([
    [1,2,2,3],
    [2,3,5,5],
    [5,6]
])
U = YoungTableaux([
    [1,3],
    [2]
])

# Convert the Young tableaux to a word
print(T.to_word())

# Calculate the product of two Young tableaux
print(T*U)

# Create a skew Young diagram
D = SkewTableaux([
    [0,0,0,0,1,3],
    [0,0,0,0,2],
    [1,2,2,3],
    [4,4,6],
    [5,6]
])

# Rectify the skew Young diagram to a Young diagram
print(D.rectify()) 
```
