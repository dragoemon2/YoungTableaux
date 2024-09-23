import copy
import itertools
import math
from functools import lru_cache

import more_itertools


def _enumerate_smaller_division_dominance_order(larger_division: list[int], length_limit : int | None = None):
    '''
    支配的順序の意味でdivisionより小さい分割を列挙する(division自身も含む)
    '''
    yield from _enumerate_smaller_division_dominance_order__recursive([], 0, list(itertools.accumulate(larger_division)), length_limit)

def _enumerate_smaller_division_dominance_order__recursive(current_division: list[int], current_division_sum: int,  accumulated_larger_division: list[int], length_limit : int | None = None):
    '''
    _enumerate_smaller_division_dominance_order内の再帰関数
    '''
    if current_division_sum == accumulated_larger_division[-1]:
        yield current_division
        return

    if length_limit is not None and len(current_division) >= length_limit:
        return
    
    index = len(current_division)
    max_value = accumulated_larger_division[index if index < len(accumulated_larger_division) else -1] - current_division_sum
    if index != 0:
        max_value = min(max_value, current_division[-1])
    
    for value in range(1, max_value+1):
        new_division = current_division + [value]
        new_division_sum = current_division_sum + value
        yield from _enumerate_smaller_division_dominance_order__recursive(new_division, new_division_sum, accumulated_larger_division, length_limit)


def _enumerate_tableaux_with__recursive(current_tableau, current_value, weights, max_division):
    '''
    enumerate_tableaux_with内部で再帰的に呼び出される関数
    '''
    weight = weights[current_value-1]

    for new_division in _expand_division_iter(current_tableau.shape.division, weight, max_division):
        new_numbers = copy.deepcopy(current_tableau.numbers)
        new_numbers.append([])
        for i in range(len(new_division)):
            new_numbers[i] += [current_value] * (new_division[i] - len(new_numbers[i]))
        while new_numbers and not new_numbers[-1]:
            new_numbers.pop()
        new_tableau = YoungTableaux(new_numbers)
        if current_value == len(weights):
            yield new_tableau
        else:
            yield from _enumerate_tableaux_with__recursive(new_tableau, current_value+1, weights, max_division)

def _sequence_iter(n, max_sequence):
    '''
    max_sequence以下の数列で、和がnになるものを列挙する
    '''
    if n == 0:
        yield [0]*len(max_sequence)
        return
    if len(max_sequence) == 1:
        if n <= max_sequence[0]:
            yield [n]
        return
    for i in range(min(n, max_sequence[0])+1):
        for seq in _sequence_iter(n-i, max_sequence[1:]):
            yield [i] + seq
    
def _expand_division_iter(division, p, max_division=None):
    '''
    ヤング図形にp個の箱を追加したヤング図形のうち、次の条件を満たすものを列挙する
    - 追加した箱のどの2つも同じ列にない
    - max_divisionが指定された場合、それを超えない
    '''

    division = division + [0] # 末尾に0を追加しておく

    # 各行に対して最大で追加できる箱の数を求める
    max_to_add = [p]
    for i in range(1, len(division)):
        max_to_add.append(division[i-1] - division[i])

    # max_divisionが指定された場合、それを超えないようにmax_to_addを調整する
    if max_division is not None:
        for i in range(min(len(max_division), len(max_to_add))):
            max_to_add[i] = min(max_to_add[i], max_division[i] - division[i])

    if len(max_to_add) > len(max_division):
        max_to_add = max_to_add[:len(max_division)]

    # 後ろから決める方が効率が良い
    max_to_add.reverse() 

    # 0からmax_to_addまでの数列を列挙し、divisionに足したものを返していく
    for seq in _sequence_iter(p, max_to_add):
        seq.reverse()
        new_division = [division[i] + seq[i] for i in range(len(max_to_add))]
        while new_division and new_division[-1] == 0:
            new_division.pop()
        yield new_division

class YoungDiagram:
    '''
    ヤング図形を表すクラス
    '''
    def __init__(self, division : list[int]):
        if isinstance(division, YoungDiagram):
            division = division.division
        self.division = division
        self.n = sum(division)
        if not self._check_diagram():
            raise ValueError("Invalid Young diagram")
    
    def _check_diagram(self):
        if len(self.division) == 0:
            return True
        
        for i in range(1, len(self.division)):
            if self.division[i] > self.division[i-1]:
                return False
            
        if self.division[-1] < 1:
            return False
        return True
    
    @lru_cache(maxsize=None)
    def conjugate(self):
        '''
        共役を求める
        '''
        new_division = []
        for i in range(self.division[0]):
            count = 0
            for j in self.division:
                if j >= (i+1):
                    count += 1
                else:
                    break

            new_division.append(count)

        return YoungDiagram(new_division)
    
    def include(self, other):
        '''
        selfがotherを含むかどうか
        '''
        if isinstance(other, tuple): # 箱(i, j)が与えられた場合
            i, j = other
            if i >= len(self.division):
                return False
            if j >= self.division[i]:
                return False
            return True
        
        elif isinstance(other, YoungDiagram): # 別のヤング図形が与えられた場合
            if self.n < other.n:
                return False
            
            if len(self.division) < len(other.division):
                return False

            for i in range(len(other.division)):
                if self.division[i] < other.division[i]:
                    return False

            return True
        
    def __contains__(self, other):
        return self.include(other)
    
    def __eq__(self, other):
        return self.division == other.division
        
    def __str__(self, fill='■'):
        result = ''
        for i in range(len(self.division)):
            result += fill * self.division[i] + '\n'
        return result[:-1]
    
    def __hash__(self):
        return hash(tuple(self.division))
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.division})'
    
    def __bool__(self):
        return len(self.division) > 0
        
    def enumerate_boxes(self):
        '''
        ヤング図形の箱を列挙する
        '''
        for i in range(len(self.division)):
            for j in range(self.division[i]):
                yield (i, j)

    def enumerate_corners(self):
        '''
        ヤング図形の角を列挙する
        '''
        for i in range(len(self.division)):
            j = self.division[i] - 1
            if self.is_corner((i, j)):
                yield (i, j)

    def enumerate_outer_boxes(self):
        '''
        ヤング図形の外側の箱であって、それを追加するとヤング図形になるものを列挙する
        '''
        if len(self.division) == 0:
            yield (0, 0)
            return

        yield (0, self.division[0])
        for i in range(1, len(self.division)):
            if self.division[i] < self.division[i-1]:
                yield (i, self.division[i])
        yield (len(self.division), 0)

    def add_box(self, i):
        '''
        ヤング図形に箱を追加する
        '''
        new_division = self.division.copy()
        new_division[i] += 1
        return YoungDiagram(new_division)

    def is_corner(self, corner):
        '''
        cornerがヤング図形の角かどうか
        '''
        i, j = corner
        return not self.include((i+1, j)) and not self.include((i, j+1)) and self.include((i, j))

    def hook_length(self, box: tuple[int, int]):
        '''
        boxのhook lengthを求める
        '''
        i, j = box
        return self.division[i] + self.conjugate().division[j] - i - j - 1
    
    @lru_cache()
    def number_of_standard_tableaux(self):
        '''
        標準Young tableauxの個数を求める (hook length formula)
        '''
        prod = 1
        for box in self.enumerate_boxes():
            prod *= self.hook_length(box)
        
        return math.factorial(self.n) // prod
    
    @lru_cache()
    def number_of_tableaux(self, max_value):
        '''
        Young tableauxの個数を求める [Stanley (1971)]
        '''
        denominator_prod = 1
        numerator_prod = 1
        for (i, j) in self.enumerate_boxes():
            denominator_prod *= self.hook_length((i, j))
            numerator_prod *= max_value + j - i
        return numerator_prod // denominator_prod
    
    def enumerate_tableaux(self, max_value):
        '''
        ヤング図形に対応するYoung tableauxを列挙するイテレータ
        '''
        for division in _enumerate_smaller_division_dominance_order(self.division, max_value):
            division += [0] * (max_value - len(division))
            for weights in more_itertools.distinct_permutations(division):
                yield from self.enumerate_tableaux_with(list(weights))
    
    def enumerate_standard_tableaux(self):
        '''
        標準Young tableauxを列挙するイテレータ
        '''
        for tableau in self.enumerate_tableaux_with([1]*self.n):
            yield tableau

    def enumerate_tableaux_with(self, weights):
        '''
        ヤング図形に対応するYoung tableauxで、重みが与えられたものを列挙するイテレータ
        '''
        if isinstance(weights, YoungDiagram):
            weights = weights.division
        elif isinstance(weights, dict):
            if dict == {}:
                weights = []
            else:
                max_num = max(weights.keys())
                weights = [weights.get(i, 0) for i in range(1,max_num+1)]
        elif isinstance(weights, list):
            weights = weights
        else:
            raise ValueError('Invalid type')
        
        if self.n != sum(weights):
            return
        
        for tableau in _enumerate_tableaux_with__recursive(YoungTableaux([]), 1, weights, self.division):
            yield tableau

    

    @classmethod
    def enumerate_diagrams(cls, n):
        '''
        n個の箱を持つヤング図形を列挙する
        '''
        for division in _enumerate_smaller_division_dominance_order([n]):
            yield YoungDiagram(division)

            

class NumberedYoungDiagram(YoungDiagram):
    '''
    番号付けされたヤング図形
    '''
    def __init__(self, numbers : list[list[int]]):
        while numbers and not numbers[-1]:
            numbers.pop()

        self.numbers = numbers
        if isinstance(numbers, NumberedYoungDiagram):
            numbers = numbers.numbers
        division = [len(line) for line in numbers]
        self.shape = YoungDiagram(division)
        super().__init__(division)
        if not self._check_numbers():
            raise ValueError("Invalid Numbered Young diagram")

    def _check_numbers(self):
        for line in self.numbers:
            for v in line:
                if not isinstance(v, int):
                    return False
                if v <= 0:
                    return False
                
        return True
    
    def to_word(self, direction='row'):
        '''
        YoungTableauxをwordに変換する
        '''
        if direction == 'row':
            result = []
            for line in self.numbers:
                result = line + result

            return result
        
        elif direction == 'column':
            result = []
            for j in range(len(self.numbers[0])):
                for i in range(len(self.numbers)-1, -1, -1):
                    if self.include((i, j)):
                        result.append(self.numbers[i][j])

            return result

        else:
            raise ValueError(f'Invalid direction: {direction}')
        
    def transpose(self):
        '''
        転置を求める
        '''
        new_numbers = []
        for j in range(len(self.numbers[0])):
            new_numbers.append([line[j] for line in self.numbers if j < len(line)])
        return self.__class__(new_numbers)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.numbers[index]
        
        elif isinstance(index, tuple):
            return self.numbers[index[0]][index[1]]
        
        else:
            raise TypeError('Invalid index type')
        
    def get(self, i, j):
        if self.shape.include((i, j)):
            return self.numbers[i][j]
        else:
            return None
        
    def __hash__(self):
        return hash(tuple([tuple(line) for line in self.numbers]))
        
    def __eq__(self, other):
        return self.numbers == other.numbers
        
    def __str__(self):
        max_len = max([len(str(i)) for line in self.numbers for i in line])
        result = ''
        for line in self.numbers:
            for i in line:
                result += str(i).rjust(max_len) + ' '
            result = result[:-1] + '\n'
        return result[:-1]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.numbers})'
    
    def latex(self, printer=None):
        if printer is None:
            printer = lambda v: str(v) if len(str(v)) == 1 else r'{\scriptstyle %d}' % v
        numbers_str = [[printer(v) for v in line] for line in self.numbers]
        text = ','.join([''.join(line) for line in numbers_str])
        text = r'\ytableaushort[]{%s}' % text
        return text
    
    def render(self, filename, dpi=30):
        '''
        ヤング図形を描画する
        '''
        import matplotlib.pyplot as plt

        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Draw the boxes and numbers
        for i, row_length in enumerate(self.division):
            for j in range(row_length):
                # Draw the box
                rect = plt.Rectangle((j, -i-1), 1, 1, fill=None, edgecolor='black', linewidth=3)
                ax.add_patch(rect)
                
                # Add the number
                ax.text((j + 0.5), (-i - 0.5), str(self.numbers[i][j]), ha='center', va='center', fontsize=40)
        
        # Set the limits and aspect ratio
        ax.set_xlim(0,(self.division[0]+1))
        ax.set_ylim((-len(self.division)-1), 0)
        ax.set_aspect('equal')
        
        
        # Remove axes
        ax.axis('off')

        print(filename)
        
        # Show the plot
        plt.savefig(filename,transparent=True, dpi=dpi)

    def max_value(self):
        '''
        最大値を求める
        '''
        values = [v for line in self.numbers for v in line]
        if not values:
            return 0
        return max(values)

    def add_box(self, i, number):
        '''
        ヤング図形に箱を追加する
        '''
        new_numbers = copy.deepcopy(self.numbers)
        if i == len(new_numbers):
            new_numbers.append([])
        new_numbers[i].append(number)
        return self.__class__(new_numbers)

def _bump_line(line : list[int], value : int) -> tuple[list[int], int | None]:
    '''
    行に対してbumpを行う
    '''
    result = line.copy()

    if line[-1] <= value:
        result.append(value)
        return result, None

    for i in range(len(line) - 1, -1, -1):
        if line[i] <= value:
            index = i + 1
            break
    else:
        index = 0

    bumped = result[index]
    result[index] = value

    return result, bumped

def _reverse_bump_line(line : list[int], value : int) -> tuple[list[int], int]:
    '''
    行に対してbumpの逆操作を行う
    '''
    result = line.copy()

    for i in range(len(line)):
        if line[i] >= value:
            index = i - 1
            break
    else:
        index = len(line) - 1

    bumped = result[index]
    result[index] = value

    return result, bumped
    
class YoungTableaux(NumberedYoungDiagram):
    '''
    番号付けされたヤング図形
    '''
    def __init__(self, numbers : list[list[int]]):
        super().__init__(numbers)
        if not self._check_tableaux():
            raise ValueError("Invalid Young tableaux")

    def _check_tableaux(self):
        for i in range(len(self.division)-1):
            for j in range(self.division[0] - 1):
                if self.include((i,j)) and self.include((i+1,j)) and self[i,j] >= self[i+1,j]:
                    return False
                if self.include((i,j)) and self.include((i,j+1)) and self[i,j] > self[i,j+1]:
                    return False
                
        return True
    
    def row_bump(self, value):
        '''
        bumpを行う
        '''
        new_numbers = copy.deepcopy(self.numbers)

        index = -1
        for index, line in enumerate(new_numbers):
            line, value = _bump_line(line, value)
            new_numbers[index] = line

            if value is None:
                break

        if value is not None:
            index += 1
            new_numbers.append([value])

        return YoungTableaux(new_numbers), index

    def reverse_row_bump(self, index):
        '''
        行bumpの逆操作を行う
        '''
        new_numbers = copy.deepcopy(self.numbers)
        value = new_numbers[index].pop()

        for i in range(index-1, -1, -1):
            line, value = _bump_line(new_numbers[i], value)
            new_numbers[i] = line

        return YoungTableaux(new_numbers), value

    
    
    

    @classmethod
    def from_word(cls, word, method='bump'):
        '''
        wordから、それとknuth同値なwordを持つYoungTableauxを生成する
        '''
        if method == 'bump':
            result = YoungTableaux([])
            for value in word:
                result, _ = result.row_bump(value)
            return result
            
    def __mul__(self, other):
        '''
        2つのYoungTableauxの積を求める (*演算子)
        もしくは、YoungTableauxと整数の積(タブロー環の元)を求める
        '''
        result = YoungTableaux(self.numbers)
        for value in other.to_word():
            result, _ = result.row_bump(value)
        return result
        
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__mul__(other)
        else:
            raise TypeError('Invalid type')
        
    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError('Invalid type')
        elif n < 0:
            raise ValueError('Invalid value')
        elif n == 0:
            return YoungTableaux([])
        
        #繰り返し二乗法
        result = YoungTableaux([])
        current = self
        while True:
            if n & 1:
                result = result * current
            if n <= 1:
                break
            current = current * current
            n >>= 1
        return result
    
    
    def is_standard(self):
        '''
        standard tableauxかどうか
        '''
        for i in range(len(self.division)-1):
            for j in range(self.division[0] - 1):
                if self.include((i,j)) and self.include((i+1,j)) and self[i,j] >= self[i+1,j]:
                    return False
                if self.include((i,j)) and self.include((i,j+1)) and self[i,j] >= self[i,j+1]:
                    return False
                
        set_numbers = set(v for line in self.numbers for v in line)
        return set_numbers == set(range(1, self.n+1))
