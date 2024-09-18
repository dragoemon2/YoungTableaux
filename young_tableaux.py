import copy
import itertools
import math
from functools import lru_cache

import more_itertools


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
        for division in self._enumerate_smaller_division_dominance_order(self.division, max_value):
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
        
        for tableau in self._enumerate_tableaux_with__recursive(YoungTableaux([]), 1, weights, self.division):
            yield tableau

    @classmethod
    def _enumerate_smaller_division_dominance_order(cls, larger_division: list[int], length_limit : int | None = None):
        '''
        支配的順序の意味でdivisionより小さい分割を列挙する(division自身も含む)
        '''
        yield from cls._enumerate_smaller_division_dominance_order__recursive([], 0, list(itertools.accumulate(larger_division)), length_limit)

    @classmethod
    def _enumerate_smaller_division_dominance_order__recursive(cls, current_division: list[int], current_division_sum: int,  accumulated_larger_division: list[int], length_limit : int | None = None):
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
            yield from cls._enumerate_smaller_division_dominance_order__recursive(new_division, new_division_sum, accumulated_larger_division, length_limit)


    @classmethod
    def _enumerate_tableaux_with__recursive(cls, current_tableau, current_value, weights, max_division):
        '''
        enumerate_tableaux_with内部で再帰的に呼び出される関数
        '''
        weight = weights[current_value-1]

        for new_division in cls._expand_division_iter(current_tableau.shape.division, weight, max_division):
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
                yield from cls._enumerate_tableaux_with__recursive(new_tableau, current_value+1, weights, max_division)
        
    @classmethod
    def _sequence_iter(cls, n, max_sequence):
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
            for seq in cls._sequence_iter(n-i, max_sequence[1:]):
                yield [i] + seq
        

    @classmethod
    def _expand_division_iter(cls, division, p, max_division=None):
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
        for seq in cls._sequence_iter(p, max_to_add):
            seq.reverse()
            new_division = [division[i] + seq[i] for i in range(len(max_to_add))]
            while new_division and new_division[-1] == 0:
                new_division.pop()
            yield new_division

            

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
        return NumberedYoungDiagram(new_numbers)

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
            line, value = self._bump_line(line, value)
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
            line, value = self._reverse_bump_line(new_numbers[i], value)
            new_numbers[i] = line

        return YoungTableaux(new_numbers), value

    @staticmethod
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
    
    @staticmethod
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
        
        elif method == 'slide':
            numbers = []
            for i in range(len(word)):
                numbers.append([0]*i + [word[i]])
            numbers.reverse()
        
            skew = SkewTableaux(numbers)
            return skew.rectify()
            
    def __mul__(self, other, method='bump'):
        '''
        2つのYoungTableauxの積を求める (*演算子)
        もしくは、YoungTableauxと整数の積(タブロー環の元)を求める
        '''
        if isinstance(other, YoungTableaux):
            if method == 'bump': # bumpを使って掛け算を行う
                return self._multiply_with_bump(other)
            elif method == 'slide': # slideを使って掛け算を行う
                return self._multiply_with_slide(other)
            else:
                raise NotImplementedError('Invalid method')
        elif isinstance(other, int) or isinstance(other, float):
            result = YoungTableauxRing()
            result._add(self, other)
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

    def __matmul__(self, other):
        '''
        2つのYoungTableauxを並べたSkewTableauxを求める (@演算子)
        '''
        if len(self.numbers) == 0:
            return SkewTableaux(other.numbers)

        numbers = copy.deepcopy(other.numbers)
        width = len(self.numbers[0])
        for i in range(len(other.numbers)):
            numbers[i] = [0] * width + numbers[i]
        numbers += self.numbers

        return SkewTableaux(numbers)
    
    def _multiply_with_bump(self, other):
        '''
        bumpを使って掛け算を行う
        '''
        result = YoungTableaux(self.numbers)
        for value in other.to_word():
            result, _ = result.row_bump(value)
        return result
    
    def _multiply_with_slide(self, other):
        '''
        slideを使って掛け算を行う
        '''
        return (self @ other).rectify()
    
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


    def __add__(self, other):
        '''
        YoungTableauxの和を求める (+演算子)
        '''
        return YoungTableauxRing(self) + YoungTableauxRing(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + (-1) * other
    
class SkewDiagram:
    def __init__(self, diagram1: YoungDiagram, diagram2: YoungDiagram):
        if not isinstance(diagram1, YoungDiagram) or not isinstance(diagram2, YoungDiagram):
            raise TypeError('Invalid type')
        
        # 外側のヤング図形
        self.diagram1 = diagram1
        # 内側のヤング図形
        self.diagram2 = diagram2
        
        self.n = diagram1.n - diagram2.n
        if not self.diagram1.include(self.diagram2):
            raise ValueError("Invalid skew diagram")
        
    def include(self, other):
        '''
        selfがotherを含むかどうか
        '''
        return self.diagram1.include(other) and not self.diagram2.include(other)
    
    def __contains__(self, other):
        return self.include(other)
    
    def conjugate(self):
        '''
        共役を求める
        '''
        return SkewDiagram(self.diagram1.conjugate(), self.diagram2.conjugate())
    
    def __eq__(self, other):
        '''
        SkewDiagramが等しいかどうか判定
        '''
        for i in range(len(self.diagram1.division)):
            if i < len(self.diagram2.division):
                if self.diagram1.division[i] == self.diagram2.division[i] and other.diagram1.division[i] == other.diagram2.division[i]:
                    continue
                if self.diagram1.division[i] != other.diagram1.division[i] or self.diagram2.division[i] != other.diagram2.division[i]:
                    return False
            else:
                if self.diagram1.division[i] != other.diagram1.division[i]:
                    return False
        return True

    def __str__(self):
        result = ''
        for i in range(len(self.diagram1.division)):
            if i < len(self.diagram2.division):
                result += '□' * self.diagram2.division[i]
                result += '■' * (self.diagram1.division[i] - self.diagram2.division[i])
            else:
                result += '■' * self.diagram1.division[i]
            result += '\n'
        return result[:-1]
    

    def enumerate_littlewood_richardson_tableaux_with(self, weight: list[int] | YoungDiagram):
        '''
        TODO: 与えられた重みに対するLittlewood-Richardson skew tableauxを列挙する
        '''
        if isinstance(weight, YoungDiagram):
            weight = weight.division
        
        if len(weight) != self.n:
            return
        
    @lru_cache()
    def reverse_numbering(self):
        '''
        逆数字付けを求める
        '''
        
    def __hash__(self):
        return hash((self.diagram1, self.diagram2))
    
class NumberedSkewDiagram(SkewDiagram):
    def __init__(self, numbers: list[list[int]]):
        diagram1 = YoungDiagram([len(line) for line in numbers])
        diagram2 = YoungDiagram([line.count(0) for line in numbers if line[0] == 0])
        super().__init__(diagram1, diagram2)
        self.numbers = numbers

class SkewTableaux(NumberedSkewDiagram):
    def __init__(self, numbers: list[list[int]]):
        '''
        numbers: list[list[int]] 内タブローは0を埋める
        '''
        super().__init__(numbers)
        if not self._check_tableaux():
            raise ValueError("Invalid skew tableaux")
        
    def _check_tableaux(self):
        for i in range(len(self.numbers)-1):
            for j in range(len(self.numbers[i])):
                if self.include((i,j)) and self.include((i+1,j)) and self.numbers[i][j] >= self.numbers[i+1][j]:
                    return False
                if self.include((i,j)) and self.include((i,j+1)) and self.numbers[i][j] > self.numbers[i][j+1]:
                    return False
                
        return True
    
    def __str__(self):
        max_len = max([len(str(i)) for line in self.numbers for i in line])
        result = ''
        for line in self.numbers:
            for i in line:
                str_i = str(i) if i != 0 else '□'
                result += str_i.rjust(max_len) + ' '
            result = result[:-1] + '\n'
        return result[:-1]
    
    def slide(self, corner):
        '''
        cornerをスライドさせる
        '''
        new_numbers = copy.deepcopy(self.numbers)
        while True:
            i, j = corner
            if self.include((i+1, j)) and not self.include((i, j+1)):
                new_numbers[i][j], new_numbers[i+1][j] = new_numbers[i+1][j], new_numbers[i][j]
                corner = (i+1, j)
            elif self.include((i, j+1)) and not self.include((i+1, j)):
                new_numbers[i][j], new_numbers[i][j+1] = new_numbers[i][j+1], new_numbers[i][j]
                corner = (i, j+1)
            elif not self.include((i+1, j)) and not self.include((i, j+1)):
                break
            else:
                lower = new_numbers[i+1][j]
                right = new_numbers[i][j+1]
                if lower <= right:
                    new_numbers[i][j], new_numbers[i+1][j] = new_numbers[i+1][j], new_numbers[i][j]
                    corner = (i+1, j)
                else:
                    new_numbers[i][j], new_numbers[i][j+1] = new_numbers[i][j+1], new_numbers[i][j]
                    corner = (i, j+1)

        new_numbers[i].pop() 
        if len(new_numbers[i]) == 0:
            new_numbers.pop(i)
        
        return SkewTableaux(new_numbers)
    
    def rectify(self) -> YoungTableaux:
        '''
        SkewTableauxを整化する
        '''
        skew_tableaux = SkewTableaux(self.numbers)
        while skew_tableaux.diagram2:
            corner = next(skew_tableaux.diagram2.enumerate_corners())
            skew_tableaux = skew_tableaux.slide(corner)
        return YoungTableaux(skew_tableaux.numbers)
    

class YoungTableauxRing:
    '''
    タブロー環の元を表すクラス
    '''
    def __init__(self, tableau:YoungTableaux | None = None):
        self.coef_dict: dict[YoungTableaux, int | float] = {}
        if isinstance(tableau, YoungTableaux):
            self._add(tableau)
        elif tableau is None:
            pass
        else:
            raise TypeError('Invalid type')
            
    def get_coefficient(self, tableau):
        if tableau in self.coef_dict:
            return tableau
        else:
            return 0
        
    def _add(self, tableau, coef=1):
        if not isinstance(tableau, YoungTableaux):
            raise TypeError('Invalid type')
        
        if tableau in self.coef_dict:
            self.coef_dict[tableau] += coef
        else:
            self.coef_dict[tableau] = coef
        
    def __str__(self):
        if len(self.coef_dict) == 0:
            return '0'
        
        result = ''
        for tableau, coef in self.coef_dict.items():
            if coef == 1:
                result += f' + {repr(tableau)}'
            elif coef == -1:
                result += f' - {repr(tableau)}'
            elif coef > 0:
                result += f' + {coef}*{repr(tableau)}'
            else:
                result += f' - {-coef}*{repr(tableau)}'
        
        return result[3:-1]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.coef_dict})'
    

    def __add__(self, other):
        if isinstance(other, YoungTableaux):
            other = YoungTableauxRing(other)
            return self + other
        
        elif isinstance(other, YoungTableauxRing):
            result = YoungTableauxRing()
            result.coef_dict = self.coef_dict.copy()
            for tableau, coef in other.coef_dict.items():
                result._add(tableau, coef)

            return result

        else:
            raise TypeError('Invalid type')
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + (-1) * other

    def __mul__(self, other):
        if isinstance(other, YoungTableaux):
            other = YoungTableauxRing(other)
            return self * other

        elif isinstance(other, YoungTableauxRing):
            result = YoungTableauxRing()
            for tableau1, coef1 in self.coef_dict.items():
                for tableau2, coef2 in other.coef_dict.items():
                    result._add(tableau1 * tableau2, coef1 * coef2)

            return result
        
        elif isinstance(other, int) or isinstance(other, float):
            result = YoungTableauxRing()
            for tableau, coef in self.coef_dict.items():
                result._add(tableau, coef * other)
            return result
        
        else:
            raise TypeError('Invalid type')
        
    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError('Invalid type')
        elif n < 0:
            raise ValueError('Invalid value')
        elif n == 0:
            return YoungTableauxRing(YoungTableaux([]))
        
        #繰り返し二乗法
        result = YoungTableauxRing(YoungTableaux([]))
        current = self
        while True:
            if n & 1:
                result = result * current
            if n <= 1:
                break
            current = current * current
            n >>= 1

        return result
    
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__mul__(other)
        elif isinstance(other, YoungTableaux):
            return YoungTableauxRing(other) * self
        else:
            raise TypeError('Invalid type')
    
    def __eq__(self, other):
        return self.coef_dict == other.coef_dict
    
    def __hash__(self):
        return hash(tuple(self.coef_dict.items()))
    
    def latex(self):
        result = ''
        for tableau, coef in self.coef_dict.items():
            if coef == 1:
                result += f' + {tableau.latex()}'
            elif coef == -1:
                result += f' - {tableau.latex()}'
            elif coef > 0:
                result += f' + {coef}{tableau.latex()}'
            else:
                result += f' - {-coef}{tableau.latex()}'
        
        return result[3:]
    

class SymmetricYoungTableauxRing:
    '''
    mを固定し、分割λに対してS_λ[m]と書かれる元の一次結合全体はタブロー環の部分環をなす。その元を表すクラス
    '''
    def __init__(self, max_value):
        self.max_value = max_value
        self.coef_dict: dict[YoungDiagram, int | float] = {}

    def expand(self):
        '''
        各項を展開して、YoungTableauxRingを返す
        '''
        result = YoungTableauxRing()
        for diagram, coef in self.coef_dict.items():
            for tableau in diagram.enumerate_tableaux(self.max_value):
                result._add(tableau, coef)
        return result
    
    def from_young_diagram(self, diagram, coef=1):
        '''
        YoungDiagramからYoungTableauxRingを生成する
        '''
        if not isinstance(diagram, YoungDiagram):
            raise TypeError('Invalid type')
        
        if diagram in self.coef_dict:
            self.coef_dict[diagram] += coef

    def __add__(self, other):
        '''
        和を求める
        '''
        if isinstance(other, SymmetricYoungTableauxRing):
            if self.max_value != other.max_value:
                raise ValueError('max_value must be same')
        
            result = SymmetricYoungTableauxRing(self.max_value)
            result.coef_dict = self.coef_dict.copy()
            for diagram, coef in other.coef_dict.items():
                if diagram in result.coef_dict:
                    result.coef_dict[diagram] += coef
                else:
                    result.coef_dict[diagram] = coef

            return result
        
        elif isinstance(other, YoungTableauxRing):
            return self.expand() + other
            
        else:
            raise TypeError('Invalid type')
        
    def __radd__(self, other):
        return self.__add__(other)

    
        
    
def robinson_schensted_knuth(matrix: list[list[int]]) -> tuple[YoungTableaux, YoungTableaux]:
    '''
    Robinson-Schensted-Knuth correspondenceを求める (行列と玉の方法, 再帰)
    '''
    # ゼロ行列の場合、空のYoung tableauxを返す
    if not any(value for row in matrix for value in row):
        return YoungTableaux([]), YoungTableaux([])
    
    # 行列のサイズ
    height = len(matrix)
    width = len(matrix[0])

    # 玉入り行列
    balls = [[[] for _ in range(width)] for _ in range(height)]
    # 玉入り行列の各玉の位置
    ball_places: dict[int, list[tuple[int,int]]] = {}
    for i in range(height):
        for j in range(width):
            min_value = 1
            if i > 0 and balls[i-1][j]:
                min_value = max(min_value, balls[i-1][j][-1]+1)
            if j > 0 and balls[i][j-1]:
                min_value = max(min_value, balls[i][j-1][-1]+1)
            for k in range(min_value, min_value+matrix[i][j]):
                balls[i][j].append(k)
                if k not in ball_places:
                    ball_places[k] = []
                ball_places[k].append((i,j))

    # P,Qの1行目を求める
    tableau_P_top = []
    tableau_Q_top = []
    max_ball_num = max(ball_places)
    for k in range(1, max_ball_num+1):
        ball_i = [p[0] for p in ball_places[k]]
        ball_j = [p[1] for p in ball_places[k]]
        tableau_P_top.append(min(ball_j)+1)
        tableau_Q_top.append(min(ball_i)+1)

    # 新たな行列を作成
    new_matrix = [[0]*width for _ in range(height)]
    for k in range(1, max_ball_num+1):
        for index in range(len(ball_places[k])-1):
            _, j = ball_places[k][index]
            i, _ = ball_places[k][index+1]
            new_matrix[i][j] += 1

    # 再帰的に残りの部分を求める
    tableau_P_bottom, tableau_Q_bottom = robinson_schensted_knuth(new_matrix)
    tableau_P = YoungTableaux([tableau_P_top] + tableau_P_bottom.numbers)
    tableau_Q = YoungTableaux([tableau_Q_top] + tableau_Q_bottom.numbers)

    return tableau_P, tableau_Q

def robinson_schensted_knuth_inverse(tableau_P: YoungTableaux, tableau_Q: YoungTableaux) -> list[list[int]]:
    '''
    Robinson-Schensted-Knuth correspondenceの逆写像
    '''
    pass

def robinson_schensted(word: list[int]) -> tuple[YoungTableaux, YoungTableaux]:
    '''
    Robinson-Schensted correspondenceを求める
    '''
    tableau_P = YoungTableaux([])
    tableau_Q_numbers = []
    for i, value in enumerate(word):
        tableau_P, index = tableau_P.row_bump(value)
        if len(tableau_Q_numbers) <= index:
            tableau_Q_numbers.append([])
        tableau_Q_numbers[index].append(i+1)

    tableau_Q = YoungTableaux(tableau_Q_numbers)
    return tableau_P, tableau_Q

def robinson_schensted_inverse(tableaux_P: YoungTableaux, tableaux_Q: YoungTableaux) -> list[int]:
    '''
    Robinson-Schensted correspondenceの逆写像
    '''
    if not tableaux_Q.is_standard():
        raise ValueError('Tableaux Q must be standard')
    
    if tableaux_P.division != tableaux_Q.division:
        raise ValueError('Invalid tableaux')
    
    indexes = {}
    for i in range(len(tableaux_Q.numbers)):
        for j in range(len(tableaux_Q.numbers[i])):
            indexes[tableaux_Q[i][j]] = (i, j)

    result = []
    for k in range(tableaux_Q.n, 0, -1):
        i, j = indexes[k]
        tableaux_P, bumped = tableaux_P.reverse_row_bump(i)
        result.append(bumped)

    result.reverse()
    return result


def littlewood_richardson_number(*args) -> int:
    '''
    Littlewood-Richardson numberを求める
    '''
    pass

def kostka_number(diagram: list[int] | YoungDiagram, weight: list[int] | YoungDiagram) -> int:
    '''
    Kostka numberを求める
    '''
    if isinstance(diagram, list):
        diagram = YoungDiagram(diagram)
    
    if isinstance(weight, YoungDiagram):
        weight = weight.division

    return len(list(diagram.enumerate_tableaux_with(weight)))


if __name__ == '__main__':
    P = YoungTableaux([
        [1,2,3],
        [4,5],
        [6]
    ])

    print(P.transpose())