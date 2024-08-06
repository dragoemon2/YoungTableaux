import copy
import math
from functools import lru_cache


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
    
    def number_of_standard_tableaux(self):
        '''
        標準Young tableauxの個数を求める
        '''
        prod = 1
        for box in self.enumerate_boxes():
            prod *= self.hook_length(box)
        
        return math.factorial(self.n) // prod
            
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
    

class NumberedYoungDiagram(YoungDiagram):
    '''
    番号付けされたヤング図形
    '''
    def __init__(self, numbers : list[list[int]]):
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
    
    def to_word(self):
        '''
        YoungTableauxをwordに変換する
        '''
        result = []
        for line in self.numbers:
            result = line + result

        return result
    
    @classmethod
    def from_word(cls, word):
        '''
        wordから、それとknuth同値なwordを持つYoungTableauxを生成する
        '''
        numbers = []
        for i in range(len(word)):
            numbers.append([0]*i + [word[i]])
        numbers.reverse()
    
        skew = SkewTableaux(numbers)
        return skew.rectify()
            
    def __mul__(self, other, method='slide'):
        '''
        2つのYoungTableauxの積を求める (*演算子)
        '''
        if not isinstance(other, YoungTableaux):
            raise TypeError('Invalid type')
        
        if method == 'bump': # bumpを使って掛け算を行う
            return self._multiply_with_bump(other)
        elif method == 'slide': # slideを使って掛け算を行う
            return self._multiply_with_slide(other)
        else:
            raise NotImplementedError('Invalid method')

    def __matmul__(self, other):
        '''
        2つのYoungTableauxを並べたSkewTableauxを求める (@演算子)
        '''
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


class SkewTableaux(SkewDiagram):
    def __init__(self, numbers: list[list[int]]):
        '''
        numbers: list[list[int]] 内タブローは0を埋める
        '''
        self.numbers = numbers
        diagram1 = YoungDiagram([len(line) for line in numbers])
        diagram2 = YoungDiagram([line.count(0) for line in numbers if line[0] == 0])
        super().__init__(diagram1, diagram2)
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




if __name__ == '__main__':
    print(YoungDiagram([10,10]).number_of_standard_tableaux())

