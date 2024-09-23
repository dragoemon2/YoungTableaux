import copy

from young_tableaux.young_diagram import YoungDiagram, YoungTableaux


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
    
