from young_tableaux.young_diagram import YoungDiagram, YoungTableaux


def _enumerate_littlewood_richardson_tableaux__recursive(box_places: list[tuple[int,int]], weight: list[int], current_word: list[int], current_weight: list[int], numbers: list[list[int]]):
    index = len(current_word)
    if index == sum(weight):
        yield SkewTableaux(numbers)
        return
    
    current_i, current_j = box_places[index]
    
    for k in range(len(weight)):
        if k > 0 and current_weight[k-1] <= current_weight[k]:
            continue # kを追加すると逆格子じゃなくなる

        if current_weight[k] >= weight[k]:
            continue # kを追加すると重みを超える

        if current_j+1 < len(numbers[current_i]) and k+1 > numbers[current_i][current_j+1]:
            continue # kを追加すると右に広義単調増加じゃなくなる

        if current_i > 0 and numbers[current_i-1][current_j] >= k+1:
            continue # kを追加すると下に狭義単調増加じゃなくなる

        # 次のステップのための情報を作成
        new_numbers = [line.copy() for line in numbers]
        new_numbers[current_i][current_j] = k+1
        new_word = current_word + [k+1]
        new_weight = current_weight.copy()
        new_weight[k] += 1

        yield from _enumerate_littlewood_richardson_tableaux__recursive(box_places, weight, new_word, new_weight, new_numbers) # 再帰

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
        return f'{self.__class__.__name__}({self.diagram1}, {self.diagram2})'

    def to_text(self):
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
        与えられた重みに対するLittlewood-Richardson skew tableauxを列挙する
        '''
        if isinstance(weight, YoungDiagram):
            weight = weight.division
        
        if sum(weight) != self.n:
            return
        
        # 右上から順に箱の位置を列挙
        box_places = []
        for i in range(len(self.diagram1.division)):
            j_end = 0 if i >= len(self.diagram2.division) else self.diagram2.division[i]
            j_start = self.diagram1.division[i]
            for j in range(j_start-1, j_end-1, -1):
                box_places.append((i, j))

        yield from _enumerate_littlewood_richardson_tableaux__recursive(box_places, weight, [], [0 for _ in range(len(weight))], [[0 for _ in range(self.diagram1.division[i])] for i in range(len(self.diagram1.division))])

    def enumerate_littlewood_richardson_tableaux(self):
        '''
        Littlewood-Richardson skew tableauxを列挙する
        '''
        for weight in YoungDiagram.enumerate_diagrams(self.n):
            yield from self.enumerate_littlewood_richardson_tableaux_with(weight)

    
    def __hash__(self):
        return hash((self.diagram1, self.diagram2))
    
    
    def assign_number_with_word(self, word: list[int]):
        if len(word) != self.n:
            raise ValueError("length of word mismatch")
        word = word.copy()
        
        numbers = []
        for i in range(len(self.diagram1.division), -1, -1):
            line = []
            for j in range(self.diagram2.division[i], self.diagram1.division[i]):
                line.append(word.pop(0))
            numbers.append([0] * (self.diagram2.division[i]) + line)

        return NumberedSkewDiagram(numbers)
    
    @staticmethod
    def from_up_down_sequence(sequence: str):
        '''
        上下列からSkewDiagramを生成する
        '''
        # u, dを+,-に変換
        if 'u' in sequence or 'd' in sequence:
            sequence = sequence.replace('u', '+').replace('d', '-')
        
        if not all(i == '+' or i == '-' for i in sequence):
            raise ValueError("Invalid sequence. please use '+' or '-'")

        height = sequence.count('+') + 1
        
        diagram1 = [0] * height
        diagram2 = [0] * (height-1)

        i, j = height-1, 0
        diagram1[i] = 1
        for s in sequence:
            if s == '+':
                j += 1
                diagram1[i] += 1
            else:
                i -= 1
                diagram2[i] = j
                diagram1[i] = j+1

        return SkewDiagram(YoungDiagram(diagram1), YoungDiagram(diagram2))
        
class NumberedSkewDiagram(SkewDiagram):
    def __init__(self, numbers: list[list[int]]):
        # Noneを0に変換
        numbers = [[0 if i is None else i for i in line] for line in numbers]
        diagram1 = YoungDiagram([len(line) for line in numbers])
        diagram2 = YoungDiagram([line.count(0) for line in numbers if line[0] == 0])
        super().__init__(diagram1, diagram2)
        self.numbers = numbers

    def enumerate(self):
        '''
        SkewDiagramの中の数字を列挙する
        '''
        for i in range(len(self.numbers)):
            yield i, ((j, self.numbers[i][j]) for j in range(self.diagram2.division[i], self.diagram1.division[i]))

    def __str__(self):
        if not self.numbers:
            return f'{self.__class__.__name__}([])'
        
        result = f'{self.__class__.__name__}([\n'
        for line in self.numbers:
            result += '    ' + str(line) + ',\n'
        result = result[:-2] # 末尾のカンマと改行を削除
        result += '\n])' # 末尾に改行を追加
        return result
    
    def to_text(self):
        max_len = max([len(str(i)) for line in self.numbers for i in line])
        result = ''
        for line in self.numbers:
            for i in line:
                str_i = str(i) if i != 0 else '□'
                result += str_i.rjust(max_len) + ' '
            result = result[:-1] + '\n'
        return result[:-1]
    
    def latex(self):
        text = ''
        text += r'\begin{ytableau}' + '\n'
        for line in self.numbers:
            text += '    ' + ' & '.join([str(i) if i != 0 else r'\none' for i in line]) + r' \\' + '\n'
        text += r'\end{ytableau}'
        return text

    def __repr__(self):
        return f'{self.__class__.__name__}({self.numbers})'

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
    
    def to_word(self, direction='row'):
        '''
        YoungTableauxをwordに変換する
        '''
        if direction == 'row':
            result = []
            for i, line in enumerate(self.numbers):
                result = line[self.diagram2[i]:] + result

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

    
    def slide(self, corner):
        '''
        cornerをスライドさせる
        '''
        new_numbers = [line.copy() for line in self.numbers]
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
    
    
    
