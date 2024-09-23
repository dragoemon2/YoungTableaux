import copy

from young_tableaux.functions import littlewood_richardson_number
from young_tableaux.skew_diagram import SkewDiagram
from young_tableaux.young_diagram import YoungDiagram, YoungTableaux


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
        if isinstance(other, YoungTableauxRing):
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
        if len(self.coef_dict) == 0:
            return '0'
        
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
    


def _multiply_monomial(diagram1: YoungDiagram, diagram2: YoungDiagram, max_value : int|None):
    '''
    2つのヤング図形に対するS_λの積を求める
    '''
    if not isinstance(diagram1, YoungDiagram) or not isinstance(diagram2, YoungDiagram):
        raise TypeError('Invalid type')

    
    result = SymmetricYoungTableauxRing(max_value=max_value)
    
    n = diagram1.n + diagram2.n  # 箱の数
    for diagram in YoungDiagram.enumerate_diagrams(n):
        coef = littlewood_richardson_number(diagram1, diagram2, diagram)
        if coef != 0:
            result += SymmetricYoungTableauxRing(diagram, max_value=max_value) * coef
            
    return result

def _expand_skew_diagram(skew_diagram: SkewDiagram):
    '''
    Skew diagramに対するS_λ/μの展開を求める
    '''
    coef_dict = {}

    for diagram in YoungDiagram.enumerate_diagrams(skew_diagram.n):
        coef = littlewood_richardson_number(skew_diagram, diagram)
        if coef != 0:
            coef_dict[diagram] = coef

        return coef_dict
        
class SymmetricYoungTableauxRing:
    '''
    mを固定し、分割λに対してS_λ[m]と書かれる元の一次結合全体はタブロー環の部分環をなす。その元を表すクラス
    '''
    def __init__(self, diagram = None, max_value: int | None = None):
        self.max_value = max_value
        self.coef_dict: dict[YoungDiagram, int | float] = {}

        if isinstance(diagram, YoungDiagram):
            self.coef_dict[diagram] = 1
        elif isinstance(diagram, SkewDiagram):
            self.coef_dict = _expand_skew_diagram(diagram)
        elif isinstance(diagram, SymmetricYoungTableauxRing):
            self.coef_dict = copy.deepcopy(diagram.coef_dict)
        elif diagram is None:
            pass
        else:
            raise TypeError('Invalid type')

        self._erase_zeros()
    
    def expand(self, max_value=None):
        '''
        各項を展開して、YoungTableauxRingを返す
        '''
        if max_value is None:
            max_value = self.max_value
        if max_value is None:
            raise ValueError('max_value must be specified')
    
        result = YoungTableauxRing()
        for diagram, coef in self.coef_dict.items():
            for tableau in diagram.enumerate_tableaux(max_value):
                result._add(tableau, coef)
        return result

    def __add__(self, other):
        '''
        和を求める
        '''
        if isinstance(other, SymmetricYoungTableauxRing):
            if self.max_value != other.max_value:
                raise ValueError('max_value must be same')
        
            result = SymmetricYoungTableauxRing(max_value=self.max_value)
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
    
    

    def __mul__(self, other):
        '''
        積を求める
        '''
        if isinstance(other, SymmetricYoungTableauxRing):
            if self.max_value != other.max_value:
                raise ValueError('max_value must be same')
        
            result = SymmetricYoungTableauxRing(max_value=self.max_value)
            for diagram1, coef1 in self.coef_dict.items():
                for diagram2, coef2 in other.coef_dict.items():
                    result += _multiply_monomial(diagram1, diagram2, self.max_value) * coef1 * coef2

            return result
            
        elif isinstance(other, YoungTableauxRing):
            return self.expand() * other
        
        elif isinstance(other, int) or isinstance(other, float):
            if other == 0:
                return SymmetricYoungTableauxRing(max_value=self.max_value)
            result = SymmetricYoungTableauxRing(max_value=self.max_value)
            result.coef_dict = {diagram: coef * other for diagram, coef in self.coef_dict.items()}
            return result
            
        else:
            raise TypeError('Invalid type')
        
    def __rmul__(self, other):
        return self.__mul__(other) # 可換
    

    def __str__(self):
        if len(self.coef_dict) == 0:
            return '0'
        
        result = ''
        for diagram, coef in self.coef_dict.items():
            if self.max_value is not None:
                monomial = f'S_{tuple(diagram.division)}[{self.max_value}]'
            else:
                monomial = f'S_{tuple(diagram.division)}'
            
            if coef == 1:
                result += f' + {monomial}'
            elif coef == -1:
                result += f' - {monomial}'
            elif coef > 0:
                result += f' + {coef}*{monomial}'
            else:
                result += f' - {-coef}*{monomial}'
        
        return result[3:]
    
    def __repr__(self):
        if len(self.coef_dict) == 0:
            return f'{self.__class__.__name__}()'
        
        result = ''
        for diagram, coef in self.coef_dict.items():
            if self.max_value is not None:
                monomial = f'{self.__class__.__name__}({repr(diagram)}, max_value={self.max_value})'
            else:
                monomial = f'{self.__class__.__name__}({repr(diagram)})'
            
            if coef == 1:
                result += f' + {monomial}'
            elif coef == -1:
                result += f' - {monomial}'
            elif coef > 0:
                result += f' + {coef}*{monomial}'
            else:
                result += f' - {-coef}*{monomial}'
        
        return result[3:]
    
    def latex(self):
        if len(self.coef_dict) == 0:
            return '0'
        
        result = ''
        for diagram, coef in self.coef_dict.items():
            if self.max_value is not None:
                monomial = 'S_{%s}^{[%d]}' % (str(tuple(diagram.division)), self.max_value)
            else:
                monomial = 'S_{%s}' % str(tuple(diagram.division))
            
            if coef == 1:
                result += f' + {monomial}'
            elif coef == -1:
                result += f' - {monomial}'
            elif coef > 0:
                result += f' + {coef}{monomial}'
            else:
                result += f' - {-coef}{monomial}'
        
        return result[3:]

    def _erase_zeros(self):
        '''
        簡約化
        '''
        for diagram, value in self.coef_dict.copy().items():
            if value == 0:
                del self.coef_dict[diagram]

            if self.max_value is not None and len(diagram.division) > self.max_value:
                del self.coef_dict[diagram]
