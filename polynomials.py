import numpy as np
from functools import lru_cache
from finite_sequence import FiniteSequence as seq
import itertools

class Polynomial:
    '''
    多変数多項式を表すクラス
    '''
    @classmethod
    def var(cls, n: int):
        '''
        多項式の変数を作成する
        '''
        return Polynomial({seq.identity(n) : 1})

    def __init__(self, coef_dict: dict[seq, int | float] = None):
        if coef_dict is None:
            coef_dict = {}
        coef_dict = {seq(key) : value for key, value in coef_dict.items() if value != 0}
        self.coef_dict : dict[seq, int | float] = coef_dict

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Polynomial({seq() : other})
        
        result = Polynomial(self.coef_dict.copy())
        for key, value in other.coef_dict.items():
            if key in result.coef_dict:
                result.coef_dict[key] += value
            else:
                result.coef_dict[key] = value
        result._simplify()
        return result
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Polynomial({seq() : other})
        
        result = Polynomial(self.coef_dict.copy())
        for key, value in other.coef_dict.items():
            if key in result.coef_dict:
                result.coef_dict[key] -= value
            else:
                result.coef_dict[key] = -value
        result._simplify()
        return result
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial({key : value * other for key, value in self.coef_dict.items()})
        
        result = Polynomial()
        for key1, value1 in self.coef_dict.items():
            for key2, value2 in other.coef_dict.items():
                # 次数
                new_key = seq(i + j for i, j in itertools.zip_longest(key1, key2, fillvalue=0))
                new_value = value1 * value2
                if new_key in result.coef_dict:
                    result.coef_dict[new_key] += new_value
                else:
                    result.coef_dict[new_key] = new_value
        result._simplify()
        return result
    
    def __neg__(self):
        return Polynomial({key : -value for key, value in self.coef_dict.items()})
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return -self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, n: int):
        if n == 0:
            return Polynomial({seq() : 1})
        if n == 1:
            return self
        if n % 2 == 0:
            tmp = self ** (n // 2)
            return tmp * tmp
        else:
            return self * (self ** (n - 1))
    
    def divide(self, other) -> tuple:
        '''
        多項式の割り算を行い、商と余りを返す
        あまりは、「辞書式順序で」元より小さいようにする

        Parameters
        ----------
        other : Polynomial
            割る多項式

        Returns
        -------
        Polynomial
            商

        Polynomial
            余り
        '''

        # 次数を取得し、長さを合わせる
        self_max_degree = self.lexicographic_degree()
        other_max_degree = other.lexicographic_degree()
        self_coef = self.coef_dict[self_max_degree]
        other_coef = other.coef_dict[other_max_degree]

        # すでにotherがselfより大きい場合は商は0
        if self_max_degree < other_max_degree:
            return Polynomial(), self

        remainder_max_degree = seq(i - j for i, j in itertools.zip_longest(self_max_degree, other_max_degree, fillvalue=0))
        while len(remainder_max_degree) and remainder_max_degree[-1] == 0:
            remainder_max_degree = remainder_max_degree[:-1]

        # 係数の計算
        if self_coef % other_coef == 0:
            coef = self_coef // other_coef
        else:
            coef = self_coef / other_coef
        
        # 再帰的に計算
        dividend = self - Polynomial({remainder_max_degree : 1}) * other * coef
        quotient, remainder = dividend.divide(other)
        quotient += Polynomial({remainder_max_degree : coef})
        return quotient, remainder
    
    def __floordiv__(self, other):
        return self.divide(other)[0]
        
    def __mod__(self, other):
        return self.divide(other)[1]
    
    def _simplify(self):
        for key in self.coef_dict.copy():
            # 係数が0の項を削除
            if self.coef_dict[key] == 0:
                del self.coef_dict[key]
    
    @lru_cache()
    def total_degree(self):
        return max(sum(key) for key in self.coef_dict.keys())
    
    @lru_cache()
    def lexicographic_degree(self):
        return max(key for key in self.coef_dict.keys())
    
    @lru_cache()
    def number_of_variables(self):
        return len(self.coef_dict)
    
    def __str__(self):
        if not self.coef_dict:
            return '0'
        return ' + '.join(f'{value} * {" * ".join(f"x{i}^{j}" for i, j in enumerate(key) if j != 0)}' for key, value in self.coef_dict.items())
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.coef_dict})'
    
    def assign_value(self, value: dict[int, int | float]):
        '''
        変数に値を代入する
        '''  
        result = 0
        for key, coef in self.coef_dict.items():
            for i, j in enumerate(key):
                coef *= value.get(i, 0) ** j
            result += coef

        return result
    
    def __eq__(self, other):
        return self.coef_dict == other.coef_dict
    
    def __hash__(self):
        return hash(tuple(self.coef_dict.items()))

    
if __name__ == '__main__': 
    x0 = Polynomial.var(0)
    x1 = Polynomial.var(1)
    