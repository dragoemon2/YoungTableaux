import itertools

class FiniteSequence:
    '''
    有限数列を表すクラス
    listやtupleと同様の操作が可能
    ただし、末尾の0は削除される
    '''
    def __init__(self, sequence: list[int] = None):
        if sequence is None:
            sequence = []
        if isinstance(sequence, tuple):
            sequence = list(sequence)
        elif isinstance(sequence, FiniteSequence):
            sequence = sequence.sequence
        elif isinstance(sequence, (int, float)):
            sequence = [sequence]
        else:
            sequence = list(sequence)
        
        self.sequence = sequence

        # 末尾の0を削除
        while self.sequence and not self.sequence[-1]:
            self.sequence.pop()
        
    def __getitem__(self, index):
        return self.sequence[index]
    
    def __eq__(self, other):
        return self.sequence == other.sequence

    def __hash__(self):
        return hash(tuple(self.sequence))
    
    def __len__(self):
        return len(self.sequence)
    
    def __str__(self):
        return str(self.sequence)
    
    def __repr__(self):
        return str(self.sequence)

    def __add__(self, other):
        new_sequence = [i + j for i, j in itertools.zip_longest(self.sequence, other.sequence, fillvalue=0)]
        return self.__class__(new_sequence)
    
    def __sub__(self, other):
        new_sequence = [i - j for i, j in itertools.zip_longest(self.sequence, other.sequence, fillvalue=0)]
        return self.__class__(new_sequence)
    
    def __mul__(self, other):
        if isinstance(other, int | float):
            new_sequence = [i * other for i in self.sequence]
            return self.__class__(new_sequence)
        else:
            raise TypeError('Invalid type')
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        return self.__class__([-i for i in self.sequence])
    
    def __iter__(self):
        return iter(self.sequence)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.sequence})'
    
    def __bool__(self):
        return bool(self.sequence)
    
    def __contains__(self, other):
        return other in self.sequence
    
    def __lt__(self, other):
        return self.sequence < other.sequence
    
    def __le__(self, other):
        return self.sequence <= other.sequence
    
    def __gt__(self, other):
        return self.sequence > other.sequence
    
    def __ge__(self, other):
        return self.sequence >= other.sequence

    @classmethod
    def identity(cls, n):
        return cls([0] * n + [1])