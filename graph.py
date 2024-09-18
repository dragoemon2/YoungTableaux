import itertools
import os

import graphviz

import young_tableaux as yt


def enumerate_knuth(word: list[int]):
    '''
    wordに対してKnuth同値なwordを列挙する
    '''
    tableau_P = yt.YoungTableaux.from_word(word)
    for tableau_Q in tableau_P.enumerate_standard_tableaux():
        yield yt.robinson_schensted_inverse(tableau_P, tableau_Q)

def enumerate_k1(word: list[int]):
    '''
    wordに対してK'変換を1度行って得られるwordを列挙する
    '''
    words: list[list[int]] = []
    for i in range(len(word)-2):
        copied = word.copy()
        if word[i+2] < word[i] <= word[i+1]:
            copied[i+1], copied[i+2] = copied[i+2], copied[i+1]
            if copied not in words:
                words.append(copied)

    return words

def enumerate_k2(word: list[int]):
    '''
    wordに対してK''変換を1度行って得られるwordを列挙する
    '''
    words: list[list[int]] = []
    for i in range(2, len(word)):
        copied = word.copy()
        if word[i-2] <= word[i] < word[i-1]:
            copied[i-1], copied[i-2] = copied[i-2], copied[i-1]
            if copied not in words:
                words.append(copied)

    return words


class WordGraph(graphviz.Digraph):
    def __init__(self):
        super().__init__(format='png', engine='dot')
        self.words = []
        if os.path.exists('tmp'):
            for file in os.listdir('tmp'):
                os.remove(f'tmp/{file}')
            os.rmdir('tmp')
        os.mkdir('tmp')

    def _add_edges(self, words, subgraph=None):
        '''
        wordsたちをエッジで結ぶ。wordsはいくつかの連結成分の合併になっているとする
        '''
        for word in words:
            for k1_neighbor in enumerate_k1(word):
                assert k1_neighbor in words
                if subgraph is not None:
                    subgraph.edge(''.join(map(str, word)), ''.join(map(str, k1_neighbor)), label="K'", color='red')
                else:
                    self.edge(''.join(map(str, word)), ''.join(map(str, k1_neighbor)), label="K'", color='red')

            for k2_neighbor in enumerate_k2(word):
                assert k2_neighbor in words
                if subgraph is not None:
                    subgraph.edge(''.join(map(str, word)), ''.join(map(str, k2_neighbor)), label="K''", color='blue')
                else:
                    self.edge(''.join(map(str, word)), ''.join(map(str, k2_neighbor)), label="K''", color='blue')


    def add_knuth_same_with(self, original_word: list[int], subgraph=None, colored_words=None):
        '''
        wordに対してKnuth同値なwordを列挙し、エッジを追加する
        '''
        if original_word in self.words:
            return
        
        if colored_words is None:
            colored_words = {}

        words = []
        for word in enumerate_knuth(original_word):
            color = colored_words.get(tuple(word), None)
            if subgraph is not None:
                subgraph.node(''.join(map(str, word)), color=color)
            else:
                self.node(''.join(map(str, word)), color=color)
            words.append(word)

        self._add_edges(words, subgraph)
        self.words += words

    def add_all_permutation(self, n, graph=None):
        '''
        nの置換を全て追加する
        '''
        words = list(map(list,itertools.permutations(range(1, n+1))))
        # for word in words:
        #     if word in self.words:
        #         continue
        #     self.node(''.join(map(str, word)))
        
        # self._add_edges(words)
        # self.words += words

        #一時ディレクトリ
        # os.mkdir('tmp')

        if graph is None:
            graph = self
        

        tableaux = set()
        for word in words:
            tableau = yt.YoungTableaux.from_word(word)
            if tableau in tableaux:
                continue
            tableaux.add(tableau)

        tableaux = list(tableaux)
        tableaux.sort(key=lambda x: x.division)
        
        # for word in words:
        #     if word in self.words:
        #         continue
        for tableau in tableaux:
            tableau : yt.YoungTableaux
            row_word = tableau.to_word(direction='row')
            column_word = tableau.to_word(direction='column')
            image_path = f'tmp/{hash(tableau)}.png'
            tableau.render(image_path, 30)
            subgraph = graphviz.Digraph(name=f'cluster_{hash(tableau)}')
            color_dict = {tuple(row_word): 'red', tuple(column_word): 'blue'} if row_word != column_word else {tuple(row_word): 'green'}
            self.add_knuth_same_with(row_word, subgraph, color_dict)
            
            
            subgraph.attr(label=(
                '<'
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                f'<TR><TD><IMG SRC="{image_path}"/></TD></TR>'
                '</TABLE>'
                '>'
            ))
            # subgraph.attr(label=(
            #     '<'
            #     '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            #     '<TR>'
            #     f'<TD><IMG SRC="{image_path}"/></TD>'
            #     # '<TD>Subgraph 1</TD>'
            #     '</TR>'
            #     '</TABLE>'
            #     '>'
            # ))
            graph.subgraph(subgraph)


        # for word in words:
        #     if word in self.words:
        #         continue
        #     tableau = yt.YoungTableaux.from_word(word)
        #     with tempfile.NamedTemporaryFile(mode='w', suffix='.png') as f:
        #         tableau.render(f.name)
        #     subgraph = graphviz.Digraph(name=f'cluster_{hash(yt.YoungTableaux.from_word(word))}')
        #     self.add_knuth_same_with(word, subgraph)
        #     self.subgraph(subgraph)
        
    

if __name__ == '__main__':
    #yt.YoungTableaux.from_word([4,1,2,3,5]).render('tableau.png')
    g = WordGraph()
    #g.add_knuth_same_with([4, 1, 2, 5, 3])
    # for i in range(1, 6):
    #     subgraph = graphviz.Digraph(name=f'cluster_{i}')
    #     g.add_all_permutation(i, subgraph)
    #     g.subgraph(subgraph)


    # subgraph = graphviz.Digraph(name='cluster_4')
    # g.add_all_permutation(4, subgraph)
    # g.subgraph(subgraph)
    N = 6
    g.add_all_permutation(N)
    g.render(f'./graph_{N}', view=True)
            