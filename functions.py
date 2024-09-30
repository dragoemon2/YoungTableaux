from young_tableaux.skew_diagram import SkewDiagram
from young_tableaux.young_diagram import (StandardYoungTableaux, YoungDiagram,
                                          YoungTableaux)


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
    if tableau_P.division != tableau_Q.division:
        raise ValueError('Tableaux P and Q must have same shape')
    
    width = tableau_P.max_value()
    height = tableau_Q.max_value()


    matrix = [[0]*width for _ in range(height)]
    min_values = [[0]*width for _ in range(height)]

    for line_P, line_Q in zip(reversed(tableau_P.numbers), reversed(tableau_Q.numbers)):
        number_places = {k+1:(i-1,j-1) for k,(i,j) in enumerate(zip(line_Q, line_P))}

        # 玉に数字を割り当てる
        ball_places: dict[int, list[tuple[int, int]]] = {}
        min_values = [[0]*width for _ in range(height)]
        for i in range(height-1,-1,-1):
            for j in range(width-1,-1,-1):
                if not matrix[i][j]:
                    min_value_candidate = []
                    if i < height-1:
                        min_value_candidate.append(min_values[i+1][j])
                    if j < width-1:
                        min_value_candidate.append(min_values[i][j+1])
                    min_values[i][j] = min(min_value_candidate, default=float('inf'))
                    continue

                k = max([k for k,(ni,nj) in number_places.items() if ni < i and nj < j])
                if i < height-1:
                    if min_values[i+1][j] <= k:
                        k = min_values[i+1][j] - 1
                if j < width-1:
                    if min_values[i][j+1] <= k:
                        k = min_values[i][j+1] - 1
                
                for l in range(k,k-matrix[i][j],-1):
                    if l not in ball_places:
                        ball_places[l] = []
                    ball_places[l].append((i,j))
                min_values[i][j] = k - matrix[i][j] + 1

        for ball_places_of_index in ball_places.values():
            ball_places_of_index.sort()

        # 行列を更新
        new_matrix = [[0]*width for _ in range(height)] # 新しい行列を作成
        for index, (i, j) in enumerate(zip(line_Q, line_P)):
            index, i, j = index+1, i-1, j-1
            ball_places_of_index = ball_places.get(index, [])
            
            ball_i = [i] + [p[0] for p in ball_places_of_index]
            ball_j = [p[1] for p in ball_places_of_index] + [j]
            for new_i, new_j in zip(ball_i, ball_j):
                new_matrix[new_i][new_j] += 1
        matrix = new_matrix

    return matrix
        

def robinson_schensted(word: list[int]) -> tuple[YoungTableaux, StandardYoungTableaux]:
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

    tableau_Q = StandardYoungTableaux(tableau_Q_numbers)
    return tableau_P, tableau_Q

def robinson_schensted_inverse(tableaux_P: YoungTableaux, tableaux_Q: YoungTableaux) -> list[int]:
    '''
    Robinson-Schensted correspondenceの逆写像
    '''
    if not tableaux_Q.is_standard():
        raise ValueError('Tableaux Q must be standard')
    
    if tableaux_P.division != tableaux_Q.division:
        raise ValueError('Tableaux P and Q must have same shape')
    
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

def _littlewood_richardson_number__recursive(constraints: dict[tuple[int, str], int], young_diagram: YoungDiagram, current_numbering: YoungTableaux, place_of_numbers: list[tuple[int, int]]):
    '''
    littlewood_richardson_number()内部で用いる再帰関数
    '''
    if young_diagram.n == current_numbering.n:
        yield current_numbering
    
    next_value = current_numbering.n + 1 # 次に入れる数字
    for (new_i, new_j) in current_numbering.enumerate_outer_boxes():
        if (new_i, new_j) not in young_diagram:
            continue

        # 制約条件を満たすかどうか
        if (next_value, 'nE') in constraints: # north-East制約:
            other_i, other_j = place_of_numbers[constraints[(next_value, 'nE')]-1]
            if not (new_i <= other_i and new_j > other_j):
                continue
        if (next_value, 'Sw') in constraints:
            other_i, other_j = place_of_numbers[constraints[(next_value, 'Sw')]-1]
            if not (new_i > other_i and new_j <= other_j):
                continue

        new_numbering = current_numbering.add_box(new_i, next_value)
        new_place_of_numbers = place_of_numbers + [(new_i, new_j)]

        yield from _littlewood_richardson_number__recursive(constraints, young_diagram, new_numbering, new_place_of_numbers)


def littlewood_richardson_number(*args) -> int:
    '''
    Littlewood-Richardson numberを求める [Remmel, Whitney (1984)]
    '''
    if len(args) == 2 and isinstance(args[0], SkewDiagram) and isinstance(args[1], YoungDiagram):
        skew_diagram, weight = args
    elif len(args) == 2 and isinstance(args[0], SkewDiagram) and isinstance(args[1], list):
        skew_diagram, weight = args
        weight = YoungDiagram(weight)
    elif len(args) == 3 and isinstance(args[0], YoungDiagram) and isinstance(args[1], YoungDiagram) and isinstance(args[2], YoungDiagram):
        if args[0] not in args[2]:
            return 0
        skew_diagram = SkewDiagram(args[2], args[0])
        weight = args[1]
    else:
        raise TypeError('Invalid type')

    # 型アノテーション
    skew_diagram : SkewDiagram
    weight : YoungDiagram

    if skew_diagram.n != weight.n:
        return 0
    
    # 逆数字付けを求める
    reverse_numbering = []
    for i in range(len(skew_diagram.diagram1.division)):
        stop = (0 if len(skew_diagram.diagram2.division) <= i else skew_diagram.diagram2.division[i]) - 1
        start = skew_diagram.diagram1.division[i] - 1
        for j in range(start, stop, -1):
            reverse_numbering.append((i, j))

    # 制約条件を求める
    consraints = {}
    for k in range(1, skew_diagram.n+1):
        i, j = reverse_numbering[k-1]
        if k > 1 and i == reverse_numbering[k-2][0]:
            consraints[(k, 'nE')] = k-1 # kはk-1のnorth-Eastにある
        if (i-1, j) in reverse_numbering:
            consraints[(k, 'Sw')] = reverse_numbering.index((i-1, j)) + 1 # kはkの上の数字のSouth-westにある


    tableaux = list(_littlewood_richardson_number__recursive(consraints, weight, YoungTableaux([]), [])) # 逆数字付けを再帰的に求める
    return len(tableaux) 
    

def kostka_number(diagram: list[int] | YoungDiagram, weight: list[int] | YoungDiagram) -> int:
    '''
    Kostka numberを求める
    '''
    if isinstance(diagram, list):
        diagram = YoungDiagram(diagram)
    
    if isinstance(weight, YoungDiagram):
        weight = weight.division

    return len(list(diagram.enumerate_tableaux_with(weight)))
