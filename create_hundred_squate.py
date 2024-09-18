import random

from young_tableaux import YoungTableaux

TEXT = r'''
\begin{figure}[H]
    \begin{align*}
        \begin{array}{|c|%s|}
            \hline
            %s
            \hline
            %s
            \hline
        \end{array}
    \end{align*}
\end{figure}
'''


MAX_NUM = 11

MIN_SIZE = 3
MAX_SIZE = 14

def create_random_tableaux(n : int):
    for _ in range(n):
        size = random.randint(MIN_SIZE, MAX_SIZE)
        word = [random.randint(1, MAX_NUM) for _ in range(size)]
        tableau = YoungTableaux.from_word(word)
        yield tableau

def create_figure(array: list[list[str]]):
    N = len(array[0]) - 1
    top = array.pop(0)
    top = ' & '.join(top)
    top = top + r' \\'

    bottom = (r' \\' + '\n').join([' & '.join(line) for line in array])
    bottom = bottom + r' \\'

    return TEXT % (N * 'c', top, bottom)


def create_hundred_square():
    N = 10
    rows : list[YoungTableaux] = list(create_random_tableaux(N))
    columns : list[YoungTableaux] = list(create_random_tableaux(N))

    problem_array = [['' for _ in range(N+1)] for _ in range(N+1)]
    answer_array = [['' for _ in range(N+1)] for _ in range(N+1)]

    for i, (row, column) in enumerate(zip(rows, columns)):
        problem_array[0][i+1] = row.latex()
        problem_array[i+1][0] = column.latex()
        answer_array[0][i+1] = row.latex()
        answer_array[i+1][0] = column.latex()

    for i in range(1, N+1):
        for j in range(1, N+1):
            answer_array[j][i] = (rows[i-1] * columns[j-1]).latex()

    problem_array_2nd = []
    answer_array_2nd = []
    for i in range(N+1):
        problem_array_2nd.append(problem_array[i])
        answer_array_2nd.append(answer_array[i])
        problem_array_2nd.append(['' for _ in range(N+1)])
        answer_array_2nd.append(['' for _ in range(N+1)])

    return create_figure(problem_array_2nd), create_figure(answer_array_2nd)



if __name__ == '__main__':
    problem, answer = create_hundred_square()
    print(problem)
    print(answer)



