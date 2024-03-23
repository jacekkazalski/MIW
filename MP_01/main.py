import numpy as np
import matplotlib.pyplot as plt

transition_matrix_computer = [
    [2 / 3, 0 / 3, 1 / 3],
    [1 / 3, 2 / 3, 0 / 3],
    [0 / 3, 2 / 3, 1 / 3]
]
transition_matrix_player = [
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3]
]
states = ['Rock', 'Paper', 'Scissors']


def first_move():
    return np.random.choice(states, p=[1 / 3, 1 / 3, 1 / 3])


def next_move(prev, matrix):
    x = states.index(prev)
    return np.random.choice(states, p=matrix[x])


def who_wins(computer_move, player_move):
    if computer_move == player_move:
        return 0
    elif (computer_move == 'Rock' and player_move == 'Paper') or \
            (computer_move == 'Paper' and player_move == 'Scissors') or \
            (computer_move == 'Scissors' and player_move == 'Rock'):
        print('Player wins')
        return 1
    elif (computer_move == 'Paper' and player_move == 'Rock') or \
            (computer_move == 'Scissors' and player_move == 'Paper') or \
            (computer_move == 'Rock' and player_move == 'Scissors'):
        print('Computer wins')
        return -1


def calculate_stationary_vector():
    eigenvalues, eigenvectors = np.linalg.eig(np.transpose(transition_matrix_computer))
    stationary_index = np.argmin(np.abs(eigenvalues - 1.0))
    stationary_vector = np.real(eigenvectors[:, stationary_index])
    stationary_vector /= stationary_vector.sum()
    return stationary_vector


def learn(last_move, result, learning_rate):
    index = states.index(last_move)

    change = result * learning_rate
    for i in range(len(transition_matrix_player[index])):
        if (0 <= transition_matrix_player[index][i] - change <= 1) and \
                (0 <= transition_matrix_player[index][i] + change <= 1):

            if i != index:
                transition_matrix_player[index][i] -= change / 2
            else:
                transition_matrix_player[index][i] += change
        else:
            break
    # Normalizacja
    total_prob = sum(transition_matrix_player[index])
    transition_matrix_player[index] = [prob / total_prob for prob in transition_matrix_player[index]]


def game(rounds, strategy=1):
    cash = 0
    cash_history = [cash]
    learning_rate = 0.02
    computer_move = first_move()
    player_move = first_move()
    stationary_vector = calculate_stationary_vector()
    for x in range(rounds):
        print(f"Computer: {computer_move} Player: {player_move}")
        result = who_wins(computer_move, player_move)
        cash += result
        cash_history.append(cash)
        computer_move = next_move(computer_move, transition_matrix_computer)
        if strategy == 1:
            player_move = np.random.choice(states,p=stationary_vector)
        else:
            learn(player_move, result, learning_rate)
            player_move = next_move(player_move, transition_matrix_player)

    print("Final score: ", cash)
    plt.plot(range(10001), cash_history)
    plt.xlabel("Game number")
    plt.ylabel("Cash")
    plt.show()


game(10000, 2)
