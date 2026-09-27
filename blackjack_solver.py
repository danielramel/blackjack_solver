from collections import defaultdict
from enum import Enum, auto
from random import Random


class ACTIONS(Enum):
    STAND = auto()
    HIT = auto()


CARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]


def draw_card(rng):
    return rng.choice(CARDS)


def dealer_total(dealer_upcard, rng):
    ace = dealer_upcard == 11
    total = dealer_upcard

    while total < 17:
        card = draw_card(rng)
        total += card

        if card == 11:
            if total > 21:
                total -= 10
            else:
                ace = True

        if total > 21 and ace:
            total -= 10
            ace = False

    return total


def reward_for_standing(player_total, dealer_final_total):
    if dealer_final_total > 21 or player_total > dealer_final_total:
        return 1.0
    if player_total < dealer_final_total:
        return -1.0
    return 0.0


def choose_training_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(list(ACTIONS))
    return max(ACTIONS, key=lambda action: q_table[state][action])


def q_learning_episode(q_table, visit_counts, discount_factor, epsilon, rng):
    ace = False
    card1 = draw_card(rng)
    card2 = draw_card(rng)
    player_total = card1 + card2

    if card1 == 11 or card2 == 11:
        ace = True
        if card1 == 11 and card2 == 11:
            player_total = 12

    dealer_upcard = draw_card(rng)

    while True:
        state = (player_total, dealer_upcard, ace)
        action = choose_training_action(q_table, state, epsilon, rng)

        if action == ACTIONS.STAND:
            reward = reward_for_standing(
                player_total,
                dealer_total(dealer_upcard, rng),
            )
            next_value = 0.0
            done = True
        else:
            card = draw_card(rng)
            player_total += card

            if card == 11:
                if player_total > 21:
                    player_total -= 10
                else:
                    ace = True

            if player_total > 21 and ace:
                player_total -= 10
                ace = False

            if player_total > 21:
                reward = -1.0
                next_value = 0.0
                done = True
            else:
                reward = 0.0
                next_state = (player_total, dealer_upcard, ace)
                next_value = max(q_table[next_state].values())
                done = False

        visit_counts[state][action] += 1
        learning_rate = 1 / visit_counts[state][action]
        old_value = q_table[state][action]
        target = reward + discount_factor * next_value
        q_table[state][action] = old_value + learning_rate * (target - old_value)

        if done:
            break


def train_q_learning(
    episodes=1_000_000,
    discount_factor=1.0,
    start_epsilon=1.0,
    end_epsilon=0.05,
    seed=42,
):
    rng = Random(seed)
    q_table = defaultdict(lambda: {action: 0.0 for action in ACTIONS})
    visit_counts = defaultdict(lambda: {action: 0 for action in ACTIONS})

    for episode in range(episodes):
        epsilon = max(
            end_epsilon,
            start_epsilon - (start_epsilon - end_epsilon) * episode / episodes,
        )
        q_learning_episode(q_table, visit_counts, discount_factor, epsilon, rng)

    return q_table, visit_counts
