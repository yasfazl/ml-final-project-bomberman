import numpy as np

from agent_code.q_learning_agent.game_utils import blast_tiles


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)

    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return field


def test_blast_extends_in_four_directions():
    field = open_field()

    result = set(blast_tiles(field, (4, 4)))

    expected = {
        (4, 4),
        (4, 3), (4, 2), (4, 1),
        (5, 4), (6, 4), (7, 4),
        (4, 5), (4, 6), (4, 7),
        (3, 4), (2, 4), (1, 4),
    }

    assert result == expected


def test_stone_wall_blocks_explosion():
    field = open_field()
    field[4, 2] = -1

    result = set(blast_tiles(field, (4, 4)))

    assert (4, 3) in result
    assert (4, 2) not in result
    assert (4, 1) not in result


def test_crate_is_hit_and_blocks_explosion():
    field = open_field()
    field[6, 4] = 1

    result = set(blast_tiles(field, (4, 4)))

    assert (5, 4) in result
    assert (6, 4) in result
    assert (7, 4) not in result