from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import settings as s


# Direction indices match callbacks.py:
# 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def blast_tiles(
    field: np.ndarray,
    bomb_position: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return every tile affected by a bomb explosion."""
    bomb_x, bomb_y = bomb_position
    width, height = field.shape

    affected_tiles = [(bomb_x, bomb_y)]

    for dx, dy in DIRECTIONS:
        for distance in range(1, s.BOMB_POWER + 1):
            position = (
                bomb_x + dx * distance,
                bomb_y + dy * distance,
            )
            x, y = position

            if not (0 <= x < width and 0 <= y < height):
                break

            tile_value = field[position]

            # Stone walls block the blast and are not affected.
            if tile_value == -1:
                break

            affected_tiles.append(position)

            # A crate is affected, but blocks the remaining blast.
            if tile_value == 1:
                break

    return affected_tiles


def _effective_bomb_timers(
    field: np.ndarray,
    bombs: list[tuple[tuple[int, int], int]],
    explosion_map: np.ndarray | None,
) -> list[float]:
    """Return bomb timers after accounting for chain reactions."""
    timers = [float(timer) for _position, timer in bombs]

    if explosion_map is not None:
        active_explosion = np.asarray(explosion_map) > 0

        for index, (position, _timer) in enumerate(bombs):
            if active_explosion[position]:
                timers[index] = 0.0

    changed = True

    while changed:
        changed = False

        for source_index, (source_position, _timer) in enumerate(bombs):
            source_blast = set(blast_tiles(field, source_position))
            source_timer = timers[source_index]

            for target_index, (target_position, _timer) in enumerate(bombs):
                if target_position not in source_blast:
                    continue

                if timers[target_index] > source_timer:
                    timers[target_index] = source_timer
                    changed = True

    return timers


def earliest_danger_times(
    game_state: dict,
    additional_bombs: Iterable[
        tuple[tuple[int, int], int]
    ] | None = None,
) -> np.ndarray:
    """
    Return the earliest bomb timer affecting each board tile.

    np.inf means safe from all currently known explosions.
    Zero means an explosion is currently active or imminent.
    """
    field = game_state["field"]

    danger_times = np.full(
        field.shape,
        np.inf,
        dtype=np.float64,
    )

    explosion_map = game_state.get("explosion_map")

    if explosion_map is not None:
        danger_times[np.asarray(explosion_map) > 0] = 0.0

    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", [])
    ]

    if additional_bombs is not None:
        bombs.extend(
            (tuple(position), int(timer))
            for position, timer in additional_bombs
        )

    effective_timers = _effective_bomb_timers(
        field,
        bombs,
        explosion_map,
    )

    for (bomb_position, _timer), effective_timer in zip(
        bombs,
        effective_timers,
    ):
        for position in blast_tiles(field, bomb_position):
            danger_times[position] = min(
                danger_times[position],
                effective_timer,
            )

    return danger_times


def tile_is_safe_at_time(
    danger_time: float,
    arrival_time: int,
) -> bool:
    """
    Return whether a tile is safe when reached after `arrival_time` moves.

    A bomb with timer t explodes after t + 1 actions. Its blast remains
    dangerous for one additional round.
    """
    if np.isinf(danger_time):
        return True

    explosion_time = float(danger_time) + 1.0
    dangerous_until = explosion_time + 1.0

    return (
        arrival_time < explosion_time
        or arrival_time > dangerous_until
    )


def nearest_safe_path(
    game_state: dict,
    additional_bombs: Iterable[
        tuple[tuple[int, int], int]
    ] | None = None,
) -> tuple[int | None, int | None]:
    """
    Find the shortest time-safe route to a tile outside every blast.

    Returns `(first_direction, distance)`. Special values:
    `(None, 0)` means the current position is already safe;
    `(None, None)` means no escape route was found.
    """
    field = game_state["field"]
    start_position = tuple(game_state["self"][3])

    danger_times = earliest_danger_times(
        game_state,
        additional_bombs=additional_bombs,
    )

    if np.isinf(danger_times[start_position]):
        return None, 0

    blocked_positions = {
        tuple(position)
        for position, _timer in game_state.get("bombs", [])
    }

    if additional_bombs is not None:
        blocked_positions.update(
            tuple(position)
            for position, _timer in additional_bombs
        )

    blocked_positions.update(
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    )

    # (position, first_direction, arrival_time)
    queue = [(start_position, None, 0)]
    queue_index = 0
    visited = {start_position}

    finite_times = danger_times[np.isfinite(danger_times)]

    if finite_times.size:
        maximum_search_steps = max(
            s.BOMB_TIMER,
            int(np.max(finite_times)) + 1,
        )
    else:
        maximum_search_steps = s.BOMB_TIMER

    while queue_index < len(queue):
        position, first_direction, arrival_time = queue[queue_index]
        queue_index += 1

        if arrival_time >= maximum_search_steps:
            continue

        x, y = position

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            next_position = (x + dx, y + dy)
            next_arrival_time = arrival_time + 1

            if next_position in visited:
                continue

            if next_position in blocked_positions:
                continue

            if field[next_position] != 0:
                continue

            if not tile_is_safe_at_time(
                danger_times[next_position],
                next_arrival_time,
            ):
                continue

            if first_direction is None:
                next_first_direction = direction_index
            else:
                next_first_direction = first_direction

            if np.isinf(danger_times[next_position]):
                return next_first_direction, next_arrival_time

            visited.add(next_position)
            queue.append(
                (
                    next_position,
                    next_first_direction,
                    next_arrival_time,
                )
            )

    return None, None


def has_escape_route_after_bomb(game_state: dict) -> bool:
    """Return whether placing a bomb now leaves a survivable route."""
    if not bool(game_state["self"][2]):
        return False

    start_position = tuple(game_state["self"][3])

    if any(
        tuple(position) == start_position
        for position, _timer in game_state.get("bombs", [])
    ):
        return False

    # At the next decision, the newly placed bomb is already at timer 3.
    timer_at_next_decision = max(s.BOMB_TIMER - 1, 0)

    _direction, distance = nearest_safe_path(
        game_state,
        additional_bombs=[
            (start_position, timer_at_next_decision)
        ],
    )

    return distance is not None


def bomb_target_counts(
    game_state: dict,
    bomb_position: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Count crates and opponents that a bomb would affect."""
    field = game_state["field"]

    if bomb_position is None:
        bomb_position = tuple(game_state["self"][3])

    affected_tiles = set(blast_tiles(field, bomb_position))

    crate_count = sum(
        field[position] == 1
        for position in affected_tiles
    )
    opponent_count = sum(
        tuple(opponent[3]) in affected_tiles
        for opponent in game_state.get("others", [])
    )

    return int(crate_count), int(opponent_count)


def nearest_opponent_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """Find the shortest walkable path to a tile adjacent to an opponent."""
    field = game_state["field"]
    start_position = tuple(game_state["self"][3])

    opponent_positions = {
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    }

    if not opponent_positions:
        return None, None

    blocked_positions = {
        tuple(position)
        for position, _timer in game_state.get("bombs", [])
    }
    blocked_positions.update(opponent_positions)

    # Queue entries: (position, first_direction, distance)
    queue = [(start_position, None, 0)]
    queue_index = 0
    visited = {start_position}

    while queue_index < len(queue):
        position, first_direction, distance = queue[queue_index]
        queue_index += 1

        if any(
            abs(position[0] - opponent_x) + abs(position[1] - opponent_y)
            == 1
            for opponent_x, opponent_y in opponent_positions
        ):
            if distance > 0:
                return first_direction, distance

        x, y = position

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            next_position = (x + dx, y + dy)

            if next_position in visited:
                continue
            if next_position in blocked_positions:
                continue
            if field[next_position] != 0:
                continue

            visited.add(next_position)
            next_first_direction = (
                direction_index
                if first_direction is None
                else first_direction
            )

            queue.append(
                (
                    next_position,
                    next_first_direction,
                    distance + 1,
                )
            )

    return None, None


def nearest_crate_bombing_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """Find the nearest reachable tile for safely bombing a crate."""
    field = game_state["field"]
    start_position = tuple(game_state["self"][3])

    if not np.any(field == 1):
        return None, None

    blocked_positions = {
        tuple(position)
        for position, _timer in game_state.get("bombs", [])
    }
    blocked_positions.update(
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    )

    # (position, first_direction, distance)
    queue = [(start_position, None, 0)]
    queue_index = 0
    visited = {start_position}

    while queue_index < len(queue):
        position, first_direction, distance = queue[queue_index]
        queue_index += 1

        crate_count, _opponent_count = bomb_target_counts(
            game_state,
            bomb_position=position,
        )

        if crate_count > 0:
            simulated_state = dict(game_state)
            agent_info = list(game_state["self"])
            agent_info[2] = True
            agent_info[3] = position
            simulated_state["self"] = tuple(agent_info)

            if has_escape_route_after_bomb(simulated_state):
                return first_direction, distance

        x, y = position

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            next_position = (x + dx, y + dy)

            if next_position in visited:
                continue
            if next_position in blocked_positions:
                continue
            if field[next_position] != 0:
                continue

            visited.add(next_position)
            next_first_direction = (
                direction_index
                if first_direction is None
                else first_direction
            )
            queue.append(
                (
                    next_position,
                    next_first_direction,
                    distance + 1,
                )
            )

    return None, None


def best_crate_bombing_path(
    game_state: dict,
    max_search_distance: int = 1,
) -> tuple[int | None, int | None, int]:
    """
    Find a nearby safe bomb position with the highest crate yield.

    Candidate positions are reachable empty tiles no more than
    ``max_search_distance`` movements away. A candidate is accepted only
    when placing a bomb there would hit at least one crate and still leave
    a time-aware escape route.

    Candidates are ordered by:

    1. larger number of crates hit;
    2. shorter path distance;
    3. BFS direction order for deterministic tie-breaking.

    Returns:
        first_direction:
            0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        distance:
            movements required to reach the selected position
        crate_count:
            crates affected by a bomb at that position

    Special result ``(None, None, 0)`` means no suitable position exists.
    ``first_direction`` is None with distance zero when the current tile is
    already the best position.
    """
    field = game_state["field"]
    start_position = tuple(game_state["self"][3])

    if max_search_distance < 0:
        raise ValueError("max_search_distance must be non-negative")

    if not np.any(field == 1):
        return None, None, 0

    blocked_positions = {
        tuple(position)
        for position, _timer in game_state.get("bombs", [])
    }
    blocked_positions.update(
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    )

    # Queue entries: (position, first_direction, distance)
    queue = [(start_position, None, 0)]
    queue_index = 0
    visited = {start_position}

    best_direction = None
    best_distance = None
    best_crate_count = 0

    while queue_index < len(queue):
        position, first_direction, distance = queue[queue_index]
        queue_index += 1

        crate_count, _opponent_count = bomb_target_counts(
            game_state,
            bomb_position=position,
        )

        if crate_count > 0:
            simulated_state = dict(game_state)
            agent_info = list(game_state["self"])
            agent_info[2] = True
            agent_info[3] = position
            simulated_state["self"] = tuple(agent_info)

            if has_escape_route_after_bomb(simulated_state):
                candidate_key = (crate_count, -distance)
                best_key = (
                    best_crate_count,
                    -best_distance if best_distance is not None else 0,
                )

                if (
                    best_distance is None
                    or candidate_key > best_key
                ):
                    best_direction = first_direction
                    best_distance = distance
                    best_crate_count = crate_count

        if distance >= max_search_distance:
            continue

        x, y = position

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            next_position = (x + dx, y + dy)

            if next_position in visited:
                continue
            if next_position in blocked_positions:
                continue
            if field[next_position] != 0:
                continue

            visited.add(next_position)
            next_first_direction = (
                direction_index
                if first_direction is None
                else first_direction
            )
            queue.append(
                (
                    next_position,
                    next_first_direction,
                    distance + 1,
                )
            )

    if best_distance is None:
        return None, None, 0

    return best_direction, best_distance, best_crate_count
