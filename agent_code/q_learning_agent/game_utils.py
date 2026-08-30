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

ACTION_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}


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


def safety_blast_tiles(
    field: np.ndarray,
    bomb_position: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return blast tiles using the rules implemented by the game engine.

    The engine stops explosions at stone walls but does not stop them at
    crates.  The learned feature helpers intentionally keep their historical
    ``blast_tiles`` semantics; this engine-accurate variant is used only by
    the inference-time survivability shield.
    """
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

            if field[position] == -1:
                break

            affected_tiles.append(position)

    return affected_tiles


def _effective_safety_bomb_timers(
    field: np.ndarray,
    bombs: list[tuple[tuple[int, int], int]],
    explosion_map: np.ndarray | None,
) -> list[float]:
    """Return conservative bomb timers for the survivability shield."""
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
            source_blast = set(
                safety_blast_tiles(field, source_position)
            )
            source_timer = timers[source_index]

            for target_index, (target_position, _timer) in enumerate(bombs):
                if target_position not in source_blast:
                    continue

                if timers[target_index] > source_timer:
                    timers[target_index] = source_timer
                    changed = True

    return timers


def time_indexed_danger_schedule(
    game_state: dict,
    additional_bombs: Iterable[
        tuple[tuple[int, int], int]
    ] | None = None,
    horizon: int | None = None,
) -> np.ndarray:
    """Return whether every tile is dangerous at each future action time.

    Index zero represents the current decision state.  Index one represents
    the board after the action selected now has been executed and world
    elements have advanced once.  Unlike ``earliest_danger_times``, this
    schedule preserves separate explosion windows from multiple bombs.
    """
    field = game_state["field"]
    explosion_map = game_state.get("explosion_map")
    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", [])
    ]

    if additional_bombs is not None:
        bombs.extend(
            (tuple(position), int(timer))
            for position, timer in additional_bombs
        )

    effective_timers = _effective_safety_bomb_timers(
        field,
        bombs,
        explosion_map,
    )

    latest_bomb_danger = max(
        (
            int(timer) + s.EXPLOSION_TIMER
            for timer in effective_timers
        ),
        default=0,
    )

    active_explosion_steps = 0
    if explosion_map is not None and np.size(explosion_map):
        active_explosion_steps = int(
            np.ceil(np.max(np.asarray(explosion_map)))
        )

    if horizon is None:
        horizon = max(
            s.BOMB_TIMER + s.EXPLOSION_TIMER,
            latest_bomb_danger,
            active_explosion_steps,
        )

    if horizon < 1:
        raise ValueError("horizon must be at least one future action")

    danger = np.zeros(
        (horizon + 1, *field.shape),
        dtype=bool,
    )

    if explosion_map is not None:
        remaining = np.asarray(explosion_map)

        for time_step in range(1, horizon + 1):
            danger[time_step] |= remaining >= time_step

    for (bomb_position, _timer), effective_timer in zip(
        bombs,
        effective_timers,
    ):
        explosion_time = int(effective_timer) + 1
        dangerous_until = (
            explosion_time + s.EXPLOSION_TIMER - 1
        )

        for time_step in range(
            max(explosion_time, 1),
            min(dangerous_until, horizon) + 1,
        ):
            for position in safety_blast_tiles(
                field,
                bomb_position,
            ):
                danger[time_step, position[0], position[1]] = True

    return danger


def action_has_survival_route(
    game_state: dict,
    action: str,
    horizon: int | None = None,
) -> bool:
    """Return whether an action leaves a route through all known danger.

    The search uses ``(position, time)`` states, so waiting and revisiting a
    tile after an explosion are represented correctly.  Opponent positions
    are treated as blocked for the short planning horizon.  The caller keeps
    a non-empty fallback if every candidate is rejected.
    """
    if action not in ACTION_DELTAS:
        raise ValueError(f"Unknown action: {action}")

    field = game_state["field"]
    width, height = field.shape
    start_position = tuple(game_state["self"][3])
    additional_bombs = None

    if action == "BOMB":
        additional_bombs = [
            (start_position, int(s.BOMB_TIMER))
        ]

    danger = time_indexed_danger_schedule(
        game_state,
        additional_bombs=additional_bombs,
        horizon=horizon,
    )
    final_time = danger.shape[0] - 1

    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", [])
    ]
    if additional_bombs is not None:
        bombs.extend(additional_bombs)

    effective_timers = _effective_safety_bomb_timers(
        field,
        bombs,
        game_state.get("explosion_map"),
    )
    bomb_blocked_until = {
        position: int(timer) + 1
        for (position, _timer), timer in zip(
            bombs,
            effective_timers,
        )
    }
    opponent_positions = {
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    }

    dx, dy = ACTION_DELTAS[action]
    first_position = (
        start_position[0] + dx,
        start_position[1] + dy,
    )

    if not (
        0 <= first_position[0] < width
        and 0 <= first_position[1] < height
    ):
        return False

    if field[first_position] != 0:
        return False

    if first_position in opponent_positions:
        return False

    first_blocked_until = bomb_blocked_until.get(first_position)
    if (
        first_blocked_until is not None
        and first_position != start_position
        and 1 <= first_blocked_until
    ):
        return False

    if danger[1, first_position[0], first_position[1]]:
        return False

    reachable_positions = {first_position}
    search_deltas = (*DIRECTIONS, (0, 0))

    for time_step in range(2, final_time + 1):
        next_reachable_positions = set()

        for position in reachable_positions:
            for move_dx, move_dy in search_deltas:
                next_position = (
                    position[0] + move_dx,
                    position[1] + move_dy,
                )
                x, y = next_position

                if not (0 <= x < width and 0 <= y < height):
                    continue

                if field[next_position] != 0:
                    continue

                if next_position in opponent_positions:
                    continue

                blocked_until = bomb_blocked_until.get(next_position)
                if (
                    blocked_until is not None
                    and next_position != position
                    and time_step <= blocked_until
                ):
                    continue

                if danger[time_step, x, y]:
                    continue

                next_reachable_positions.add(next_position)

        if not next_reachable_positions:
            return False

        reachable_positions = next_reachable_positions

    return bool(reachable_positions)


def bomb_has_robust_escape_route(
    game_state: dict,
    maximum_escape_steps: int = 2,
) -> bool:
    """Return whether a new bomb has a short, disruption-tolerant escape.

    A merely possible four-step escape is fragile when an opponent is nearby:
    one collision can consume the only spare action and turn the bomb into a
    suicide.  The ordinary time-expanded route is sufficient when every
    opponent is far away.  Otherwise this gate requires at least two distinct
    exit tiles outside the engine-accurate blast within
    ``maximum_escape_steps`` movements.  One opponent cannot occupy both
    exits at once.
    """
    if maximum_escape_steps < 1:
        raise ValueError("maximum_escape_steps must be positive")

    if not bool(game_state["self"][2]):
        return False

    field = game_state["field"]
    width, height = field.shape
    start_position = tuple(game_state["self"][3])

    if any(
        tuple(position) == start_position
        for position, _timer in game_state.get("bombs", [])
    ):
        return False

    if not action_has_survival_route(game_state, "BOMB"):
        return False

    opponent_positions = {
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    }
    nearby_opponent = any(
        abs(position[0] - start_position[0])
        + abs(position[1] - start_position[1])
        <= s.BOMB_POWER + 1
        for position in opponent_positions
    )

    if not nearby_opponent:
        return True

    additional_bomb = (
        start_position,
        int(s.BOMB_TIMER),
    )
    danger = time_indexed_danger_schedule(
        game_state,
        additional_bombs=[additional_bomb],
    )

    if danger[1, start_position[0], start_position[1]]:
        return False

    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", [])
    ]
    bombs.append(additional_bomb)
    effective_timers = _effective_safety_bomb_timers(
        field,
        bombs,
        game_state.get("explosion_map"),
    )
    bomb_blocked_until = {
        position: int(timer) + 1
        for (position, _timer), timer in zip(
            bombs,
            effective_timers,
        )
    }
    own_blast = set(
        safety_blast_tiles(field, start_position)
    )
    reachable_positions = {start_position}
    safe_exit_positions = set()

    for movement_step in range(1, maximum_escape_steps + 1):
        time_step = movement_step + 1
        next_reachable_positions = set()

        for position in reachable_positions:
            for dx, dy in DIRECTIONS:
                next_position = (
                    position[0] + dx,
                    position[1] + dy,
                )
                x, y = next_position

                if not (0 <= x < width and 0 <= y < height):
                    continue

                if field[next_position] != 0:
                    continue

                if next_position in opponent_positions:
                    continue

                blocked_until = bomb_blocked_until.get(next_position)
                if (
                    blocked_until is not None
                    and next_position != position
                    and time_step <= blocked_until
                ):
                    continue

                if danger[time_step, x, y]:
                    continue

                if next_position not in own_blast:
                    safe_exit_positions.add(next_position)
                    continue

                next_reachable_positions.add(next_position)

        if len(safe_exit_positions) >= 2:
            return True

        if not next_reachable_positions:
            break

        reachable_positions = next_reachable_positions

    return len(safe_exit_positions) >= 2


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
