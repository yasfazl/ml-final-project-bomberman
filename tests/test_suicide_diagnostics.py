from types import SimpleNamespace

from suicide_diagnostics import DeathDiagnostics, classify_death


def cause(killer, *, placed_step=10):
    return {
        "killer": killer,
        "bomb_position": [5, 5],
        "placed_step": placed_step,
        "blast_tiles": [[5, 5], [5, 4], [5, 6], [4, 5], [6, 5]],
    }


def action(step, position, *, event="MOVED_UP", safety=None):
    return {
        "step": step,
        "outcome_event": event,
        "previous_position": list(position),
        "resulting_position": list(position),
        "opponent_adjacent": False,
        "opponent_bombs_present": False,
        "safety": safety,
    }


def test_classifies_ground_truth_opponent_blast():
    result = classify_death(
        "dqn_agent",
        14,
        (5, 5),
        [cause("rule_based_agent")],
        [],
    )
    assert result["category"] == "opponent_blast"
    assert result["opponent_kill"]
    assert not result["self_kill"]


def test_classifies_unsafe_own_bomb_placement():
    history = [
        action(
            10,
            (5, 5),
            event="BOMB_DROPPED",
            safety={
                "selected_action_survivable": False,
                "robust_bomb_escape": False,
                "legacy_bomb_escape": True,
            },
        )
    ]
    result = classify_death(
        "dqn_agent",
        14,
        (5, 5),
        [cause("dqn_agent")],
        history,
    )
    assert result["category"] == "unsafe_own_bomb_placement"
    assert result["self_kill"]


def test_classifies_reentry_after_leaving_own_blast():
    history = [
        action(
            10,
            (5, 5),
            event="BOMB_DROPPED",
            safety={
                "selected_action_survivable": True,
                "robust_bomb_escape": True,
                "legacy_bomb_escape": True,
            },
        ),
        action(11, (4, 4), safety={"selected_action_survivable": True}),
        action(12, (5, 4), safety={"selected_action_survivable": True}),
    ]
    result = classify_death(
        "dqn_agent",
        14,
        (5, 4),
        [cause("dqn_agent")],
        history,
    )
    assert result["category"] == "reentered_own_blast"
    assert result["left_own_blast"]
    assert result["reentered_own_blast"]


def test_classifies_failed_clear_when_agent_never_leaves_blast():
    history = [
        action(
            10,
            (5, 5),
            event="BOMB_DROPPED",
            safety={
                "selected_action_survivable": True,
                "robust_bomb_escape": True,
                "legacy_bomb_escape": True,
            },
        ),
        action(11, (5, 4), safety={"selected_action_survivable": True}),
    ]
    result = classify_death(
        "dqn_agent",
        14,
        (5, 4),
        [cause("dqn_agent")],
        history,
    )
    assert result["category"] == "failed_to_clear_own_blast"
    assert not result["left_own_blast"]


def test_classifies_unsafe_post_bomb_action():
    history = [
        action(
            10,
            (5, 5),
            event="BOMB_DROPPED",
            safety={
                "selected_action_survivable": True,
                "robust_bomb_escape": True,
                "legacy_bomb_escape": True,
            },
        ),
        action(11, (5, 4), safety={"selected_action_survivable": False}),
    ]
    result = classify_death(
        "dqn_agent",
        14,
        (5, 4),
        [cause("dqn_agent")],
        history,
    )
    assert result["category"] == "unsafe_post_bomb_action"
    assert result["unsafe_post_bomb_action_seen"]


def test_diagnostics_disabled_does_not_create_file(tmp_path):
    path = tmp_path / "disabled.jsonl"
    diagnostics = DeathDiagnostics(
        enabled=False,
        path=path,
        scenario="classic",
        seed=42,
    )
    diagnostics.start_round(1)
    diagnostics.record_action(
        world=SimpleNamespace(),
        agent=SimpleNamespace(code_name="dqn_agent"),
        requested_action="WAIT",
        outcome_event="WAITED",
        previous_position=(1, 1),
    )
    assert not path.exists()


def test_enabled_diagnostics_writes_header_only_until_a_death(tmp_path):
    path = tmp_path / "enabled.jsonl"
    DeathDiagnostics(
        enabled=True,
        path=path,
        scenario="empty",
        seed=7,
    )
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert '"record_type": "run_start"' in lines[0]
