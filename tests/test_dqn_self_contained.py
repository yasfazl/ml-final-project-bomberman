import ast
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DQN_DIR = REPO_ROOT / "agent_code" / "dqn_agent"


def test_dqn_runtime_has_no_sibling_agent_imports():
    for path in DQN_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "q_learning_agent" not in module, (
                    path.name,
                    module,
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "q_learning_agent" not in alias.name, (
                        path.name,
                        alias.name,
                    )


def test_dqn_imports_when_it_is_the_only_agent(tmp_path):
    isolated_root = tmp_path / "official_repo"
    isolated_agent = isolated_root / "agent_code" / "dqn_agent"
    isolated_agent.parent.mkdir(parents=True)
    (isolated_agent.parent / "__init__.py").write_text("", encoding="utf-8")
    shutil.copytree(DQN_DIR, isolated_agent)
    shutil.copy2(REPO_ROOT / "events.py", isolated_root / "events.py")
    shutil.copy2(REPO_ROOT / "settings.py", isolated_root / "settings.py")
    shutil.copy2(REPO_ROOT / "fallbacks.py", isolated_root / "fallbacks.py")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import agent_code.dqn_agent.callbacks; "
                "import agent_code.dqn_agent.train"
            ),
        ],
        cwd=isolated_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
