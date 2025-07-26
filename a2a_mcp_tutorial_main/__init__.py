import importlib, sys, pathlib

# 动态把内部的 `a2a_servers` 子包注入到 sys.modules，
# 这样可以直接 `import a2a_servers`，避免调整所有 import 路径。
_pkg_path = pathlib.Path(__file__).parent
_a2a_servers_path = _pkg_path / "a2a_servers"
if _a2a_servers_path.exists():
    module = importlib.import_module("a2a_mcp_tutorial_main.a2a_servers")
    sys.modules.setdefault("a2a_servers", module)
