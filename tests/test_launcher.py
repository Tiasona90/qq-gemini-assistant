"""Launcher regression tests with isolated config and no external API requests."""
import tkinter as tk
import io
from unittest.mock import Mock

import pytest
import gui_launcher as gui


@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def launcher(tmp_path, monkeypatch, tk_root):
    monkeypatch.setattr(gui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gui, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(gui, "ENV_EXAMPLE", tmp_path / ".env.example")
    (tmp_path / ".env").write_text("UNRELATED_OPTION=keep\n", encoding="utf-8")
    monkeypatch.setattr(gui.messagebox, "showinfo", Mock())
    monkeypatch.setattr(gui.messagebox, "showerror", Mock())
    monkeypatch.setattr(gui.messagebox, "showwarning", Mock())
    root = tk_root
    app = gui.AssistantGUI(root)
    yield app
    for callback in root.tk.splitlist(root.tk.call("after", "info")):
        root.after_cancel(callback)
    for widget in root.winfo_children():
        widget.destroy()


def test_launcher_pages_build_and_switch(launcher):
    app = launcher
    assert len(app.notebook.tabs()) == 5
    for tab in app.notebook.tabs():
        app.notebook.select(tab)
        app._sync_navigation()
        app.root.update_idletasks()
    assert "服务未启动" in app.overview_text.get()


def test_launcher_save_roundtrip_preserves_unknown_options(launcher):
    app = launcher
    app.ctx_var.set(0)
    app.search_var.set(False)
    app.forward_url_var.set("ws://localhost:3001/events")
    assert app.save_ui_to_env(notify=False)
    saved = gui.dotenv.dotenv_values(gui.ENV_FILE)
    assert saved["UNRELATED_OPTION"] == "keep"
    assert saved["DEFAULT_CONTEXT_MESSAGES"] == "0"
    assert saved["ENABLE_SEARCH_DEFAULT"] == "false"
    app.load_env_to_ui()
    assert app.ctx_var.get() == 0
    assert app.search_var.get() is False
    assert app.forward_url_var.get() == "ws://localhost:3001/events"


def test_invalid_config_does_not_write_or_start(launcher, monkeypatch):
    app = launcher
    before = gui.ENV_FILE.read_bytes()
    app.key_var.set("dummy-key")
    app.whitelist_var.set("10001")
    app.port_var.set(70000)
    spawn = Mock()
    monkeypatch.setattr(gui.subprocess, "Popen", spawn)
    app.start_service()
    assert gui.ENV_FILE.read_bytes() == before
    spawn.assert_not_called()
    gui.messagebox.showerror.assert_called_once()


def test_service_start_and_async_stop(launcher, monkeypatch):
    app = launcher
    app.key_var.set("dummy-key")
    app.whitelist_var.set("10001")
    process = Mock(pid=1234, stdout=io.StringIO(""))
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(gui.subprocess, "Popen", spawn)
    app.start_service()
    assert app.is_running
    assert spawn.call_args.kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    gui.messagebox.showinfo.assert_not_called()
    app.stop_service()
    assert app.stopping
    process.terminate.assert_called_once()
    process.poll.return_value = 0
    app._finish_stop()
    assert not app.is_running
    assert not app.stopping


def test_logs_redact_keys_and_bound_memory(launcher):
    app = launcher
    app.key_var.set("dummy-secret-key")
    app.token_var.set("dummy-access-token")
    app.append_log("dummy-secret-key dummy-access-token")
    text = app.log_area.get("1.0", tk.END)
    assert "dummy-secret-key" not in text
    assert "dummy-access-token" not in text
    app.append_log("line\n" * 5100)
    assert int(app.log_area.index("end-1c").split(".")[0]) <= 5001
