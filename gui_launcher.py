"""
GUI Parameter Configuration and Service Launcher for QQ Gemini Assistant.
Provides a modern desktop interface for:
1. One-click Start / Stop of the QQ Assistant daemon
2. Full parameter configuration (.env reading & saving)
3. Real-time log streaming with search & copy
4. Token usage and cost statistics dashboard
5. Diagnostics: Gemini API connectivity test & automated test execution
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import dotenv

# Determine project directories
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

# Ensure .env exists
if not ENV_FILE.exists() and ENV_EXAMPLE.exists():
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)


class AssistantGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("QQ Gemini 智能助手 - 控制中心与配置启动器")
        icon_file = PROJECT_ROOT / "assets" / "launcher.ico"
        if icon_file.exists() and sys.platform == "win32":
            try:
                self.root.iconbitmap(str(icon_file))
            except tk.TclError:
                pass
        self.root.geometry("1180x820")
        self.root.configure(bg="#eef2f8")
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.root.minsize(1020, 720)
        self.stopping = False
        self.close_after_stop = False
        self.started_at = None

        # Child process management
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False

        # Set up ttk styling
        self._init_styles()

        # Build UI layout
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # Load configuration into fields
        self.load_env_to_ui()
        self._refresh_overview()

        # Start background timer for log consumption and status polling
        self.root.after(100, self._process_log_queue)
        self.root.after(1000, self._poll_service_status)

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#eef2f8", foreground="#26344b")
        style.configure("TFrame", background="#eef2f8")
        style.configure("TLabel", background="#eef2f8")
        style.configure("TButton", padding=(14, 8), borderwidth=0, background="#e1e8f4")
        style.map("TButton", background=[("active", "#d2def1")])
        style.configure("TEntry", padding=7, fieldbackground="#ffffff", bordercolor="#d8e0ed")
        style.configure("TCombobox", padding=6, fieldbackground="#ffffff")
        style.configure("TSpinbox", padding=6, fieldbackground="#ffffff")
        style.configure("TLabelframe", background="#ffffff", bordercolor="#dce4ef", relief="solid")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#344a6e", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Header.TFrame", background="#111e36")
        style.configure("HeaderTitle.TLabel", background="#111e36", foreground="#ffffff", font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("HeaderStatus.TLabel", background="#111e36", foreground="#a8bddc")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 12, "bold"), foreground="#142848")
        style.configure("FieldLabel.TLabel", foreground="#465977")
        style.configure("StatNum.TLabel", font=("Segoe UI", 19, "bold"), foreground="#4263eb")
        style.configure("StatLabel.TLabel", foreground="#6b7c96")
        style.layout("Workspace.TNotebook.Tab", [])
        style.configure("Workspace.TNotebook", background="#eef2f8", borderwidth=0)

    def _build_header(self) -> None:
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=(28, 22))
        header_frame.pack(fill=tk.X)

        title_box = ttk.Frame(header_frame, style="Header.TFrame")
        title_box.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = ttk.Label(title_box, text="QQ Gemini 智能助手", style="HeaderTitle.TLabel")
        title_lbl.pack(anchor=tk.W)

        self.status_lbl = ttk.Label(
            title_box,
            text="服务未启动 · 请先检查配置",
            style="HeaderStatus.TLabel"
        )
        self.status_lbl.pack(anchor=tk.W, pady=(2, 0))

        # Action button container in header
        btn_box = ttk.Frame(header_frame, style="Header.TFrame")
        btn_box.pack(side=tk.RIGHT, fill=tk.Y)

        self.start_btn = tk.Button(
            btn_box,
            text="▶ 启动服务",
            command=self.start_service,
            bg="#5271ff",
            fg="#ffffff",
            activebackground="#4059d9",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.start_btn.pack(side=tk.LEFT, padx=6)

        self.stop_btn = tk.Button(
            btn_box,
            text="⏹ 停止服务",
            command=self.stop_service,
            bg="#344764",
            fg="#ffffff",
            activebackground="#465a77",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=6)

    def _build_notebook(self) -> None:
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(body, bg="#182740", width=196)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="工作空间", bg="#182740", fg="#849bbc",
                 font=("Microsoft YaHei UI", 10), anchor="w").pack(fill=tk.X, padx=22, pady=(28, 14))
        self.notebook = ttk.Notebook(body, style="Workspace.TNotebook")
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=22, pady=20)
        self.nav_buttons = []
        pages = [("运行总览", "服务状态与快捷操作", "tab_home", self._build_home),
                 ("参数配置", "模型、连接与会话", "tab_config", self._build_config_tab),
                 ("运行日志", "实时输出与故障排查", "tab_logs", self._build_logs_tab),
                 ("用量统计", "请求、Token 与估算", "tab_stats", self._build_stats_tab),
                 ("诊断工具", "API 测试与离线模拟", "tab_tools", self._build_tools_tab)]
        for index, (title, desc, attr, builder) in enumerate(pages):
            page = ttk.Frame(self.notebook, padding=4)
            setattr(self, attr, page)
            self.notebook.add(page, text=title)
            button = tk.Button(sidebar, text=f"{title}\n{desc}", anchor="w", justify="left",
                               bg="#182740", fg="#a9bbd4", activebackground="#294063",
                               activeforeground="#ffffff", relief="flat", bd=0, padx=20, pady=16,
                               cursor="hand2", command=lambda i=index: self.notebook.select(i))
            button.pack(fill=tk.X, padx=10, pady=3)
            self.nav_buttons.append(button)
            builder()
        tk.Label(sidebar, text="QQ × GEMINI\n本地控制中心  /  v2.0", justify="left", anchor="w",
                 bg="#182740", fg="#758cad", font=("Microsoft YaHei UI", 9)).pack(side=tk.BOTTOM, fill=tk.X, padx=22, pady=24)
        self.notebook.bind("<<NotebookTabChanged>>", self._sync_navigation)
        self._sync_navigation()

    def _sync_navigation(self, event=None) -> None:
        current = self.notebook.index(self.notebook.select())
        for index, button in enumerate(self.nav_buttons):
            button.configure(bg="#304976" if index == current else "#182740",
                             fg="#ffffff" if index == current else "#a9bbd4")
        if hasattr(self, "overview_text") and hasattr(self, "db_path_var"):
            self._refresh_overview()

    def _build_home(self) -> None:
        ttk.Label(self.tab_home, text="让每一次对话，都准备就绪。", font=("Microsoft YaHei UI", 23, "bold"),
                  foreground="#182b4c").pack(anchor="w", pady=(14, 8))
        ttk.Label(self.tab_home, text="配置 Gemini、连接 QQ，并在一个窗口里管理运行状态。",
                  foreground="#70819a").pack(anchor="w", pady=(0, 26))
        card = ttk.LabelFrame(self.tab_home, text=" 运行概况 ", padding=24)
        card.pack(fill=tk.X, pady=(0, 20))
        self.overview_text = tk.StringVar(value="正在读取配置…")
        ttk.Label(card, textvariable=self.overview_text, justify="left", font=("Microsoft YaHei UI", 11),
                  wraplength=680).pack(anchor="w")
        actions = ttk.Frame(self.tab_home)
        actions.pack(fill=tk.X, pady=(0, 22))
        ttk.Button(actions, text="编辑连接与模型", command=lambda: self.notebook.select(self.tab_config)).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(actions, text="查看运行日志", command=lambda: self.notebook.select(self.tab_logs)).pack(side=tk.LEFT)
        guide = ttk.LabelFrame(self.tab_home, text=" 三步开始使用 ", padding=24)
        guide.pack(fill=tk.X)
        for title, detail in [
            ("01  配置模型与访问权限", "填写 API Key 和 QQ 白名单；保存前会检查参数。"),
            ("02  连接 NapCat / OneBot 11", "选择连接模式，核对 WebSocket 地址和 Access Token。"),
            ("03  启动服务，观察日志", "右上角启动后查看连接日志；进程运行不代表 QQ 已连接。")]:
            ttk.Label(guide, text=title, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(6, 4))
            ttk.Label(guide, text=detail, foreground="#6a7c96").pack(anchor="w", pady=(0, 12))

    def _refresh_overview(self) -> None:
        state = "正在停止" if self.stopping else ("进程运行中" if self.is_running else "服务未启动")
        duration = f" · 已运行 {int(time.monotonic() - self.started_at)} 秒" if self.started_at else ""
        mode = self.mode_var.get()
        if mode == "reverse_ws":
            endpoint = f"ws://{self.host_var.get()}:{self.root.globalgetvar(self.port_var._name)}{self.path_var.get()}"
        else:
            endpoint = self.forward_url_var.get()
        count = len([v for v in self.whitelist_var.get().split(",") if v.strip()])
        self.overview_text.set(f"{state}{duration}\n\n默认模型   {self.model_var.get()}\n连接地址   {endpoint}\n访问权限   {count} 个 QQ 账号\n\nQQ 连接状态请以运行日志为准；配置保存后下次启动生效。")

    def _build_config_tab(self) -> None:
        # Create a scrollable container for config
        canvas = tk.Canvas(self.tab_config, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_config, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        def scroll_config(event):
            if self.notebook.select() == str(self.tab_config):
                canvas.yview_scroll(-int(event.delta / 120), "units")
        self.root.bind("<MouseWheel>", scroll_config, add="+")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Section 1: Gemini API
        sec1 = ttk.LabelFrame(scroll_frame, text=" Google Gemini 官方 API 配置 ", padding=12)
        sec1.pack(fill=tk.X, pady=(0, 12))

        # API Key with show/hide toggle
        ttk.Label(sec1, text="Gemini API Key:", style="FieldLabel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        key_box = ttk.Frame(sec1)
        key_box.grid(row=0, column=1, sticky=tk.EW, pady=4, padx=6)
        sec1.columnconfigure(1, weight=1)

        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(key_box, textvariable=self.key_var, show="*")
        self.key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.show_key_var = tk.BooleanVar(value=False)
        self.toggle_key_btn = ttk.Checkbutton(
            key_box,
            text="显示明文",
            variable=self.show_key_var,
            command=self._toggle_key_visibility
        )
        self.toggle_key_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Model selection
        ttk.Label(sec1, text="默认模型 (Model):", style="FieldLabel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.model_var = tk.StringVar(value="gemini-3.8-flash")
        model_cb = ttk.Combobox(
            sec1,
            textvariable=self.model_var,
            values=[
                "gemini-3.8-flash",
                "gemini-3.5-flash-lite",
                "gemini-4-argon",
            ]
        )
        model_cb.grid(row=1, column=1, sticky=tk.W, pady=4, padx=6)

        # Thinking Level (strictly 3 tiers: LOW, MEDIUM, HIGH)
        ttk.Label(sec1, text="思考强度 (Thinking):", style="FieldLabel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.thinking_var = tk.StringVar(value="LOW")
        thinking_cb = ttk.Combobox(
            sec1,
            textvariable=self.thinking_var,
            values=["LOW", "MEDIUM", "HIGH"],
            state="readonly"
        )
        thinking_cb.grid(row=2, column=1, sticky=tk.W, pady=4, padx=6)

        # Context Messages & Max Tokens
        ttk.Label(sec1, text="默认携带历史消息:", style="FieldLabel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=4)
        ctx_box = ttk.Frame(sec1)
        ctx_box.grid(row=3, column=1, sticky=tk.W, pady=4, padx=6)

        self.ctx_var = tk.IntVar(value=4)
        ttk.Spinbox(ctx_box, from_=0, to=40, textvariable=self.ctx_var, width=6).pack(side=tk.LEFT)
        ttk.Label(ctx_box, text=" 条 (4条即包含前2轮问答对话)").pack(side=tk.LEFT, padx=4)

        ttk.Label(sec1, text="单次生成最大 Token:", style="FieldLabel.TLabel").grid(row=4, column=0, sticky=tk.W, pady=4)
        self.tokens_var = tk.IntVar(value=2048)
        ttk.Spinbox(sec1, from_=256, to=8192, increment=256, textvariable=self.tokens_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=4, padx=6)

        # Search Grounding Default Checkbox
        self.search_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sec1,
            text="默认允许 Google Search 实时搜索（用户可单独关闭）",
            variable=self.search_var
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(6, 2))

        # System Prompt Editor
        ttk.Label(sec1, text="系统提示词 (Prompt):", style="FieldLabel.TLabel").grid(row=6, column=0, sticky=tk.NW, pady=(6, 4))
        self.prompt_text = scrolledtext.ScrolledText(
            sec1,
            height=4,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
            bg="#f8fafc",
            fg="#0f172a"
        )
        self.prompt_text.grid(row=6, column=1, sticky=tk.EW, pady=(6, 4), padx=6)

        # Section 2: QQ / OneBot 11
        sec2 = ttk.LabelFrame(scroll_frame, text=" QQ / OneBot 11 通信配置 ", padding=12)
        sec2.pack(fill=tk.X, pady=(0, 12))
        sec2.columnconfigure(1, weight=1)

        # Whitelist
        ttk.Label(sec2, text="允许的 QQ 白名单:", style="FieldLabel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.whitelist_var = tk.StringVar()
        ttk.Entry(sec2, textvariable=self.whitelist_var).grid(row=0, column=1, sticky=tk.EW, pady=4, padx=6)
        ttk.Label(sec2, text="* 多个 QQ 号以半角逗号分隔，留空将拒绝一切消息", foreground="#94a3b8", font=("Segoe UI", 8)).grid(row=1, column=1, sticky=tk.W, padx=6)

        # Mode & Port
        ttk.Label(sec2, text="通信架构模式:", style="FieldLabel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4)
        mode_box = ttk.Frame(sec2)
        mode_box.grid(row=2, column=1, sticky=tk.W, pady=4, padx=6)

        self.mode_var = tk.StringVar(value="reverse_ws")
        ttk.Radiobutton(mode_box, text="反向连接（NapCat 连入）", variable=self.mode_var, value="reverse_ws").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_box, text="正向连接", variable=self.mode_var, value="forward_ws").pack(side=tk.LEFT, padx=12)

        ttk.Label(sec2, text="反向监听端口:", style="FieldLabel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.port_var = tk.IntVar(value=8080)
        ttk.Spinbox(sec2, from_=1024, to=65535, textvariable=self.port_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=4, padx=6)

        ttk.Label(sec2, text="通信 Access Token:", style="FieldLabel.TLabel").grid(row=4, column=0, sticky=tk.W, pady=4)
        self.token_var = tk.StringVar()
        ttk.Entry(sec2, textvariable=self.token_var, show="*").grid(row=4, column=1, sticky=tk.EW, pady=4, padx=6)

        self.host_var = tk.StringVar(value="127.0.0.1")
        self.path_var = tk.StringVar(value="/onebot/v11/ws")
        self.forward_url_var = tk.StringVar(value="ws://127.0.0.1:3001")
        for row, label, variable in [(5, "反向监听地址:", self.host_var),
                                     (6, "反向 WebSocket 路径:", self.path_var),
                                     (7, "正向 WebSocket 地址:", self.forward_url_var)]:
            ttk.Label(sec2, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
            ttk.Entry(sec2, textvariable=variable).grid(row=row, column=1, sticky=tk.EW, padx=6, pady=4)

        # Section 3: Performance & Storage
        sec3 = ttk.LabelFrame(scroll_frame, text=" 系统性能与存储配置 ", padding=12)
        sec3.pack(fill=tk.X, pady=(0, 12))
        sec3.columnconfigure(1, weight=1)

        ttk.Label(sec3, text="全局最大并发数:", style="FieldLabel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.concurrency_var = tk.IntVar(value=2)
        ttk.Spinbox(sec3, from_=1, to=10, textvariable=self.concurrency_var, width=6).grid(row=0, column=1, sticky=tk.W, pady=4, padx=6)

        ttk.Label(sec3, text="SQLite 数据文件:", style="FieldLabel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.db_path_var = tk.StringVar(value="data/assistant.db")
        ttk.Entry(sec3, textvariable=self.db_path_var).grid(row=1, column=1, sticky=tk.EW, pady=4, padx=6)

        # Section 4: Dynamic Session Memory & Context Compression
        sec4 = ttk.LabelFrame(scroll_frame, text=" 动态会话记忆与智能上下文压缩 ", padding=12)
        sec4.pack(fill=tk.X, pady=(0, 12))
        sec4.columnconfigure(1, weight=1)

        ttk.Label(sec4, text="连续会话超时判定:", style="FieldLabel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        t_box = ttk.Frame(sec4)
        t_box.grid(row=0, column=1, sticky=tk.W, pady=4, padx=6)
        self.session_timeout_var = tk.IntVar(value=600)
        ttk.Spinbox(t_box, from_=60, to=3600, increment=60, textvariable=self.session_timeout_var, width=8).pack(side=tk.LEFT)
        ttk.Label(t_box, text=" 秒 (默认 600 秒 = 10 分钟，在此间隔内连续互动保持完整会话)").pack(side=tk.LEFT, padx=4)

        ttk.Label(sec4, text="触发自动压缩轮数:", style="FieldLabel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=4)
        th_box = ttk.Frame(sec4)
        th_box.grid(row=1, column=1, sticky=tk.W, pady=4, padx=6)
        self.session_thresh_var = tk.IntVar(value=10)
        ttk.Spinbox(th_box, from_=5, to=30, textvariable=self.session_thresh_var, width=8).pack(side=tk.LEFT)
        ttk.Label(th_box, text=" 轮 (达到此上限自动触发 gemini-3.5-flash-lite 后台提炼高密度摘要)").pack(side=tk.LEFT, padx=4)

        ttk.Label(sec4, text="压缩后保留近期对话:", style="FieldLabel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4)
        k_box = ttk.Frame(sec4)
        k_box.grid(row=2, column=1, sticky=tk.W, pady=4, padx=6)
        self.session_keep_var = tk.IntVar(value=4)
        ttk.Spinbox(k_box, from_=2, to=10, textvariable=self.session_keep_var, width=8).pack(side=tk.LEFT)
        ttk.Label(k_box, text=" 轮 (更早的历史轮次将合并提炼为背景摘要，最近这几轮原样保留)").pack(side=tk.LEFT, padx=4)

        # Action Buttons
        btn_bar = ttk.Frame(scroll_frame)
        btn_bar.pack(fill=tk.X, pady=10)

        save_btn = tk.Button(
            btn_bar,
            text="💾 保存配置至 .env",
            command=self.save_ui_to_env,
            bg="#5271ff",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2"
        )
        save_btn.pack(side=tk.LEFT, padx=(0, 10))

        reload_btn = ttk.Button(
            btn_bar,
            text="🔄 重新从 .env 加载",
            command=self.load_env_to_ui
        )
        reload_btn.pack(side=tk.LEFT)

    def _toggle_key_visibility(self) -> None:
        if self.show_key_var.get():
            self.key_entry.configure(show="")
        else:
            self.key_entry.configure(show="*")

    # ============================================================
    # TAB 2: Live Logs
    # ============================================================
    def _build_logs_tab(self) -> None:
        log_tool_bar = ttk.Frame(self.tab_logs)
        log_tool_bar.pack(fill=tk.X, pady=(0, 8))

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_tool_bar, text="自动滚至底部", variable=self.auto_scroll_var).pack(side=tk.LEFT)

        clear_btn = ttk.Button(log_tool_bar, text="清空日志", command=self.clear_logs)
        clear_btn.pack(side=tk.RIGHT, padx=4)

        copy_btn = ttk.Button(log_tool_bar, text="复制全部日志", command=self.copy_logs)
        copy_btn.pack(side=tk.RIGHT, padx=4)

        # ScrolledText for logs
        self.log_area = scrolledtext.ScrolledText(
            self.tab_logs,
            wrap=tk.WORD,
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#ffffff",
            font=("Consolas", 10),
            padx=10,
            pady=10
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # Configure color tags
        self.log_area.tag_config("INFO", foreground="#38bdf8")
        self.log_area.tag_config("WARNING", foreground="#facc15")
        self.log_area.tag_config("ERROR", foreground="#f87171")
        self.log_area.tag_config("SUCCESS", foreground="#4ade80")

    def clear_logs(self) -> None:
        self.log_area.delete("1.0", tk.END)

    def copy_logs(self) -> None:
        text = self.log_area.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("提示", "日志内容已复制到系统剪贴板！")

    def append_log(self, text: str) -> None:
        tag = None
        if "[ERROR]" in text or "Traceback" in text or "Exception" in text:
            tag = "ERROR"
        elif "[WARNING]" in text:
            tag = "WARNING"
        elif "[INFO]" in text:
            tag = "INFO"
        elif "成功" in text or "PASSED" in text:
            tag = "SUCCESS"

        for secret in (self.key_var.get().strip(), self.token_var.get().strip()):
            if secret:
                text = text.replace(secret, "[已隐藏]")
        self.log_area.insert(tk.END, text + "\n", tag)
        lines = int(self.log_area.index("end-1c").split(".")[0])
        if lines > 5000:
            self.log_area.delete("1.0", f"{lines - 5000}.0")
        if self.auto_scroll_var.get():
            self.log_area.see(tk.END)

    # ============================================================
    # TAB 3: Usage Stats & Costs
    # ============================================================
    def _build_stats_tab(self) -> None:
        stat_bar = ttk.Frame(self.tab_stats)
        stat_bar.pack(fill=tk.X, pady=(0, 14))

        refresh_btn = ttk.Button(stat_bar, text="🔄 刷新最新统计", command=self.refresh_stats)
        refresh_btn.pack(side=tk.LEFT)

        # Metric Cards Frame
        cards_frame = ttk.Frame(self.tab_stats)
        cards_frame.pack(fill=tk.X, pady=6)

        self.stat_requests = self._create_metric_card(cards_frame, "累计请求数", "0 次", 0)
        self.stat_tokens = self._create_metric_card(cards_frame, "总计 Token 消耗", "0", 1)
        self.stat_searches = self._create_metric_card(cards_frame, "Google 搜索次数", "0 次", 2)
        self.stat_cost = self._create_metric_card(cards_frame, "估算花费 (USD/CNY)", "$0.00 / ¥0.00", 3)

        # Detailed breakdown
        details_frame = ttk.LabelFrame(self.tab_stats, text=" 数据库明细概览 ", padding=12)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=12)

        self.stats_detail_text = scrolledtext.ScrolledText(
            details_frame,
            font=("Consolas", 10),
            bg="#f8fafc",
            fg="#1e293b",
            height=12
        )
        self.stats_detail_text.pack(fill=tk.BOTH, expand=True)

        # Initial refresh
        self.refresh_stats()

    def _create_metric_card(self, parent: ttk.Frame, title: str, init_val: str, col: int) -> ttk.Label:
        card = ttk.Frame(parent, padding=12, style="Card.TFrame")
        card.grid(row=0, column=col, padx=6, pady=4, sticky=tk.NSEW)
        parent.columnconfigure(col, weight=1)

        ttk.Label(card, text=title, style="StatLabel.TLabel").pack(anchor=tk.W)
        num_lbl = ttk.Label(card, text=init_val, style="StatNum.TLabel")
        num_lbl.pack(anchor=tk.W, pady=(4, 0))
        return num_lbl

    def refresh_stats(self) -> None:
        db_path = PROJECT_ROOT / self.db_path_var.get()
        if not db_path.exists():
            self.stats_detail_text.delete("1.0", tk.END)
            self.stats_detail_text.insert(tk.END, "数据库文件尚不存在（首次启动服务处理消息后将自动创建）。")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # Global usage
            cur.execute("""
                SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(candidate_tokens), 0),
                       COALESCE(SUM(total_tokens), 0), COALESCE(SUM(search_queries), 0)
                FROM usage_stats;
            """)
            row = cur.fetchone() or (0, 0, 0, 0, 0)
            reqs, p_toks, c_toks, tot_toks, searches = row

            # Message count
            cur.execute("SELECT COUNT(*) FROM messages;")
            msg_count = cur.fetchone()[0]

            # Price calculations
            cost_usd = (p_toks * 0.000000075) + (c_toks * 0.00000030)
            cost_cny = cost_usd * 7.25

            self.stat_requests.configure(text=f"{reqs:,} 次")
            self.stat_tokens.configure(text=f"{tot_toks:,}")
            self.stat_searches.configure(text=f"{searches:,} 次")
            self.stat_cost.configure(text=f"${cost_usd:.4f} / ¥{cost_cny:.3f}")

            # Top active users
            cur.execute("""
                SELECT user_id, COUNT(*), SUM(total_tokens), SUM(search_queries)
                FROM usage_stats
                GROUP BY user_id
                ORDER BY COUNT(*) DESC
                LIMIT 10;
            """)
            user_rows = cur.fetchall()

            detail = []
            detail.append("【系统全局概览】")
            detail.append(f"• 数据库存储路径: {db_path}")
            detail.append(f"• 历史消息总条数: {msg_count} 条")
            detail.append(f"• API 请求调用次数: {reqs} 次")
            detail.append(f"• 输入 Prompt Tokens: {p_toks:,}")
            detail.append(f"• 输出 Candidate Tokens: {c_toks:,}")
            detail.append(f"• 合计 Token 消耗: {tot_toks:,}")
            detail.append(f"• Google 检索触发: {searches} 次")
            detail.append(f"• 估算 API 账单花费: ${cost_usd:.6f} USD (约 {cost_cny:.4f} 元人民币)")
            detail.append("  (提示: 若当前使用 Google 官方免费层 Unpaid Tier，实际计费为 0 元)\n")

            detail.append("【活跃用户用量排行】")
            detail.append(f"{'QQ 号码':<15}{'请求次数':<12}{'累计 Token':<16}{'搜索次数':<10}")
            detail.append("-" * 55)
            for u_id, u_req, u_tok, u_search in user_rows:
                detail.append(f"{str(u_id):<15}{str(u_req):<12}{f'{u_tok:,}':<16}{str(u_search):<10}")

            conn.close()

            self.stats_detail_text.delete("1.0", tk.END)
            self.stats_detail_text.insert(tk.END, "\n".join(detail))

        except Exception as e:
            self.stats_detail_text.delete("1.0", tk.END)
            self.stats_detail_text.insert(tk.END, f"读取统计数据出现异常: {e}")

    # ============================================================
    # TAB 4: Diagnostics & Tools
    # ============================================================
    def _build_tools_tab(self) -> None:
        tool_frame = ttk.Frame(self.tab_tools)
        tool_frame.pack(fill=tk.BOTH, expand=True)

        # Card 1: API Test
        card1 = ttk.LabelFrame(tool_frame, text=" 1. Gemini API 官方接口连通性验证 ", padding=12)
        card1.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(card1, text="向 Google 官方接口发送轻量测试请求，快速验证密钥有效性、网络连接及响应时延。").pack(anchor=tk.W, pady=(0, 6))
        self.api_test_btn = ttk.Button(card1, text="⚡ 开始连通性测试", command=self.test_gemini_api)
        self.api_test_btn.pack(side=tk.LEFT)

        # Card 2: Pytest Suite
        card2 = ttk.LabelFrame(tool_frame, text=" 2. 自动化测试套件 (Pytest) ", padding=12)
        card2.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(card2, text="运行包括多用户隔离、并发无串台、三级自适应裁切采样、高解析图片分析、白名单与 WebSocket 在内的 20 项全套自动化测试。").pack(anchor=tk.W, pady=(0, 6))
        self.pytest_btn = ttk.Button(card2, text="🧪 运行全套自动化测试", command=self.run_pytest_suite)
        self.pytest_btn.pack(side=tk.LEFT)

        # Card 3: Interactive Console
        card3 = ttk.LabelFrame(tool_frame, text=" 3. 命令行交互模拟终端 (Console Mock) ", padding=12)
        card3.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(card3, text="无需启动 NapCat 或扫码登录 QQ，直接在终端中模拟 QQ 用户发送消息与指令（支持 /help, //成本 等）。").pack(anchor=tk.W, pady=(0, 6))
        console_btn = ttk.Button(card3, text="💬 启动模拟控制台终端", command=self.open_console_window)
        console_btn.pack(side=tk.LEFT)

    def test_gemini_api(self) -> None:
        api_key = self.key_var.get().strip()
        model_name = self.model_var.get().strip()

        if not api_key:
            messagebox.showwarning("提示", "请先在【参数配置】标签中填写 Gemini API Key！")
            return

        self.api_test_btn.configure(state=tk.DISABLED)
        self.append_log(f"\n[INFO] 正在向 Gemini 官方接口发起轻量极速连通性测试 (Model: {model_name})...")

        def _worker() -> None:
            t0 = time.time()
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)

                # Use LOW thinking level and low token count to minimize latency and save quota
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                    max_output_tokens=32,
                    temperature=0.2,
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents="请简短回复：API连接正常",
                    config=config,
                )
                cost_ms = int((time.time() - t0) * 1000)
                reply = response.text.strip() if response.text else "[成功收到响应]"

                self.root.after(0, lambda: self._on_api_test_success(reply, cost_ms))
            except Exception as e:
                self.root.after(0, lambda error=str(e): self._on_api_test_failed(error))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_api_test_success(self, reply: str, cost_ms: int) -> None:
        self.api_test_btn.configure(state=tk.NORMAL)
        msg = f"[SUCCESS] Gemini API 连通测试通过！耗时: {cost_ms} ms | 响应: {reply}"
        self.append_log(msg)
        messagebox.showinfo(
            "测试成功",
            f"Gemini API 响应正常！\n\n• 往返时延: {cost_ms} 毫秒\n• 模型回复: {reply}"
        )

    def _on_api_test_failed(self, error: str) -> None:
        self.api_test_btn.configure(state=tk.NORMAL)
        msg = f"[ERROR] Gemini API 测试失败: {error}"
        self.append_log(msg)

        # Provide friendly diagnosis for common causes
        if "429" in error or "ResourceExhausted" in error:
            tip = (
                "【触发 Google 免费层频率限制 (429 Quota Exceeded)】\n\n"
                "原因分析：\n"
                "1. Google AI Studio 免费层账号有极严格的每分钟请求限制 (RPM 限制，通常为 5~15 次/分)。\n"
                "2. 连续快速点击测试或高并发会迅速耗尽这一分钟内的配额。\n\n"
                "解决建议：\n"
                "• 请等待 30~60 秒，让配额周期刷新后再试；\n"
                "• 正式使用时，ChatManager 已内置全局并发限制与指数退避，可有效平滑请求。"
            )
        elif "403" in error or "API_KEY_INVALID" in error:
            tip = (
                "【API 密钥无效或未授权 (403)】\n\n"
                "请检查输入的 Gemini API Key 是否完整无误，或在 Google AI Studio 重新生成一个。"
            )
        elif "404" in error or "NOT_FOUND" in error:
            tip = (
                "【模型名称不存在 (404)】\n\n"
                "当前填写的模型名称在 Google API 库中未找到。建议下拉选择 gemini-3.8-flash 或 gemini-3.5-flash-lite 测试。"
            )
        else:
            tip = f"连通测试未通过，详细错误如下：\n\n{error}"

        messagebox.showerror("测试未通过", tip)

    def run_pytest_suite(self) -> None:
        self.pytest_btn.configure(state=tk.DISABLED)
        self.append_log("\n[INFO] 正在执行全套自动化测试 (Pytest)...")

        def _worker() -> None:
            cmd = [str(VENV_PYTHON), "-m", "pytest", "tests", "-v"]
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            )
            for line in p.stdout:
                clean_line = line.rstrip()
                self.root.after(0, lambda l=clean_line: self.append_log(l))
            p.wait()

            def _done(code: int) -> None:
                self.pytest_btn.configure(state=tk.NORMAL)
                if code == 0:
                    self.append_log("[SUCCESS] 所有自动化测试用例通过 (Exit Code 0)！")
                    messagebox.showinfo("测试通过", "恭喜！全套自动化测试全部通过，详情请查看日志。")
                else:
                    self.append_log(f"[ERROR] 存在测试用例失败 (Exit Code {code})")
                    messagebox.showwarning("测试未全过", "部分测试用例未通过，请在【运行日志】查看详细 Traceback。")

            self.root.after(0, lambda: _done(p.returncode))

        threading.Thread(target=_worker, daemon=True).start()

    def open_console_window(self) -> None:
        cmd = f'start cmd /k ""{VENV_PYTHON}" main.py --console --mock"'
        os.system(cmd)
        self.append_log("[INFO] 已在独立 CMD 窗口中启动模拟交互终端。")

    # ============================================================
    # Status Bar
    # ============================================================
    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, relief="sunken", padding=(8, 4))
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.sb_left = ttk.Label(
            bar,
            text=f"Python: {VENV_PYTHON.name} | 项目目录: {PROJECT_ROOT.name}",
            font=("Segoe UI", 8),
            foreground="#64748b"
        )
        self.sb_left.pack(side=tk.LEFT)

        self.sb_right = ttk.Label(
            bar,
            text="尚未验证 QQ 连接",
            font=("Segoe UI", 8),
            foreground="#64748b"
        )
        self.sb_right.pack(side=tk.RIGHT)

    # ============================================================
    # Configuration Load / Save (.env)
    # ============================================================
    def load_env_to_ui(self) -> None:
        vals = dotenv.dotenv_values(str(ENV_FILE)) if ENV_FILE.exists() else {}

        self.key_var.set(vals.get("GEMINI_API_KEY", ""))
        self.model_var.set(vals.get("GEMINI_MODEL", "gemini-3.8-flash"))
        self.thinking_var.set(vals.get("DEFAULT_THINKING_LEVEL", "LOW").upper())

        try:
            self.ctx_var.set(int(vals.get("DEFAULT_CONTEXT_MESSAGES", "4")))
        except (ValueError, TypeError):
            self.ctx_var.set(4)

        try:
            self.tokens_var.set(int(vals.get("MAX_OUTPUT_TOKENS", "2048")))
        except (ValueError, TypeError):
            self.tokens_var.set(2048)

        search_val = vals.get("ENABLE_SEARCH_DEFAULT", "true").lower() in ("true", "1", "yes")
        self.search_var.set(search_val)

        self.whitelist_var.set(vals.get("QQ_WHITELIST", ""))
        self.mode_var.set(vals.get("ONEBOT_MODE") or "reverse_ws")
        self.host_var.set(vals.get("REVERSE_WS_HOST") or "127.0.0.1")
        self.path_var.set(vals.get("REVERSE_WS_PATH") or "/onebot/v11/ws")
        self.forward_url_var.set(vals.get("FORWARD_WS_URL") or "ws://127.0.0.1:3001")

        try:
            self.port_var.set(int(vals.get("REVERSE_WS_PORT", "8080")))
        except (ValueError, TypeError):
            self.port_var.set(8080)

        self.token_var.set(vals.get("ONEBOT_ACCESS_TOKEN", ""))

        try:
            self.concurrency_var.set(int(vals.get("MAX_GEMINI_CONCURRENCY", "2")))
        except (ValueError, TypeError):
            self.concurrency_var.set(2)

        self.db_path_var.set(vals.get("SQLITE_DB_PATH", "data/assistant.db"))

        try:
            self.session_timeout_var.set(int(float(vals.get("SESSION_TIMEOUT_SECONDS", "600"))))
        except (ValueError, TypeError):
            self.session_timeout_var.set(600)

        try:
            self.session_thresh_var.set(int(vals.get("SESSION_COMPRESSION_THRESHOLD_ROUNDS", "10")))
        except (ValueError, TypeError):
            self.session_thresh_var.set(10)

        try:
            self.session_keep_var.set(int(vals.get("SESSION_KEEP_RECENT_ROUNDS", "4")))
        except (ValueError, TypeError):
            self.session_keep_var.set(4)

        # Load system prompt
        prompt_file = PROJECT_ROOT / "prompts" / "system_prompt.txt"
        if prompt_file.exists():
            try:
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert(tk.END, prompt_file.read_text(encoding="utf-8").strip())
            except Exception:
                pass

        self.append_log("[INFO] 已从 .env 及提示词文件成功载入配置。")

    def _collect_config(self) -> dict[str, str]:
        values = {
            "GEMINI_API_KEY": self.key_var.get().strip(),
            "GEMINI_MODEL": self.model_var.get().strip(),
            "DEFAULT_THINKING_LEVEL": self.thinking_var.get().upper(),
            "ENABLE_SEARCH_DEFAULT": "true" if self.search_var.get() else "false",
            "QQ_WHITELIST": self.whitelist_var.get().strip(),
            "ONEBOT_MODE": self.mode_var.get(),
            "ONEBOT_ACCESS_TOKEN": self.token_var.get().strip(),
            "SQLITE_DB_PATH": self.db_path_var.get().strip(),
            "REVERSE_WS_HOST": self.host_var.get().strip(),
            "REVERSE_WS_PATH": self.path_var.get().strip(),
            "FORWARD_WS_URL": self.forward_url_var.get().strip(),
        }
        for key, variable, low, high, label in [
            ("DEFAULT_CONTEXT_MESSAGES", self.ctx_var, 0, 50, "历史消息条数"),
            ("MAX_OUTPUT_TOKENS", self.tokens_var, 100, 8192, "最大输出 Token"),
            ("REVERSE_WS_PORT", self.port_var, 1, 65535, "监听端口"),
            ("MAX_GEMINI_CONCURRENCY", self.concurrency_var, 1, 10, "并发数"),
            ("SESSION_TIMEOUT_SECONDS", self.session_timeout_var, 1, 86400, "会话超时秒数"),
            ("SESSION_COMPRESSION_THRESHOLD_ROUNDS", self.session_thresh_var, 2, 100, "压缩阈值"),
            ("SESSION_KEEP_RECENT_ROUNDS", self.session_keep_var, 1, 99, "保留轮数")]:
            try:
                number = variable.get()
            except (tk.TclError, ValueError):
                raise ValueError(f"{label}必须填写整数。") from None
            if not low <= number <= high:
                raise ValueError(f"{label}应在 {low}～{high} 之间。")
            values[key] = str(number)
        if int(values["SESSION_KEEP_RECENT_ROUNDS"]) >= int(values["SESSION_COMPRESSION_THRESHOLD_ROUNDS"]):
            raise ValueError("压缩后保留轮数必须小于触发压缩轮数。")
        if values["DEFAULT_THINKING_LEVEL"] not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError("思考强度请选择 LOW、MEDIUM 或 HIGH。")
        if not values["GEMINI_MODEL"] or not values["SQLITE_DB_PATH"]:
            raise ValueError("模型名称和数据库路径不能为空。")
        if values["QQ_WHITELIST"] and not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", values["QQ_WHITELIST"]):
            raise ValueError("QQ 白名单请填写数字账号，多个账号以英文逗号分隔。")
        if values["ONEBOT_MODE"] == "reverse_ws":
            if not values["REVERSE_WS_HOST"] or not values["REVERSE_WS_PATH"].startswith("/"):
                raise ValueError("请填写监听地址，反向连接路径必须以 / 开头。")
        elif values["ONEBOT_MODE"] == "forward_ws":
            from urllib.parse import urlsplit
            url = urlsplit(values["FORWARD_WS_URL"])
            if url.scheme not in ("ws", "wss") or not url.hostname or url.username or url.password:
                raise ValueError("正向地址应为 ws:// 或 wss:// 开头的 WebSocket 地址。")
            _ = url.port
        else:
            raise ValueError("请选择有效的通信模式。")
        return values

    def save_ui_to_env(self, notify: bool = True) -> bool:
        try:
            values = self._collect_config()
            # Validate everything before replacing the file, preserving unrelated keys/comments.
            temporary = ENV_FILE.with_name(".env.launcher.tmp")
            try:
                temporary.write_text(ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else "", encoding="utf-8")
                for key, value in values.items():
                    dotenv.set_key(str(temporary), key, value)
                temporary.replace(ENV_FILE)
            finally:
                temporary.unlink(missing_ok=True)
            prompt_file = PROJECT_ROOT / "prompts" / "system_prompt.txt"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(self.prompt_text.get("1.0", tk.END).strip(), encoding="utf-8")
        except (ValueError, tk.TclError, OSError) as error:
            messagebox.showerror("配置未保存", str(error))
            return False
        self.append_log("[INFO] 配置已保存，下次启动服务时生效。")
        self._refresh_overview()
        if notify:
            messagebox.showinfo("保存成功", "配置已保存，下次启动服务时生效。")
        return True

    def start_service(self) -> None:
        if self.is_running:
            return

        # Auto save settings first
        if not self.key_var.get().strip() or not self.whitelist_var.get().strip():
            messagebox.showwarning("启动前检查", "请填写 Gemini API Key 和 QQ 白名单后再启动。")
            self.notebook.select(self.tab_config)
            return
        if not self.save_ui_to_env(notify=False):
            self.notebook.select(self.tab_config)
            return

        self.append_log("\n" + "=" * 50)
        self.append_log(f"[INFO] 正在启动 QQ Gemini 智能助手服务...")

        cmd = [str(VENV_PYTHON), "main.py"]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            )
            self.is_running = True
            self.started_at = time.monotonic()

            # Reader thread for streaming logs
            def _log_reader(proc: subprocess.Popen, q: queue.Queue) -> None:
                for line in proc.stdout:
                    q.put(line.rstrip())
                proc.stdout.close()

            t = threading.Thread(target=_log_reader, args=(self.process, self.log_queue), daemon=True)
            t.start()

            self.status_lbl.configure(text=f"进程运行中 · PID {self.process.pid} · QQ 连接请查看日志")
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.notebook.select(self.tab_logs)

        except Exception as e:
            self.append_log(f"[ERROR] 启动进程失败: {e}")
            messagebox.showerror("启动失败", f"无法启动服务进程: {e}")

    def stop_service(self) -> None:
        if not self.process or self.stopping:
            return
        self.stopping = True
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_lbl.configure(text="正在停止服务…")
        self.append_log("[INFO] 正在停止服务进程…")
        try:
            self.process.terminate()
        except OSError as error:
            self.append_log(f"[WARNING] 停止请求: {error}")
        self.stop_deadline = time.monotonic() + 2
        self.root.after(50, self._finish_stop)

    def _finish_stop(self) -> None:
        if self.process and self.process.poll() is None:
            if time.monotonic() >= self.stop_deadline:
                try:
                    self.process.kill()
                except OSError:
                    pass
            self.root.after(100, self._finish_stop)
            return
        self.is_running = False
        self.stopping = False
        self.process = None
        self.started_at = None
        self.status_lbl.configure(text="服务已停止")
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.append_log("[INFO] 服务进程已停止。")
        self._refresh_overview()
        if self.close_after_stop:
            self.root.destroy()

    def _process_log_queue(self) -> None:
        for _ in range(min(self.log_queue.qsize(), 200)):
            line = self.log_queue.get_nowait()
            self.append_log(line)
        self.root.after(100, self._process_log_queue)

    def _poll_service_status(self) -> None:
        if self.is_running and self.process and not self.stopping:
            ret = self.process.poll()
            if ret is not None:
                # Process exited unexpectedly
                self.append_log(f"[WARNING] 服务进程已退出 (Exit code: {ret})")
                self.started_at = None
                self.is_running = False
                self.process = None
                self.status_lbl.configure(text="服务未启动 · 请先检查配置")
                self.start_btn.configure(state=tk.NORMAL)
                self.stop_btn.configure(state=tk.DISABLED)

        self._refresh_overview()
        self.root.after(1000, self._poll_service_status)

    def on_closing(self) -> None:
        if self.is_running and self.process:
            if messagebox.askyesno("确认退出", "智能助手服务正在后台运行中。\n退出本启动器将同时停止服务，是否确认？"):
                self.close_after_stop = True
                self.stop_service()
        else:
            self.root.destroy()


def run_gui() -> None:
    # Set Windows taskbar AppUserModelID for proper icon and grouping
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("QQGeminiAssistant.Launcher.1.0")
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    app = AssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
