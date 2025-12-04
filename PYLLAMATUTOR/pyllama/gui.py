"""Minimal PyQt chat UI for the Python tutor agent."""
from __future__ import annotations

import sys
import html
import os
import re
from pathlib import Path
import keyword

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

from agent import TutorAgent
from system_profile import build_presets, detect_system

QUICK_PROMPTS = {
    "Sorting algorithms": [
        "Explain bubble sort to a beginner.",
        "Show Python code for merge sort with comments.",
        "What is the time complexity of quicksort?",
        "Compare bubble sort and insertion sort for small arrays.",
        "How does selection sort work? Give a Python example.",
        "When would you choose merge sort over quicksort?",
        "Explain stable vs unstable sorting with examples.",
        "Implement insertion sort in Python with comments.",
        "What is the best/worst/average case of merge sort?",
        "How does quicksort choose a pivot? Show a simple pivot strategy.",
        "Demonstrate counting sort and when it is efficient.",
        "Explain why bubble sort is usually avoided in production.",
        "Describe heap sort and its complexity.",
        "How do you sort a list of dictionaries by a key in Python?",
        "What is Timsort and why does Python use it?",
    ],
    "Data types": [
        "What are Python lists and tuples? Compare them with examples.",
        "Explain dictionaries with a small example.",
        "How do you convert a string to an int safely?",
        "Show how to check a variable's type and convert between types.",
        "When should you use a set instead of a list?",
        "Explain mutability using lists and tuples.",
        "How do you copy a list without linking to the original?",
        "Demonstrate list slicing and what it returns.",
        "How do you merge two dictionaries in Python?",
        "What is the difference between None, 0, and an empty string?",
        "Explain how to store nested data (dicts of lists, etc.).",
        "Show how to iterate over key/value pairs in a dictionary.",
        "Explain how to use enums or constants for fixed values.",
        "What is the difference between bytes and str in Python?",
        "How do you check if a value exists in a list or dict?",
    ],
    "Beginner/noob": [
        "How do I print 'Hello, world!' in Python?",
        "What is a function and how do I define one?",
        "Explain if/else with a simple code example.",
        "How do I get input from a user and print it back?",
        "Show a for loop that sums numbers from 1 to 10.",
        "What is a variable and how do I name it well?",
        "Explain indentation rules in Python with a tiny example.",
        "Show how to handle a ZeroDivisionError safely.",
        "How do I use a while loop to repeat until a condition changes?",
        "Explain the difference between = and == in Python.",
        "How do I comment my code? Give single-line and multi-line examples.",
        "Show how to write and call a function with parameters.",
        "How do I return a value from a function and print it?",
        "Explain lists and how to append/remove items.",
        "What is a module and how do I import math to use sqrt?",
    ],
}


class AskWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, float)
    error = QtCore.pyqtSignal(str)

    def __init__(self, agent: TutorAgent, user_text: str) -> None:
        super().__init__()
        self.agent = agent
        self.user_text = user_text

    @QtCore.pyqtSlot()
    def run(self) -> None:
        import time

        start = time.perf_counter()
        try:
            resp = self.agent.ask(self.user_text)
            elapsed = time.perf_counter() - start
            self.finished.emit(resp, elapsed)
        except Exception as exc:  # pragma: no cover - worker error path
            self.error.emit(str(exc))


class RescanWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict, bool)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            profile, ok = detect_system()
            presets = build_presets(profile, ok)
            self.finished.emit({"profile": profile, "presets": presets}, ok)
        except Exception:
            self.finished.emit({}, False)


class WarmupWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, agent: TutorAgent) -> None:
        super().__init__()
        self.agent = agent

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.agent.client.ensure_ready()
            self.finished.emit(True, "Model ready.")
        except Exception as exc:  # pragma: no cover - worker error path
            self.finished.emit(False, str(exc))



class ChatWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        profile=None,
        profile_ok: bool = False,
        presets=None,
        agent: TutorAgent = None,
        history_data=None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("")
        self.resize(680, 620)
        self.setMinimumSize(680, 520)
        self.setMaximumSize(1920, 1080)
        self.setFont(QtGui.QFont("Helvetica", 11))

        icon_path = Path(__file__).parent / "images" / "AppIcon.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self.agent = agent or TutorAgent()

        self.tabs = QtWidgets.QTabWidget()
        chat_tab = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_tab)
        eval_tab = QtWidgets.QWidget()
        eval_layout = QtWidgets.QVBoxLayout(eval_tab)

        # Logo (top of window, outside tabs)
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        logo_path = Path(__file__).parent / "images" / "LogoLight.png"
        if logo_path.exists():
            pix = QtGui.QPixmap(str(logo_path))
            self.logo_label.setPixmap(pix.scaledToWidth(int(160 * 1.3), QtCore.Qt.SmoothTransformation))

        self.history = QtWidgets.QTextEdit()
        self.history.setReadOnly(True)
        self.history.setMinimumHeight(400)
        self.history.setAcceptRichText(True)
        self.history_font_size = 11
        self.history.setFont(QtGui.QFont("Helvetica", self.history_font_size))
        if history_data and len(history_data) > 0:
            self.history_data = history_data
        else:
            self.history_data = self.load_history_data()
        self.session_id = QtCore.QDateTime.currentDateTime().toString("yyyyMMddHHmmss")
        self.last_user_message = ""

        # Preset detection (from passed data or local scan)
        if profile is None or presets is None:
            self.profile, self.profile_ok = detect_system()
            self.presets = build_presets(self.profile, self.profile_ok)
        else:
            self.profile = profile
            self.profile_ok = profile_ok
            self.presets = presets
        # choose default preset: balanced if available
        self.current_preset = "balanced" if "balanced" in self.presets else next(iter(self.presets.keys()))
        recommended = self.presets[self.current_preset]

        model_label = QtWidgets.QLabel(f"Model: {self.agent.client._active_model or 'initializing…'}")
        model_label.setAlignment(QtCore.Qt.AlignLeft)
        self.model_label = model_label

        send_button = QtWidgets.QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        self.send_button = send_button
        self.send_on_enter = QtWidgets.QCheckBox("Send on Enter")
        self.send_on_enter.setChecked(False)
        # Toggle Enter behavior
        self.send_on_enter.stateChanged.connect(self.update_shortcut)

        self.input = QtWidgets.QPlainTextEdit()
        self.input.setMinimumHeight(100)
        self.input.setMaximumHeight(160)
        self.input.setPlaceholderText("Ask a Python question or paste code…")
        self.input.installEventFilter(self)
        self.input.setFont(QtGui.QFont("Helvetica", 11))

        # Zoom controls for chat history
        self.zoom_out_btn = QtWidgets.QToolButton()
        self.zoom_out_btn.setText("−")
        self.zoom_in_btn = QtWidgets.QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_out_btn.setAutoRaise(True)
        self.zoom_in_btn.setAutoRaise(True)
        self.zoom_out_btn.clicked.connect(lambda: self.adjust_history_font(-1))
        self.zoom_in_btn.clicked.connect(lambda: self.adjust_history_font(1))

        self.usage_label = QtWidgets.QLabel("Session tokens: 0 | Approx cost: $0.0000")
        self.usage_label.setAlignment(QtCore.Qt.AlignRight)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.time_label = QtWidgets.QLabel("Last response: -- s")
        self.time_label.setAlignment(QtCore.Qt.AlignLeft)
        self.token_label = QtWidgets.QLabel("Prompt/Completion tokens: -- / --")
        self.token_label.setAlignment(QtCore.Qt.AlignLeft)

        self.loading = QtWidgets.QProgressBar()
        self.loading.setRange(0, 0)
        self.loading.setVisible(False)
        self.rescan_progress = QtWidgets.QProgressBar()
        self.rescan_progress.setRange(0, 0)
        self.rescan_progress.setVisible(False)

        chat_layout.addWidget(model_label)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(model_label)
        top_row.addStretch()
        top_row.addWidget(self.zoom_out_btn)
        top_row.addWidget(self.zoom_in_btn)
        chat_layout.addLayout(top_row)

        # Wrap history in a styled container to add spacing and radius
        # Plain history area with spacing
        chat_layout.addWidget(self.history)
        chat_layout.addSpacing(16)

        input_box = QtWidgets.QVBoxLayout()
        input_box.addWidget(self.input)
        send_row = QtWidgets.QHBoxLayout()
        send_row.addWidget(self.send_on_enter)
        send_row.addStretch()
        send_row.addWidget(send_button)
        input_box.addLayout(send_row)
        chat_layout.addLayout(input_box)

        qp_button = QtWidgets.QPushButton("Quick Prompts")
        qp_button.clicked.connect(self.show_quick_prompts)
        chat_layout.addWidget(qp_button)

        chat_layout.addWidget(self.loading)
        info_bar = QtWidgets.QHBoxLayout()
        info_bar.addWidget(self.info_label)
        info_bar.addWidget(self.time_label)
        info_bar.addWidget(self.token_label)
        info_bar.addStretch()
        info_bar.addWidget(self.usage_label)
        chat_layout.addLayout(info_bar)

        # Config side panel content
        form = QtWidgets.QVBoxLayout()
        form.setContentsMargins(6, 6, 6, 6)
        header_color = "#1c6be2"  # system-like blue
        preset_group = QtWidgets.QGroupBox("Presets")
        preset_group.setStyleSheet(f"QGroupBox{{color:{header_color}; font-weight:bold; font-size:12pt; padding-top:4px;}}")
        preset_form = QtWidgets.QFormLayout(preset_group)
        device_group = QtWidgets.QGroupBox("Model / Device")
        device_group.setStyleSheet(f"QGroupBox{{color:{header_color}; font-weight:bold; font-size:12pt; padding-top:4px;}}")
        device_form = QtWidgets.QFormLayout(device_group)
        perf_group = QtWidgets.QGroupBox("Performance")
        perf_group.setStyleSheet(f"QGroupBox{{color:{header_color}; font-weight:bold; font-size:12pt; padding-top:4px;}}")
        perf_form = QtWidgets.QFormLayout(perf_group)
        self.max_tokens_input = QtWidgets.QSpinBox()
        self.max_tokens_input.setRange(32, 2048)
        self.max_tokens_input.setValue(int(recommended.get("max_tokens", 512)))
        self.device_select = QtWidgets.QComboBox()
        self.device_select.addItems(["auto", "cpu", "gpu"])
        self.device_select.setCurrentText(recommended.get("device", "auto"))
        self.cpu_threads_input = QtWidgets.QSpinBox()
        self.cpu_threads_input.setRange(1, 64)
        self.cpu_threads_input.setValue(int(recommended.get("threads", 1)))
        self.quant_select = QtWidgets.QComboBox()
        self.quant_select.addItems(["none", "fp8", "int4"])
        self.quant_select.setCurrentText("none")
        if getattr(self.profile, "has_mps", False) and not getattr(self.profile, "has_cuda", False):
            # Disable GPU choice if only MPS is present
            idx = self.device_select.findText("gpu")
            if idx >= 0:
                self.device_select.model().item(idx).setEnabled(False)
        quant_help = QtWidgets.QLabel(
            "Quantization: fp8 = faster on GPUs that support FP8 with small accuracy tradeoff; "
            "int4 = more aggressive compression for CPU, often faster but lower quality."
        )
        quant_help.setWordWrap(True)
        gpu_text = "CUDA" if getattr(self.profile, "has_cuda", False) else ("MPS" if getattr(self.profile, "has_mps", False) else "None")
        self.preset_status = QtWidgets.QLabel(
            f"Scan {'ok' if self.profile_ok else 'fallback'} | Cores: {self.profile.cores}, RAM: {self.profile.ram_gb:.1f} GB, GPU: {gpu_text}"
        )
        self.preset_select = QtWidgets.QComboBox()
        for name in self.presets:
            self.preset_select.addItem(name)
        self.preset_select.setCurrentText(self.current_preset)
        self.preset_select.currentTextChanged.connect(self.set_preset_fields)
        self.rescan_button = QtWidgets.QPushButton("Rescan System")
        self.rescan_button.clicked.connect(self.start_rescan)
        self.manual_mode = QtWidgets.QCheckBox("Manual mode (ignore presets)")
        self.manual_mode.stateChanged.connect(self.toggle_manual_mode)
        apply_btn = QtWidgets.QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_settings)
        self.apply_btn = apply_btn

        preset_form.addRow(self.preset_status)
        preset_form.addRow("Preset:", self.preset_select)
        preset_form.addRow(self.manual_mode)
        preset_form.addRow(self.rescan_button)
        preset_form.addRow("Scan progress:", self.rescan_progress)

        device_form.addRow("Max new tokens:", self.max_tokens_input)
        device_form.addRow("Device:", self.device_select)
        device_form.addRow("Quantization:", self.quant_select)
        device_form.addRow(quant_help)

        perf_form.addRow("CPU threads:", self.cpu_threads_input)
        cpu_help = QtWidgets.QLabel("Higher threads can speed CPU inference but may impact system responsiveness.")
        cpu_help.setWordWrap(True)
        perf_form.addRow(cpu_help)
        perf_form.addRow(apply_btn)

        # Scrollable container for config so it doesn't overflow when shown
        config_scroll = QtWidgets.QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_inner = QtWidgets.QWidget()
        form.addWidget(preset_group)
        form.addWidget(device_group)
        form.addWidget(perf_group)
        form.addStretch()
        config_inner.setLayout(form)
        config_scroll.setWidget(config_inner)

        # Evaluation tab
        self.eval_count = 0
        self.eval_total_time = 0.0
        self.eval_prompt_tokens = 0
        self.eval_completion_tokens = 0
        self.eval_summary = QtWidgets.QLabel("No evaluations yet.")
        self.eval_summary.setWordWrap(True)
        eval_layout.addWidget(self.eval_summary)
        # Recent turns table
        self.turn_table = QtWidgets.QTableWidget(0, 9)
        self.turn_table.setHorizontalHeaderLabels(
            ["Time", "Resp time (s)", "Prompt tok", "Completion tok", "Cost", "Model", "Device", "Threads", "Quant"]
        )
        self.turn_table.horizontalHeader().setStretchLastSection(True)
        self.turn_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        eval_layout.addWidget(self.turn_table)
        self.wipe_button = QtWidgets.QPushButton("Wipe saved evaluations")
        self.wipe_button.setStyleSheet("QPushButton { background:#c62828; color:white; padding:6px 10px; border-radius:4px; }"
                                       "QPushButton:hover { background:#b71c1c; }")
        self.wipe_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.wipe_button.clicked.connect(self.wipe_history)
        eval_layout.addWidget(self.wipe_button, alignment=QtCore.Qt.AlignLeft)
        self.model_info_eval = QtWidgets.QLabel("Model/device: --")
        eval_layout.addWidget(self.model_info_eval)
        self.save_status = QtWidgets.QLabel("Save status: --")
        eval_layout.addWidget(self.save_status)
        # Config side panel (toggle button)
        config_widget = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_widget)
        config_layout.setContentsMargins(8, 8, 8, 8)
        config_layout.addWidget(config_scroll, stretch=1)
        self.config_dock = QtWidgets.QDockWidget("", self)
        self.config_dock.setWidget(config_widget)
        self.config_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        self.config_dock.setTitleBarWidget(QtWidgets.QWidget())  # remove header bar
        self.config_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.config_dock.hide()
        self.config_dock.setMinimumWidth(260)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.config_dock)

        self.config_btn = QtWidgets.QPushButton("Config")
        btn_style = "QPushButton { padding:6px 10px; border:1px solid #999; border-radius:6px; } QPushButton:hover { border-color:#008cff; }"
        self.config_btn.setStyleSheet(btn_style)
        self.config_btn.clicked.connect(lambda: self.toggle_panel(self.config_dock, self.config_btn))

        # Build tabs
        eval_tab.setLayout(eval_layout)
        self.tabs.addTab(chat_tab, "Chat")
        self.tabs.addTab(eval_tab, "Evaluation")

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.config_btn, alignment=QtCore.Qt.AlignLeft)
        top_bar.addStretch()
        top_bar.addWidget(self.logo_label, alignment=QtCore.Qt.AlignCenter)
        top_bar.addStretch()
        # balance right with an invisible spacer matching config button width
        spacer = QtWidgets.QWidget()
        spacer.setFixedWidth(self.config_btn.sizeHint().width())
        top_bar.addWidget(spacer)
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.tabs)
        # Signature footer
        sig = QtWidgets.QLabel("by Anatole Dupuis")
        sig.setAlignment(QtCore.Qt.AlignLeft)
        sig.setStyleSheet("color: #555; font-size: 9pt; padding-top:4px;")
        main_layout.addWidget(sig, alignment=QtCore.Qt.AlignLeft)
        self.setCentralWidget(central)

        # Enter/shortcut handling
        self.shortcut = QtWidgets.QShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Return, self)
        self.shortcut.activated.connect(self.send_message)

        self._thread: QtCore.QThread | None = None
        self._worker: AskWorker | None = None
        self._warmup_thread: QtCore.QThread | None = None
        self._warmup_worker: WarmupWorker | None = None
        self.model_ready = False
        self._pending_counter = 0
        self.start_warmup(initial=True)
        # Restore history and eval table
        self.restore_history()

    @QtCore.pyqtSlot(int)
    def update_shortcut(self, state: int) -> None:
        if state == QtCore.Qt.Checked:
            self.shortcut.setKey(QtCore.Qt.Key_Return)
        else:
            self.shortcut.setKey(QtCore.Qt.CTRL + QtCore.Qt.Key_Return)

    def eventFilter(self, obj, event):  # pragma: no cover - GUI path
        if obj is self.input and hasattr(self, "send_on_enter") and self.send_on_enter.isChecked():
            if event.type() == QtCore.QEvent.KeyPress:
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    if not (event.modifiers() & QtCore.Qt.ControlModifier):
                        self.send_message()
                        return True
        if obj is self.logo_label and self.logo_pix is not None:
            if event.type() == QtCore.QEvent.Enter:
                self.logo_grow_start = QtCore.QTime.currentTime()
                self.logo_label.setPixmap(self.logo_pix.scaledToWidth(180, QtCore.Qt.SmoothTransformation))
            elif event.type() == QtCore.QEvent.MouseMove:
                if hasattr(self, "logo_grow_start"):
                    elapsed_ms = self.logo_grow_start.msecsTo(QtCore.QTime.currentTime())
                    factor = min(3.0, 1.0 + elapsed_ms / 600.0)  # smoother growth
                    scaled = self.logo_pix.scaledToWidth(int(160 * factor), QtCore.Qt.SmoothTransformation)
                    self.logo_label.setPixmap(scaled)
            elif event.type() == QtCore.QEvent.Leave:
                # aggressive snap back
                self.logo_label.setPixmap(self.logo_pix.scaledToWidth(160, QtCore.Qt.FastTransformation))
        return super().eventFilter(obj, event)

    def adjust_history_font(self, delta: int) -> None:
        self.history_font_size = max(8, min(18, self.history_font_size + delta))
        font = self.history.font()
        font.setPointSize(self.history_font_size)
        self.history.setFont(font)
        self.save_history(status_text="Saving…")

    def restore_history(self) -> None:
        if not self.history_data:
            return
        # Oldest first for replay; table shows newest first
        for turn in self.history_data:
            ts = turn.get("timestamp", "")
            usage = turn.get("usage", {})
            try:
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
            except Exception:
                prompt_tokens = completion_tokens = 0
            cost = float(turn.get("cost", 0.0))
            model = turn.get("model", "unknown")
            device = turn.get("device", "unknown")
            threads = turn.get("threads", "")
            quant = turn.get("quant", "none")
            elapsed = turn.get("elapsed", None)
            self.insert_eval_row(ts, prompt_tokens, completion_tokens, cost, model, device, threads, quant, session=False, elapsed=elapsed)
        self.save_status.setText("Save status: loaded history.")
        self.refresh_eval_totals()

    def save_history(self, status_text: str = "Saved.") -> None:
        try:
            os.makedirs(os.path.join(Path.home(), ".pyllama"), exist_ok=True)
            hist_path = Path.home() / ".pyllama" / "chat_history.json"
            payload = {"turns": self.history_data}
            import json

            with hist_path.open("w") as f:
                json.dump(payload, f, indent=2)
            self.save_status.setText(f"Save status: {status_text}")
        except Exception as exc:
            self.save_status.setText(f"Save status: error ({exc})")

    def load_history_data(self) -> list:
        hist_path = Path.home() / ".pyllama" / "chat_history.json"
        if not hist_path.exists():
            return []
        try:
            import json

            with hist_path.open() as f:
                data = json.load(f)
            return data.get("turns", []) if isinstance(data, dict) else []
        except Exception:
            return []

    def wipe_history(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Wipe evaluations",
            "This will remove all saved evaluation rows. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.history_data = []
        self.turn_table.setRowCount(0)
        hist_path = Path.home() / ".pyllama" / "chat_history.json"
        try:
            if hist_path.exists():
                hist_path.unlink()
            self.save_history(status_text="Cleared.")
        except Exception as exc:
            self.save_status.setText(f"Save status: error clearing ({exc})")
        self.refresh_eval_totals()

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI path
        h = event.size().height()
        if h <= 520:
            self.history.setMinimumHeight(200)
            self.input.setFixedHeight(90)
        else:
            self.history.setMinimumHeight(280)
            self.input.setFixedHeight(120)
        return super().resizeEvent(event)

    def toggle_panel(self, dock: QtWidgets.QDockWidget, button: QtWidgets.QPushButton) -> None:
        # Simple bounce animation on click
        start = button.geometry()
        up = QtCore.QRect(start.x(), start.y() - 6, start.width(), start.height())
        anim = QtCore.QPropertyAnimation(button, b"geometry", self)
        anim.setDuration(180)
        anim.setKeyValueAt(0, start)
        anim.setKeyValueAt(0.5, up)
        anim.setKeyValueAt(1, start)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        dock.setVisible(not dock.isVisible())

    def append_message(self, speaker: str, text: str, is_html: bool = False) -> None:
        is_user = speaker.lower().startswith("you")
        color = "green" if is_user else "red" if speaker.lower().startswith("tutor") else "black"
        body = text if is_html else html.escape(text).replace("\n", "<br>")
        block = f"<div style='margin:6px 0;'><b style='color:{color}'>{speaker}:</b><br>{body}</div>"
        self.history.append(block)
        self.history.verticalScrollBar().setValue(self.history.verticalScrollBar().maximum())

    def format_response(self, text: str) -> str:
        lines = []
        in_code = False
        code_lines = []

        def highlight_code_block(code: str) -> str:
            """Lightweight syntax tinting for Python-like code."""
            kw_pattern = r"\b(" + "|".join(keyword.kwlist) + r")\b"
            builtin_pattern = r"\b(len|range|print|int|str|list|dict|set|float|input|enumerate|zip|sum|min|max|open|sorted)\b"
            # comments
            code = re.sub(r"(?m)#.*$", lambda m: f"<span style='color:#888'>{html.escape(m.group(0))}</span>", code)
            # strings
            code = re.sub(
                r"([\"'])(?:(?=(\\?))\2.)*?\1",
                lambda m: f"<span style='color:#228B22'>{html.escape(m.group(0))}</span>",
                code,
            )
            # keywords (after strings/comments)
            code = re.sub(kw_pattern, lambda m: f"<span style='color:#0000CD;font-weight:bold'>{m.group(1)}</span>", code)
            # builtins
            code = re.sub(builtin_pattern, lambda m: f"<span style='color:#8A2BE2'>{m.group(1)}</span>", code)
            # numbers
            code = re.sub(r"\b\d+(?:\.\d+)?\b", lambda m: f"<span style='color:#b58900'>{m.group(0)}</span>", code)
            return code.replace(" ", "&nbsp;").replace("\t", "&nbsp;" * 4).replace("\n", "<br>")
        for line in text.splitlines():
            if line.strip().startswith("```"):
                if not in_code:
                    in_code = True
                    code_lines = []
                else:
                    # closing fence
                    raw_code = "\n".join(code_lines)
                    code_html = highlight_code_block(raw_code)
                    lines.append(
                        "<pre style='margin:4px 0; background:#f5f5f5; padding:6px; border-radius:6px;'>"
                        f"<code style='font-family:\"Menlo\",\"Courier New\", monospace; color:#111;'>{code_html}</code></pre>"
                    )
                    code_lines = []
                    in_code = False
                continue
            if in_code:
                code_lines.append(line)
                continue
            stripped = line.strip()
            if stripped.startswith("###"):
                header = stripped.lstrip("#").strip().rstrip(":")
                lines.append(f"<b>{html.escape(header)}:</b>")
            else:
                lines.append(html.escape(line))
        if in_code and code_lines:
            raw_code = "\n".join(code_lines)
            code_html = highlight_code_block(raw_code)
            lines.append(
                "<pre style='margin:4px 0; background:#f5f5f5; padding:6px; border-radius:6px;'>"
                f"<code style='font-family:\"Menlo\",\"Courier New\", monospace; color:#111;'>{code_html}</code></pre>"
            )
        return "<br>".join(lines)

    def start_warmup(self, initial: bool = False) -> None:
        """Ensure the model is initialized without blocking the UI."""
        if getattr(self.agent.client, "_initialized", False):
            self.model_ready = True
            self.model_label.setText(f"Model: {self.agent.client._active_model or 'ready'}")
            self.info_label.setText(
                f"Model: {self.agent.client._active_model or 'unknown'} | Device: {self.agent.client.active_device}"
            )
            self.send_button.setDisabled(False)
            self.input.setDisabled(False)
            return

        # prevent sending during warmup
        self.model_ready = False
        self.send_button.setDisabled(True)
        self.input.setDisabled(True)
        self.loading.setVisible(True)
        self.model_label.setText("Model: initializing…")
        self.info_label.setText("Warming up model in background…")

        if self._warmup_thread:
            self._warmup_thread.quit()
            self._warmup_thread.wait()

        self._warmup_thread = QtCore.QThread()
        self._warmup_worker = WarmupWorker(self.agent)
        self._warmup_worker.moveToThread(self._warmup_thread)
        self._warmup_thread.started.connect(self._warmup_worker.run)
        self._warmup_worker.finished.connect(self.on_warmup_done)
        self._warmup_worker.finished.connect(self._warmup_thread.quit)
        self._warmup_thread.start()

    @QtCore.pyqtSlot(bool, str)
    def on_warmup_done(self, ok: bool, msg: str) -> None:
        self.loading.setVisible(False)
        self.model_ready = ok
        if ok:
            self.model_label.setText(f"Model: {self.agent.client._active_model or 'ready'}")
            self.info_label.setText(
                f"Model: {self.agent.client._active_model or 'unknown'} | Device: {self.agent.client.active_device}"
            )
            self.send_button.setDisabled(False)
            self.input.setDisabled(False)
        else:
            self.model_label.setText("Model failed to initialize")
            self.info_label.setText(msg)
        self._warmup_thread = None
        self._warmup_worker = None
    def send_message(self) -> None:
        user_text = self.input.toPlainText().strip()
        if not user_text:
            return
        self.last_user_message = user_text
        self.append_message("You", user_text)
        self.input.clear()

        self.send_button.setDisabled(True)
        self.input.setDisabled(True)
        self.apply_btn.setDisabled(True)
        self.loading.setVisible(True)

        # Clean up any previous worker/thread before starting a new one
        if self._thread:
            self.cleanup_thread()

        self._thread = QtCore.QThread()
        self._worker = AskWorker(self.agent, user_text)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_response)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self.cleanup_thread)
        self._worker.error.connect(self.cleanup_thread)
        self._thread.start()

    @QtCore.pyqtSlot(object, float)
    def on_response(self, response, elapsed: float) -> None:
        formatted = self.format_response(response.text)
        self.append_message("Tutor", formatted, is_html=True)
        self.usage_label.setText(
            f"Session tokens: {response.session_tokens} | Approx cost: ${response.session_cost:.4f}"
        )
        if getattr(self.agent.client, "_active_model", None):
            self.model_label.setText(f"Model: {self.agent.client._active_model}")
        self.info_label.setText(
            f"Model: {self.agent.client._active_model or 'unknown'} | Device: {self.agent.client.active_device}"
        )
        self.time_label.setText(f"Last response: {elapsed:.2f}s")
        self.token_label.setText(
            f"Prompt/Completion tokens: {response.usage.prompt_tokens} / {response.usage.completion_tokens}"
        )
        if not hasattr(self, "last_user_message"):
            self.last_user_message = ""
        placeholder = "placeholder because the open-source llama model is not available" in response.text.lower()
        if not placeholder:
            self.last_response_text = response.text
            stamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            self.insert_eval_row(
                stamp,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.cost,
                self.agent.client._active_model or "unknown",
                self.agent.client.active_device,
                os.environ.get("OMP_NUM_THREADS", ""),
                os.getenv("LLAMA_QUANTIZATION", "none"),
                session=True,
                elapsed=elapsed,
            )
            self.history_data.append(
                {
                    "timestamp": stamp,
                    "user": "",
                    "tutor": "",
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                    "elapsed": elapsed,
                    "cost": response.cost,
                    "model": self.agent.client._active_model or "unknown",
                    "device": self.agent.client.active_device,
                    "threads": os.environ.get("OMP_NUM_THREADS", ""),
                    "quant": os.getenv("LLAMA_QUANTIZATION", "none"),
                    "session_id": self.session_id,
                }
            )
            self.save_history()
            self.refresh_eval_totals()
        self.send_button.setDisabled(False)
        self.input.setDisabled(False)
        self.apply_btn.setDisabled(False)
        self.loading.setVisible(False)

    @QtCore.pyqtSlot(str)
    def on_error(self, msg: str) -> None:  # pragma: no cover - error path
        self.append_message("Tutor", f"An error occurred: {msg}")
        self.send_button.setDisabled(False)
        self.input.setDisabled(False)
        self.loading.setVisible(False)

    @QtCore.pyqtSlot()
    def cleanup_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        if self._warmup_thread:
            self._warmup_thread.quit()
            self._warmup_thread.wait()
        self._warmup_thread = None
        self._warmup_worker = None
        # Save on cleanup
        self.save_history()

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI path
        self.cleanup_thread()
        super().closeEvent(event)

    def apply_settings(self) -> None:
        max_tokens = self.max_tokens_input.value()
        device_pref = self.device_select.currentText()
        cpu_threads = self.cpu_threads_input.value()
        quant_pref = self.quant_select.currentText()

        if device_pref:
            os.environ["LLAMA_DEVICE"] = device_pref
        else:
            os.environ.pop("LLAMA_DEVICE", None)

        # CPU threads hint
        try:
            import torch

            torch.set_num_threads(cpu_threads)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        if quant_pref:
            os.environ["LLAMA_QUANTIZATION"] = quant_pref
        else:
            os.environ.pop("LLAMA_QUANTIZATION", None)

        # Recreate the agent with new settings
        self.agent = TutorAgent(model=None, max_new_tokens=max_tokens, quantization=quant_pref)
        self.model_label.setText(f"Model: {self.agent.client._active_model or 'initializing…'}")
        self.info_label.setText(
            f"Model: {self.agent.client._active_model or 'unknown'} | Device: {self.agent.client.active_device}"
        )
        self.tabs.setCurrentIndex(0)

    @QtCore.pyqtSlot(str)
    def set_preset_fields(self, name: str) -> None:
        if self.manual_mode.isChecked() or name not in self.presets:
            return
        self.current_preset = name
        preset = self.presets[name]
        self.max_tokens_input.setValue(int(preset.get("max_tokens", 512)))
        self.device_select.setCurrentText(str(preset.get("device", "auto")))
        self.cpu_threads_input.setValue(int(preset.get("threads", 1)))

    def start_rescan(self) -> None:
        self.rescan_button.setDisabled(True)
        self.rescan_button.setText("Rescanning...")
        self._rescan_thread = QtCore.QThread()
        self._rescan_worker = RescanWorker()
        self._rescan_worker.moveToThread(self._rescan_thread)
        self._rescan_thread.started.connect(self._rescan_worker.run)
        self._rescan_worker.finished.connect(self.on_rescan_done)
        self._rescan_worker.finished.connect(self._rescan_thread.quit)
        self._rescan_thread.start()
        self.preset_status.setText("Scan in progress...") 
        self.rescan_button.setEnabled(False)

    @QtCore.pyqtSlot(dict, bool)
    def on_rescan_done(self, result: dict, ok: bool) -> None:
        if result:
            # result contains keys: profile, presets
            self.profile = result.get("profile", self.profile)
            self.presets = result.get("presets", self.presets)
            self.profile_ok = ok
            self.preset_status.setText(
                f"Scan {'ok' if self.profile_ok else 'fallback'} | Cores: {self.profile.cores}, RAM: {self.profile.ram_gb:.1f} GB, GPU: {self.profile.has_gpu}"
            )
            # Reset preset selection to balanced or first
            if "balanced" in self.presets:
                self.preset_select.setCurrentText("balanced")
            else:
                self.preset_select.setCurrentIndex(0)
            # Reapply fields from current preset
            if not self.manual_mode.isChecked():
                self.set_preset_fields(self.preset_select.currentText())
        self.rescan_button.setEnabled(True)
        self.rescan_button.setText("Rescan System")

    def toggle_manual_mode(self, state: int) -> None:
        manual = state == QtCore.Qt.Checked
        self.preset_select.setEnabled(not manual)
        if not manual:
            self.set_preset_fields(self.preset_select.currentText())

    def show_quick_prompts(self) -> None:
        menu = QtWidgets.QMenu(self)
        for category, prompts in QUICK_PROMPTS.items():
            cat_menu = menu.addMenu(category)
            for prompt in prompts:
                action = QtWidgets.QAction(prompt, self)
                action.triggered.connect(lambda checked, p=prompt: self.input.setPlainText(p))
                cat_menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    def update_eval(self, response, elapsed: float) -> None:
        # Deprecated; kept for compatibility. Use refresh_eval_totals instead.
        self.refresh_eval_totals()

    def refresh_eval_totals(self) -> None:
        total_responses = len(self.history_data)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        elapsed_values = []
        for turn in self.history_data:
            usage = turn.get("usage", {})
            try:
                total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            except Exception:
                pass
            try:
                total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            except Exception:
                pass
            try:
                elapsed_val = float(turn.get("elapsed", 0.0) or 0.0)
                if elapsed_val > 0:
                    elapsed_values.append(elapsed_val)
            except Exception:
                pass
        avg_time = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0.0
        self.eval_count = total_responses
        self.eval_total_time = sum(elapsed_values)
        self.eval_prompt_tokens = total_prompt_tokens
        self.eval_completion_tokens = total_completion_tokens
        if total_responses == 0:
            self.eval_summary.setText("No evaluations yet.")
        else:
            self.eval_summary.setText(
                f"Total responses: {total_responses}\n"
                f"Average response time: {avg_time:.2f}s\n"
                f"Total prompt tokens: {total_prompt_tokens}\n"
                f"Total completion tokens: {total_completion_tokens}"
            )

    def insert_eval_row(
        self,
        timestamp: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        model: str,
        device: str,
        threads: str,
        quant: str,
        session: bool,
        elapsed: float | None = None,
    ) -> None:
        # newest at top
        row = 0
        self.turn_table.insertRow(row)
        elapsed_display = f"{float(elapsed):.2f}" if elapsed is not None else "--"
        values = [
            timestamp,
            elapsed_display,
            str(prompt_tokens),
            str(completion_tokens),
            f"${float(cost):.4f}",
            model,
            device,
            threads,
            quant,
        ]
        for col, val in enumerate(values):
            item = QtWidgets.QTableWidgetItem(val)
            if session:
                pal = self.palette()
                item.setBackground(pal.alternateBase())
            self.turn_table.setItem(row, col, item)
        if self.turn_table.rowCount() > 200:
            self.turn_table.removeRow(self.turn_table.rowCount() - 1)
        self.model_info_eval.setText(
            f"Model: {model or 'unknown'} | Device: {device} | Quantization: {quant}"
        )


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
