"""Top-level launcher for PYLLAMATUTOR with dependency check splash."""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

ROOT = Path(__file__).resolve().parent / "pyllama"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gui import ChatWindow  # type: ignore
from system_profile import build_presets, detect_system


REQUIREMENTS = ROOT / "requirements.txt"


def missing_dependencies() -> List[str]:
    if not REQUIREMENTS.exists():
        return []
    reqs = []
    with REQUIREMENTS.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line.split("#")[0].strip())
    missing = []
    for req in reqs:
        try:
            import importlib.metadata as importlib_metadata

            importlib_metadata.version(req.split("==")[0].split(">=")[0])
        except Exception:
            missing.append(req)
    return missing


class SetupWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool)

    def __init__(self, deps: List[str]) -> None:
        super().__init__()
        self.deps = deps
        self.profile = None
        self.profile_ok = False
        self.presets = {}
        self.agent = None
        self.model_ready = False
        self.history_data = []

    @QtCore.pyqtSlot()
    def run(self) -> None:
        total = max(1, len(self.deps))
        completed = 0
        if self.deps:
            for dep in self.deps:
                self.progress.emit(completed, total, f"Installing {dep}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
                    completed += 1
                    self.progress.emit(completed, total, f"Installed {dep}")
                except Exception as exc:
                    self.progress.emit(completed, total, f"Failed {dep}: {exc}")
                    self.finished.emit(False)
                    return
        else:
            self.progress.emit(completed, total, "All dependencies present.")

        # System scan
        self.progress.emit(total, total, "Scanning system profile…")
        try:
            prof, ok = detect_system()
            self.profile = prof
            self.profile_ok = ok
            self.presets = build_presets(prof, ok)
            self.progress.emit(total, total, "System scan complete.")
            # Choose preset defaults and set env hints
            chosen = "balanced" if "balanced" in self.presets else next(iter(self.presets.keys()))
            preset = self.presets[chosen]
            os.environ["LLAMA_DEVICE"] = str(preset.get("device", "auto"))
            os.environ["OMP_NUM_THREADS"] = str(preset.get("threads", 1))
            # Pre-initialize agent/model so main UI opens ready
            from agent import TutorAgent  # type: ignore

            self.agent = TutorAgent(model=None, max_new_tokens=int(preset.get("max_tokens", 512)), quantization="none")
            self.agent.client.ensure_ready()
            self.model_ready = True
            self.progress.emit(total, total, "Model initialized.")
            # Load history if present
            hist_path = Path.home() / ".pyllama" / "chat_history.json"
            if hist_path.exists():
                try:
                    import json

                    with hist_path.open() as hf:
                        data = json.load(hf)
                        self.history_data = data.get("turns", [])
                    self.progress.emit(total, total, f"History loaded: {len(self.history_data)} entries.")
                except Exception as exc:
                    self.progress.emit(total, total, f"History load failed: {exc}")
        except Exception as exc:
            self.progress.emit(total, total, f"System scan failed: {exc}")
            self.finished.emit(False)
            return

        self.finished.emit(True)


class Splash(QtWidgets.QDialog):
    def __init__(self, deps: List[str]) -> None:
        super().__init__()
        self.setWindowTitle("")
        icon_path = ROOT / "images" / "AppIcon.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        self.resize(420, 260)
        layout = QtWidgets.QVBoxLayout(self)

        logo = QtWidgets.QLabel()
        logo.setAlignment(QtCore.Qt.AlignCenter)
        logo_path = ROOT / "images" / "LogoLight.png"
        if logo_path.exists():
            pix = QtGui.QPixmap(str(logo_path))
            logo.setPixmap(pix.scaledToWidth(180, QtCore.Qt.SmoothTransformation))
        layout.addWidget(logo)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, max(1, len(deps)))
        layout.addWidget(self.bar)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        layout.addWidget(self.log)

        self.status = QtWidgets.QLabel("Loading Meta LLaMa Python Tutor…")
        layout.addWidget(self.status)

        self.scan_bar = QtWidgets.QProgressBar()
        self.scan_bar.setRange(0, 0)
        self.scan_bar.setVisible(False)
        layout.addWidget(self.scan_bar)

        sig = QtWidgets.QLabel("by Anatole Dupuis")
        sig.setAlignment(QtCore.Qt.AlignLeft)
        sig.setStyleSheet("color: #555; font-size: 9pt; padding-top:4px;")
        layout.addWidget(sig)

        self.worker = SetupWorker(deps)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

        self.ok = False
        self.cancelled = False
        self.profile = None
        self.profile_ok = False
        self.presets = {}

        btns = QtWidgets.QHBoxLayout()
        exit_btn = QtWidgets.QPushButton("Exit")
        exit_btn.clicked.connect(self.on_exit)
        btns.addStretch()
        btns.addWidget(exit_btn)
        layout.addLayout(btns)

    @QtCore.pyqtSlot(int, int, str)
    def on_progress(self, done: int, total: int, msg: str) -> None:
        self.bar.setMaximum(max(1, total))
        self.bar.setValue(done)
        self.log.append(msg)
        self.status.setText("Loading Meta LLaMa Python Tutor…")
        if "Scanning system profile" in msg or "Model initialized" in msg:
            self.scan_bar.setVisible(True)

    @QtCore.pyqtSlot(bool)
    def on_finished(self, ok: bool) -> None:
        self.ok = ok
        self.profile = self.worker.profile
        self.profile_ok = self.worker.profile_ok
        self.presets = self.worker.presets
        if ok:
            self.log.append("Loading Meta LLaMa Python Tutor…")
        self.accept() if ok else self.reject()

    @QtCore.pyqtSlot()
    def on_exit(self) -> None:
        self.cancelled = True
        self.reject()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    icon_path = ROOT / "images" / "AppIcon.png"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    deps = missing_dependencies()
    splash = Splash(deps)
    if splash.exec_() != QtWidgets.QDialog.Accepted:
        sys.exit(1)

    # Launch main window with pre-scanned info and preloaded agent if available
    agent = splash.worker.agent if hasattr(splash, "worker") else None
    window = ChatWindow(
        profile=getattr(splash, "profile", None),
        profile_ok=getattr(splash, "profile_ok", False),
        presets=getattr(splash, "presets", None),
        agent=agent,
        history_data=getattr(splash.worker, "history_data", None),
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
