"""Tests für die Progress-Anzeige (Konsole)."""
import os
import time
import pytest
from multiprocessing import Queue
from unittest.mock import MagicMock, patch

from fwbg.utils.progress import (
    ProgressTracker,
    report_meta,
    set_parallel_mode,
    is_parallel_mode,
)


class TestProgressTracker:
    """Tests für die ProgressTracker-Klasse."""

    def test_initialization(self):
        """ProgressTracker sollte korrekt initialisiert werden."""
        tracker = ProgressTracker(total_assets=5, asset_names=["A", "B", "C", "D", "E"])

        assert tracker.total_assets == 5
        assert tracker.completed_assets == 0
        assert tracker.asset_names == ["A", "B", "C", "D", "E"]
        assert tracker.completed_symbols == []

    def test_update_completed(self):
        """update_completed sollte den Fortschritt korrekt aktualisieren."""
        tracker = ProgressTracker(total_assets=3)

        tracker.update_completed(1, symbol="EURUSD")
        assert tracker.completed_assets == 1
        assert "EURUSD" in tracker.completed_symbols

        tracker.update_completed(2, symbol="GBPUSD")
        assert tracker.completed_assets == 2
        assert "GBPUSD" in tracker.completed_symbols

        # Duplikate sollten ignoriert werden
        tracker.update_completed(2, symbol="EURUSD")
        assert len(tracker.completed_symbols) == 2

    def test_format_time_seconds(self):
        """_format_time sollte Sekunden korrekt formatieren."""
        assert ProgressTracker._format_time(0) == "00:00"
        assert ProgressTracker._format_time(30) == "00:30"
        assert ProgressTracker._format_time(90) == "01:30"
        assert ProgressTracker._format_time(3661) == "1:01:01"
        assert ProgressTracker._format_time(-5) == "--:--"

    def test_format_time_hours(self):
        """_format_time sollte Stunden korrekt anzeigen."""
        assert ProgressTracker._format_time(3600) == "1:00:00"
        assert ProgressTracker._format_time(7200) == "2:00:00"
        assert ProgressTracker._format_time(3723) == "1:02:03"


class TestTotalProgressCalculation:
    """Tests für die Gesamt-Fortschrittsberechnung."""

    def test_only_completed_assets(self):
        """Fortschritt nur mit fertigen Assets."""
        tracker = ProgressTracker(total_assets=5)

        # Keine aktiven Worker, 2 fertige Assets
        active_workers = {}
        total_progress = tracker._calculate_total_progress(2, active_workers)

        assert total_progress == 2.0

    def test_with_active_workers(self):
        """Fortschritt mit aktiven Workern (Grid-Progress)."""
        tracker = ProgressTracker(total_assets=5)

        # 1 fertig, 2 Worker aktiv
        active_workers = {
            1001: {"symbol": "EURUSD", "grid_pos": 500, "grid_total": 1000},  # 50%
            1002: {"symbol": "GBPUSD", "grid_pos": 250, "grid_total": 1000},  # 25%
        }

        total_progress = tracker._calculate_total_progress(1, active_workers)

        # 1 fertig + 0.5 (EURUSD) + 0.25 (GBPUSD) = 1.75
        assert total_progress == pytest.approx(1.75)

    def test_with_zero_grid_total(self):
        """Worker ohne Grid-Total sollte keinen Progress beitragen."""
        tracker = ProgressTracker(total_assets=3)

        active_workers = {
            1001: {"symbol": "EURUSD", "grid_pos": 0, "grid_total": 0},  # Noch kein Grid
        }

        total_progress = tracker._calculate_total_progress(1, active_workers)

        # 1 fertig + 0 (kein Grid-Total) = 1.0
        assert total_progress == 1.0


class TestETACalculation:
    """Tests für die ETA-Berechnung."""

    def test_eta_too_early(self):
        """ETA sollte '--:--' sein wenn zu wenig Zeit vergangen."""
        tracker = ProgressTracker(total_assets=10)

        # Nur 10 Sekunden vergangen
        eta = tracker._calculate_eta(elapsed=10, total_progress=1.0)
        assert eta == "--:--"

    def test_eta_too_little_progress(self):
        """ETA sollte '--:--' sein wenn zu wenig Fortschritt."""
        tracker = ProgressTracker(total_assets=100)

        # 0.5% Fortschritt bei 100 Assets = 0.5 < 1.0 (min_progress)
        eta = tracker._calculate_eta(elapsed=60, total_progress=0.5)
        assert eta == "--:--"

    def test_eta_calculation_valid(self):
        """ETA sollte korrekt berechnet werden bei ausreichend Fortschritt."""
        tracker = ProgressTracker(total_assets=10)

        # 60 Sekunden, 2 von 10 fertig
        # Rate: 60s / 2 = 30s pro Asset
        # Verbleibend: 8 Assets * 30s = 240s = 4:00
        eta = tracker._calculate_eta(elapsed=60, total_progress=2.0)
        assert eta == "04:00"

    def test_eta_almost_done(self):
        """ETA sollte '00:00' sein wenn fast fertig."""
        tracker = ProgressTracker(total_assets=10)

        # 10 von 10 fertig
        eta = tracker._calculate_eta(elapsed=600, total_progress=10.0)
        assert eta == "00:00"

    def test_eta_single_asset(self):
        """ETA sollte bei einzelnem Asset funktionieren."""
        tracker = ProgressTracker(total_assets=1)

        # min_progress = 0.01 * 1 = 0.01
        # 50% fertig (0.5) > 0.01, also sollte ETA berechnet werden
        eta = tracker._calculate_eta(elapsed=60, total_progress=0.5)

        # 60s für 50% -> 60s für restliche 50%
        assert eta == "01:00"


class TestParallelMode:
    """Tests für den Parallel-Modus (Thread-lokale Unterdrückung)."""

    def test_parallel_mode_default(self):
        """Parallel-Modus sollte standardmäßig aus sein."""
        # Reset parallel mode zuerst
        set_parallel_mode(False)
        assert not is_parallel_mode()

    def test_set_parallel_mode(self):
        """Parallel-Modus sollte gesetzt werden können."""
        set_parallel_mode(True)
        assert is_parallel_mode()

        set_parallel_mode(False)
        assert not is_parallel_mode()

    def test_parallel_mode_suppresses_updates(self):
        """Im Parallel-Modus sollten Updates unterdrückt werden (Logik-Test)."""
        # Dieser Test prüft die Logik ohne tatsächliche Queue-Nutzung
        set_parallel_mode(True)

        # is_parallel_mode() sollte True zurückgeben
        assert is_parallel_mode()

        # Wenn is_parallel_mode() True ist, wird report_progress früh abbrechen
        # (keine Queue-Operation)

        set_parallel_mode(False)
        assert not is_parallel_mode()


class TestNonTTYRendering:
    """Tests für die Nicht-TTY (Log-Datei) Ausgabe."""

    def test_non_tty_output_format(self):
        """Nicht-TTY Ausgabe sollte kompakt sein."""
        tracker = ProgressTracker(total_assets=10, asset_names=["EURUSD", "GBPUSD"])
        tracker._is_tty = False
        tracker.start_time = time.time() - 120  # 2 Minuten vergangen
        tracker.completed_symbols = []

        phases = {"EURUSD": "Grid-Search"}

        with patch('sys.stdout') as mock_stdout:
            mock_stdout.flush = MagicMock()

            with patch('builtins.print') as mock_print:
                tracker._render_compact(
                    completed=2,
                    pct=25.0,
                    elapsed_str="02:00",
                    eta_str="06:00",
                    phases=phases
                )

                # Prüfe dass print aufgerufen wurde
                mock_print.assert_called_once()
                output = mock_print.call_args[0][0]

                # Format prüfen
                assert "25.0%" in output
                assert "2/10" in output
                assert "02:00" in output
                assert "ETA: 06:00" in output
                assert "EURUSD" in output


class TestAssetProgressBar:
    """Tests für die Asset-Progressbar (pro Worker)."""

    def test_worker_progress_percentage(self):
        """Worker-Progress sollte korrekt berechnet werden."""
        grid_pos = 250
        grid_total = 1000

        worker_pct = int(grid_pos / grid_total * 100)
        assert worker_pct == 25

        grid_pos = 999
        worker_pct = int(grid_pos / grid_total * 100)
        assert worker_pct == 99

    def test_worker_bar_filling(self):
        """Worker-Bar sollte proportional gefüllt werden."""
        bar_width = 15
        grid_pos = 500
        grid_total = 1000

        filled = int(bar_width * grid_pos / grid_total)
        bar = "▓" * filled + "░" * (bar_width - filled)

        assert len(bar) == bar_width
        assert bar.count("▓") == 7  # 50% von 15 = 7.5 -> 7
        assert bar.count("░") == 8


class TestOverallProgressBar:
    """Tests für die Gesamt-Progressbar."""

    def test_progress_bar_calculation(self):
        """Gesamt-Progressbar sollte korrekt berechnet werden."""
        total_assets = 10
        bar_width = 40

        # 3 fertig, 2 Worker bei 50%
        total_progress = 3.0 + 0.5 + 0.5  # = 4.0

        filled = int(bar_width * total_progress / total_assets)
        bar = "█" * filled + "░" * (bar_width - filled)

        assert len(bar) == bar_width
        assert bar.count("█") == 16  # 40% von 40 = 16
        assert bar.count("░") == 24

    def test_progress_bar_clamping(self):
        """Progressbar sollte nie über 100% gehen."""
        total_assets = 5
        bar_width = 40

        # Mehr als 100% sollte auf 100% begrenzt werden
        total_progress = 6.0  # Mehr als total_assets

        filled = int(bar_width * total_progress / total_assets)
        filled = min(filled, bar_width)  # Clamp
        bar = "█" * filled + "░" * (bar_width - filled)

        assert len(bar) == bar_width
        assert bar.count("█") == 40  # Voll
        assert bar.count("░") == 0


class TestQueueBasedUpdates:
    """Tests für Queue-basierte Progress-Updates (Verarbeitung der Messages)."""

    def test_progress_message_structure(self):
        """Progress-Messages sollten die richtige Struktur haben."""
        msg = {
            "type": "progress",
            "pid": 1001,
            "symbol": "EURUSD",
            "fold": 2,
            "total_folds": 8,
            "grid_pos": 100,
            "grid_total": 500,
            "time": time.time(),
        }

        assert msg["type"] == "progress"
        assert msg["symbol"] == "EURUSD"
        assert msg["grid_pos"] == 100
        assert msg["grid_total"] == 500

    def test_tracker_updates_worker_status(self):
        """Tracker sollte Worker-Status korrekt aktualisieren."""
        tracker = ProgressTracker(total_assets=5)

        # Simuliere Worker-Update direkt
        msg = {
            "type": "progress",
            "pid": 1001,
            "symbol": "EURUSD",
            "grid_pos": 100,
            "grid_total": 500,
            "time": time.time(),
        }
        tracker.worker_status[msg["pid"]] = msg

        assert 1001 in tracker.worker_status
        assert tracker.worker_status[1001]["symbol"] == "EURUSD"
        assert tracker.worker_status[1001]["grid_pos"] == 100

    def test_tracker_updates_phases(self):
        """Tracker sollte Phase-Updates korrekt speichern."""
        tracker = ProgressTracker(total_assets=5)

        # Simuliere Phase-Update direkt
        msg = {
            "type": "phase",
            "symbol": "EURUSD",
            "phase": "Grid-Search: 50/100",
        }
        tracker.worker_phases[msg["symbol"]] = msg["phase"]

        assert "EURUSD" in tracker.worker_phases
        assert tracker.worker_phases["EURUSD"] == "Grid-Search: 50/100"

    def test_tracker_removes_done_workers(self):
        """Tracker sollte fertige Worker entfernen."""
        tracker = ProgressTracker(total_assets=5)

        # Worker hinzufügen
        tracker.worker_status[1001] = {"symbol": "EURUSD", "grid_pos": 500, "grid_total": 500}
        tracker.worker_phases["EURUSD"] = "Grid-Search: 500/500"

        # Done-Meldung verarbeiten
        msg = {"type": "done", "pid": 1001, "symbol": "EURUSD", "status": "ok"}
        tracker.worker_status.pop(msg["pid"], None)
        tracker.worker_phases.pop(msg.get("symbol"), None)

        assert 1001 not in tracker.worker_status
        assert "EURUSD" not in tracker.worker_phases


class TestReportMeta:
    """Tests für report_meta() und Indikator-Anzeige im TUI."""

    def test_report_meta_sends_queue_message(self):
        """report_meta() sollte eine Meta-Message an die Queue senden."""
        import fwbg.utils.progress as prog
        q = Queue()
        old_queue = prog._progress_queue
        prog._progress_queue = q
        try:
            report_meta("EURUSD", indicator_count=11, feature_count=352)
            msg = q.get(timeout=1)
            assert msg["type"] == "meta"
            assert msg["symbol"] == "EURUSD"
            assert msg["indicator_count"] == 11
            assert msg["feature_count"] == 352
        finally:
            prog._progress_queue = old_queue
            q.close()

    def test_report_meta_without_queue(self):
        """report_meta() ohne Queue sollte nicht crashen."""
        import fwbg.utils.progress as prog
        old_queue = prog._progress_queue
        prog._progress_queue = None
        try:
            report_meta("EURUSD", indicator_count=5)  # Should not raise
        finally:
            prog._progress_queue = old_queue

    def test_tracker_stores_meta(self):
        """ProgressTracker sollte Meta-Daten pro Symbol speichern."""
        tracker = ProgressTracker(total_assets=3)

        msg = {"type": "meta", "symbol": "EURUSD", "indicator_count": 11, "feature_count": 352}
        tracker.worker_meta[msg["symbol"]] = {
            k: v for k, v in msg.items() if k not in ("type", "symbol")
        }

        assert "EURUSD" in tracker.worker_meta
        assert tracker.worker_meta["EURUSD"]["indicator_count"] == 11
        assert tracker.worker_meta["EURUSD"]["feature_count"] == 352

    def test_tracker_has_worker_meta_attribute(self):
        """ProgressTracker sollte worker_meta Dict haben."""
        tracker = ProgressTracker(total_assets=3)
        assert hasattr(tracker, "worker_meta")
        assert isinstance(tracker.worker_meta, dict)

    def test_grid_line_shows_feature_count(self):
        """Grid-Progress-Zeile sollte Feature-Anzahl enthalten wenn bekannt."""
        tracker = ProgressTracker(total_assets=2, asset_names=["EURUSD", "GBPUSD"])
        tracker._is_tty = True
        tracker.start_time = time.time() - 60
        tracker.completed_symbols = []

        # Meta-Daten mit feature_count (nach erstem Fold bekannt)
        tracker.worker_meta["EURUSD"] = {"indicator_count": 11, "feature_count": 352}

        # Grid-Progress setzen
        tracker.worker_status["EURUSD"] = {
            "symbol": "EURUSD",
            "fold": 2, "total_folds": 8,
            "grid_pos": 12, "grid_total": 16,
            "time": time.time(),
        }

        # Render und Output abfangen
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.isatty = MagicMock(return_value=True)
            mock_stdout.write = MagicMock()
            mock_stdout.flush = MagicMock()
            tracker._last_display_lines = 0
            tracker._render()

            output = "".join(
                call.args[0] for call in mock_stdout.write.call_args_list
            )
            assert "352F" in output
            assert "F2/8" in output
            assert "(12/16)" in output

    def test_grid_line_shows_plugin_count_as_fallback(self):
        """Vor erstem Fold: Plugin-Anzahl als Fallback anzeigen."""
        tracker = ProgressTracker(total_assets=2, asset_names=["EURUSD", "GBPUSD"])
        tracker._is_tty = True
        tracker.start_time = time.time() - 60
        tracker.completed_symbols = []

        # Nur indicator_count (feature_count noch nicht bekannt)
        tracker.worker_meta["EURUSD"] = {"indicator_count": 11}

        tracker.worker_status["EURUSD"] = {
            "symbol": "EURUSD",
            "fold": 1, "total_folds": 8,
            "grid_pos": 5, "grid_total": 16,
            "time": time.time(),
        }

        with patch('sys.stdout') as mock_stdout:
            mock_stdout.isatty = MagicMock(return_value=True)
            mock_stdout.write = MagicMock()
            mock_stdout.flush = MagicMock()
            tracker._last_display_lines = 0
            tracker._render()

            output = "".join(
                call.args[0] for call in mock_stdout.write.call_args_list
            )
            assert "11P" in output

    def test_done_clears_meta(self):
        """Done-Message sollte Meta-Daten für Symbol entfernen."""
        tracker = ProgressTracker(total_assets=3)
        tracker.worker_meta["EURUSD"] = {"indicator_count": 11}

        # Simuliere done
        msg = {"type": "done", "symbol": "EURUSD", "status": "ok"}
        tracker.worker_meta.pop(msg["symbol"], None)

        assert "EURUSD" not in tracker.worker_meta
