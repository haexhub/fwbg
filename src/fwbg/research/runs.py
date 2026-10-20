"""The research gate on the backtest run path (``python -m fwbg.cli``).

``main()`` prepares the run metadata, ``run_optimizer`` calls
:meth:`ResearchRun.before_run` once the run's symbols are known (which puts
the run on record as pending), and :func:`run_gated` records the outcome.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.core.presets import resolve_presets
from fwbg.research.card import CARD_ID_RE, ResearchCard, load_card
from fwbg.research.fingerprint import FINGERPRINT_VERSION, canonical_config
from fwbg.research.gate import Decision, Denial, DenialCode, RunRequest, check
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialStatus, read_trials
from fwbg.research.legacy import run_window
from fwbg.research.pending import (
    STAGING,
    PendingError,
    PendingRun,
    finish,
    live_runs,
    recover_stale,
    tag_run,
    valid_run_id,
    write_pending,
)
from fwbg.research.repo import REGISTER_PATHS, ResearchRepoError, commit_register, register_lock

GATE_ENV = "FWBG_RESEARCH_GATE"

BeforeRun = Callable[[Mapping[str, Any], Sequence[str]], dict[str, Any]]


def gate_enabled() -> bool:
    """On unless ``FWBG_RESEARCH_GATE=off``, which only the test suite sets."""
    return os.environ.get(GATE_ENV, "").strip().lower() != "off"


class GateDenied(Exception):
    """The register refused the run; nothing was started or written."""

    def __init__(self, denials: Sequence[Denial]) -> None:
        super().__init__("; ".join(denial.message for denial in denials))
        self.denials = tuple(denials)


class GateChecked(Exception):
    """``--check``: the run would be allowed; it stops before it starts."""

    def __init__(self, decision: Decision, symbols: tuple[str, ...]) -> None:
        super().__init__("allowed")
        self.decision = decision
        self.symbols = symbols


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The research flags of the backtest command."""
    group = parser.add_argument_group("research register")
    group.add_argument("--research-id", metavar="S-NNNN", help="research card this run iterates")
    group.add_argument("--note", help="one line: what changed since the last iteration and why")
    group.add_argument("--reason", help="why an identical configuration runs again")
    group.add_argument(
        "--check", action="store_true", help="check the run against the register; do not run it"
    )


def prepare_metadata(metadata: Mapping[str, Any], default_timeframe: str) -> dict[str, Any]:
    """The one metadata dict the gate fingerprints, the run uses and the register stores.

    Raises:
        OSError, ValueError: a preset cannot be loaded or the timeframe is unknown.
    """
    prepared = resolve_presets(metadata)
    timeframe = prepared.get("timeframe") or default_timeframe
    prepared["timeframe"] = Timeframe.from_str(timeframe).canonical
    data_path = prepared.get("_cli_data_path")
    if isinstance(data_path, str) and data_path:
        prepared["_cli_data_path"] = str(Path(data_path).resolve())
    return prepared


@dataclass
class ResearchRun:
    """Gate and record one backtest run."""

    research_dir: Path
    run_id: str
    run_dir: Path
    research_id: str | None
    note: str | None
    reason: str | None = None
    check_only: bool = False
    _pending: PendingRun | None = field(default=None, init=False, repr=False)

    def before_run(self, metadata: Mapping[str, Any], symbols: Sequence[str]) -> dict[str, Any]:
        """Gate the run once its symbols are known; put it on record; return its metadata.

        Raises:
            GateDenied: the run must not start (also when the register is unreadable).
            GateChecked: ``check_only`` and the run would be allowed.
        """
        canonical = tuple(sorted({symbol.upper() for symbol in symbols}))
        request = RunRequest(self.research_id, metadata, canonical, self.note, self.reason)
        try:
            with register_lock(self.research_dir):
                return self._gate(request, canonical)
        except (GateDenied, GateChecked):
            raise
        except Exception as exc:  # fail closed: an unreadable register never lets a run through
            message = f"research register: {exc}"
            if isinstance(exc, PendingError):
                message += f"; inspect or remove the marker in {self.research_dir / STAGING}"
            raise GateDenied([Denial(DenialCode.REGISTER_UNREADABLE, message)]) from exc

    def _gate(self, request: RunRequest, symbols: tuple[str, ...]) -> dict[str, Any]:
        if recover_stale(self.research_dir):
            try:
                commit_register(self.research_dir, REGISTER_PATHS, "trial: record interrupted runs")
            except ResearchRepoError as exc:
                _warn_uncommitted("interrupted runs", exc)
        card: ResearchCard | None = None
        if self.research_id is not None and CARD_ID_RE.fullmatch(self.research_id):
            card = load_card(self.research_dir, self.research_id)
        trials = read_trials(self.research_dir / LEDGER_FILENAME)
        decision = check(card, request, trials)
        denials = list(decision.denials) + self._run_denials(decision, trials)
        if denials:
            raise GateDenied(denials)
        if self.check_only:
            raise GateChecked(decision, symbols)
        assert card is not None and decision.fingerprint is not None  # allowed decisions have both
        metadata = copy.deepcopy(dict(decision.effective_metadata))
        pending = PendingRun(
            run_id=self.run_id,
            run_dir=self.run_dir,
            pid=os.getpid(),
            trial=Trial(
                strategy=card.id,
                run_id=self.run_id,
                kind=decision.kind,
                status=TrialStatus.FAILED,  # replaced by the outcome when recorded
                ts=datetime.now().isoformat(timespec="seconds"),
                fingerprint=decision.fingerprint,
                fingerprint_version=FINGERPRINT_VERSION,
                criteria_hash=card.criteria_hash,
                holdout_cutoff=card.holdout_cutoff,
                code={},
                window=run_window(metadata),
                note=self.note,
                reason=self.reason if self.reason and self.reason.strip() else None,
            ),
            # The run may mutate the returned metadata; the marker must not change with it.
            config=copy.deepcopy(canonical_config(metadata, symbols)),
        )
        write_pending(self.research_dir, pending)
        # Only a run on record may be recorded: a failed marker write refuses the run.
        self._pending = pending
        return metadata

    def _run_denials(self, decision: Decision, trials: Sequence[Trial]) -> list[Denial]:
        """A run id is used once; an identical configuration must not run twice at once."""
        if not valid_run_id(self.run_id):
            # Checked first: every check below uses the run id as a file name.
            message = f"invalid run id {self.run_id!r}; use a plain name without '/'"
            return [Denial(DenialCode.INVALID_RUN_ID, message)]
        denials = []
        if (self.run_dir / "config.json").exists() or (self.run_dir / "grid_details").exists():
            denials.append(
                Denial(DenialCode.RUN_ID_IN_USE, f"run id {self.run_id} already holds a run")
            )
        recorded = [trial.trial_id for trial in trials if trial.run_id == self.run_id]
        if recorded:
            denials.append(
                Denial(
                    DenialCode.RUN_ID_IN_USE,
                    f"run id {self.run_id} is already recorded ({', '.join(map(str, recorded))})",
                )
            )
        running = live_runs(self.research_dir)
        if any(run.run_id == self.run_id for run in running):
            denials.append(Denial(DenialCode.RUN_ID_IN_USE, f"run {self.run_id} is running"))
        if decision.fingerprint is not None and not (self.reason and self.reason.strip()):
            same = [run.run_id for run in running if run.trial.fingerprint == decision.fingerprint]
            if same:
                denials.append(
                    Denial(
                        DenialCode.DUPLICATE_FINGERPRINT,
                        f"identical configuration is running now (run {', '.join(same)}); "
                        "pass a reason to run it again",
                    )
                )
        return denials

    def record(self) -> Trial | None:
        """Record the gated run's outcome; warn instead of failing the run.

        ``None`` if the gate never let the run start or recording failed (the
        marker then stays and the next gated run records it).
        """
        if self._pending is None:
            return None
        try:
            with register_lock(self.research_dir):
                trial = finish(self.research_dir, self._pending)
                try:
                    commit_register(
                        self.research_dir,
                        REGISTER_PATHS,
                        f"trial {trial.trial_id}: {trial.strategy} run {self.run_id}",
                    )
                except ResearchRepoError as exc:
                    _warn_uncommitted(str(trial.trial_id), exc)
        except Exception as exc:
            # Decision: recording never fails a run whose results already exist.
            print(
                f"Warning: run {self.run_id} is not in the ledger yet ({exc}); "
                "the next gated run records it",
                file=sys.stderr,
            )
            return None
        try:
            tag_run(self.run_dir, trial)
        except (OSError, ValueError, TypeError) as exc:
            print(
                f"Warning: could not tag run {self.run_id} with {trial.trial_id}: {exc}",
                file=sys.stderr,
            )
        return trial


def run_gated(research: ResearchRun, run: Callable[[BeforeRun], Any]) -> Any:
    """Run *run* behind the gate and record it.

    Exits with 2 on a refusal and 0 after ``--check``. Any other exception is
    recorded (outcome from the run directory) and re-raised. A process ended
    by ``os._exit`` (the pool's Ctrl-C handler) or SIGKILL keeps its pending
    marker and is recorded by the next gated run.
    """
    try:
        result = run(research.before_run)
    except GateDenied as exc:
        print("Research gate refused the run:", file=sys.stderr)
        for denial in exc.denials:
            print(f"  - [{denial.code.value}] {denial.message}", file=sys.stderr)
        raise SystemExit(2) from None
    except GateChecked as exc:
        _print_check(exc.decision, exc.symbols)
        raise SystemExit(0) from None
    except BaseException:
        research.record()
        raise
    if research._pending is None and not research.check_only:
        print(
            f"Warning: run {research.run_id} never passed the research gate and was not recorded",
            file=sys.stderr,
        )
    research.record()
    return result


def _warn_uncommitted(what: str, exc: ResearchRepoError) -> None:
    print(
        f"Warning: {what} not committed yet ({exc}); the next register commit includes it",
        file=sys.stderr,
    )


def _print_check(decision: Decision, symbols: tuple[str, ...]) -> None:
    window = run_window(decision.effective_metadata)
    print(f"Research gate: allowed ({decision.kind.value})")
    print(f"  fingerprint: {decision.fingerprint}")
    print(f"  symbols:     {', '.join(symbols)}")
    print(
        f"  window:      {window['start'] or 'open'} .. {window['end'] or 'open'} (end exclusive)"
    )
