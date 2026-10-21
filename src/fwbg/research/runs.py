"""The research gate on the backtest run path (``python -m fwbg.cli``).

``main()`` prepares the run metadata, ``run_optimizer`` calls
:meth:`ResearchRun.before_run` once the run's symbols are known (which puts
the run on record as pending), and :func:`run_gated` records the outcome.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.core.presets import resolve_presets
from fwbg.research.card import CARD_ID_RE, CardError, ResearchCard, load_card
from fwbg.research.fingerprint import FINGERPRINT_VERSION, canonical_config
from fwbg.research.gate import Decision, Denial, DenialCode, RunRequest, check
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    LedgerError,
    Trial,
    TrialKind,
    TrialStatus,
    read_trials,
)
from fwbg.research.legacy import LegacyRunError, run_window
from fwbg.research.pending import (
    STAGING,
    PendingError,
    PendingRun,
    finish,
    live_runs,
    read_pending,
    recover_stale,
    tag_run,
    valid_run_id,
    write_pending,
)
from fwbg.research.repo import REGISTER_PATHS, ResearchRepoError, commit_register, register_lock

GATE_ENV = "FWBG_RESEARCH_GATE"
# ``--check-json`` prints its result as one stdout line with this prefix.
CHECK_PREFIX = "RESEARCH_CHECK "

BeforeRun = Callable[[Mapping[str, Any], Sequence[str]], dict[str, Any]]


def gate_enabled() -> bool:
    """On unless ``FWBG_RESEARCH_GATE=off``, which only the test suite sets."""
    return not (
        os.environ.get(GATE_ENV, "").strip().lower() == "off"
        and "PYTEST_CURRENT_TEST" in os.environ
    )


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
        "--holdout",
        action="store_true",
        help="look at the card's holdout once: develop before its cutoff, evaluate after it",
    )
    group.add_argument(
        "--check", action="store_true", help="check the run against the register; do not run it"
    )
    group.add_argument(
        "--check-json",
        action="store_true",
        help="like --check; also print the result as one JSON line",
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
    holdout: bool = False
    _pending: PendingRun | None = field(default=None, init=False, repr=False)

    def before_run(self, metadata: Mapping[str, Any], symbols: Sequence[str]) -> dict[str, Any]:
        """Gate the run once its symbols are known; put it on record; return its metadata.

        Raises:
            GateDenied: the run must not start (also when the register is unreadable).
            GateChecked: ``check_only`` and the run would be allowed.
        """
        canonical = tuple(sorted({symbol.upper() for symbol in symbols}))
        request = RunRequest(
            self.research_id, metadata, canonical, self.note, self.reason, holdout=self.holdout
        )
        try:
            with register_lock(self.research_dir):
                return self._gate(request, canonical)
        except (GateDenied, GateChecked):
            raise
        except (
            OSError,
            TypeError,
            ValueError,
            CardError,
            LedgerError,
            LegacyRunError,
            PendingError,
            ResearchRepoError,
        ) as exc:  # fail closed: an unreadable register never lets a run through
            message = f"research register: {exc}"
            if isinstance(exc, PendingError):
                message += f"; inspect or remove the marker in {self.research_dir / STAGING}"
            raise GateDenied([Denial(DenialCode.REGISTER_UNREADABLE, message)]) from exc

    def _gate(self, request: RunRequest, symbols: tuple[str, ...]) -> dict[str, Any]:
        record_stale(self.research_dir)
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
        """A run id is used once; an identical configuration must not run twice at once;
        a holdout look still pending counts like a recorded one."""
        denials = run_id_denials(self.research_dir, self.run_id, trials, run_dir=self.run_dir)
        if any(denial.code is DenialCode.INVALID_RUN_ID for denial in denials):
            return denials
        denials.extend(running_duplicates(self.research_dir, decision.fingerprint, self.reason))
        if decision.kind is TrialKind.HOLDOUT and not (self.reason and self.reason.strip()):
            denials.extend(pending_holdout_looks(self.research_dir, self.research_id))
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
                    warn_uncommitted(str(trial.trial_id), exc)
        except (
            OSError,
            TypeError,
            ValueError,
            CardError,
            LedgerError,
            LegacyRunError,
            PendingError,
            ResearchRepoError,
        ) as exc:
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


def run_gated(
    research: ResearchRun, run: Callable[[BeforeRun], Any], *, as_json: bool = False
) -> Any:
    """Run *run* behind the gate and record it.

    Exits with 2 on a refusal and 0 after ``--check``; *as_json* also prints
    either result as a :data:`CHECK_PREFIX` line. Any other exception is
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
        if as_json:
            print(CHECK_PREFIX + json.dumps(check_result(exc.denials)), flush=True)
        raise SystemExit(2) from None
    except GateChecked as exc:
        _print_check(exc.decision, exc.symbols)
        if as_json:
            payload = check_result(decision=exc.decision, symbols=exc.symbols)
            print(CHECK_PREFIX + json.dumps(payload), flush=True)
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


def check_result(
    denials: Sequence[Denial] = (),
    decision: Decision | None = None,
    symbols: Sequence[str] = (),
) -> dict[str, Any]:
    """The ``--check-json`` payload: allowed with kind, fingerprint, symbols, window and
    holdout cutoff (``None`` for an iteration), or refused with every denial and its
    HTTP status."""
    if denials:
        return {
            "allowed": False,
            "denials": [
                {"code": d.code.value, "status": d.code.http_status, "message": d.message}
                for d in denials
            ],
        }
    assert decision is not None
    return {
        "allowed": True,
        "kind": decision.kind.value,
        "fingerprint": decision.fingerprint,
        "symbols": list(symbols),
        "window": run_window(decision.effective_metadata),
        "holdout_cutoff": decision.effective_metadata.get("holdout_cutoff"),
        "denials": [],
    }


def parse_check_output(stdout: str, returncode: int) -> dict[str, Any] | None:
    """The ``--check-json`` payload in *stdout*: the last prefixed line, which must be
    valid and match the exit code; ``None`` otherwise.

    An allowed payload needs exit 0, a refusal exit 2: a line from a CLI that failed
    before its gate, or one a plugin forged, never counts as a gate result.
    """
    for line in reversed(stdout.splitlines()):
        if line.startswith(CHECK_PREFIX):
            try:
                payload = json.loads(line.removeprefix(CHECK_PREFIX))
            except ValueError:
                return None
            if not isinstance(payload, dict):
                return None
            allowed = payload.get("allowed")
            consistent = (allowed is True and returncode == 0) or (
                allowed is False and returncode == 2
            )
            return payload if consistent and _valid_check_payload(payload) else None
    return None


def _valid_check_payload(payload: Mapping[str, Any]) -> bool:
    """Validate the complete machine-readable gate result before trusting it."""
    if payload.get("allowed") is True:
        return (
            payload.get("kind") in {kind.value for kind in TrialKind}
            and isinstance(payload.get("fingerprint"), str)
            and bool(payload["fingerprint"])
            and isinstance(payload.get("symbols"), list)
            and all(isinstance(symbol, str) for symbol in payload["symbols"])
            and _valid_window(payload.get("window"))
            and (
                payload.get("holdout_cutoff") is None or isinstance(payload["holdout_cutoff"], str)
            )
            and payload.get("denials") == []
        )
    denials = payload.get("denials")
    return (
        payload.get("allowed") is False
        and isinstance(denials, list)
        and bool(denials)
        and all(
            isinstance(denial, dict)
            and isinstance(denial.get("code"), str)
            and denial.get("status") in (409, 422)
            and isinstance(denial.get("message"), str)
            and bool(denial["message"])
            for denial in denials
        )
    )


def _valid_window(window: object) -> bool:
    """Check the stable window shape emitted by :func:`check_result`."""
    return (
        isinstance(window, dict)
        and set(window) == {"start", "end"}
        and all(value is None or isinstance(value, str) for value in window.values())
    )


def run_id_denials(
    research_dir: Path, run_id: str, trials: Sequence[Trial], *, run_dir: Path | None = None
) -> list[Denial]:
    """A run id must be a plain name used once: not in the ledger, not pending, and
    (with *run_dir*) not holding a run already."""
    if not valid_run_id(run_id):
        # Checked first: every check below uses the run id as a file name.
        message = f"invalid run id {run_id!r}; use a plain name without '/'"
        return [Denial(DenialCode.INVALID_RUN_ID, message)]
    denials = []
    if run_dir is not None and (
        (run_dir / "config.json").exists() or (run_dir / "grid_details").exists()
    ):
        denials.append(Denial(DenialCode.RUN_ID_IN_USE, f"run id {run_id} already holds a run"))
    recorded = [trial.trial_id for trial in trials if trial.run_id == run_id]
    if recorded:
        denials.append(
            Denial(
                DenialCode.RUN_ID_IN_USE,
                f"run id {run_id} is already recorded ({', '.join(map(str, recorded))})",
            )
        )
    # Every marker, not only live ones: a stale marker that could not be recorded keeps its id.
    if any(run.run_id == run_id for run in read_pending(research_dir)):
        denials.append(
            Denial(DenialCode.RUN_ID_IN_USE, f"run {run_id} is pending (running or unrecorded)")
        )
    return denials


def running_duplicates(
    research_dir: Path, run_fingerprint: str | None, reason: str | None
) -> list[Denial]:
    """An identical configuration must not run twice at once, unless *reason* says why."""
    if run_fingerprint is None or (reason and reason.strip()):
        return []
    same = [
        run.run_id for run in live_runs(research_dir) if run.trial.fingerprint == run_fingerprint
    ]
    if not same:
        return []
    return [
        Denial(
            DenialCode.DUPLICATE_FINGERPRINT,
            f"identical configuration is running now (run {', '.join(same)}); "
            "pass a reason to run it again",
        )
    ]


def pending_holdout_looks(research_dir: Path, research_id: str | None) -> list[Denial]:
    """A holdout look of *research_id* that is running or unrecorded; call without a reason.

    Every marker counts, not only live ones: a dead look may already have shown results.
    """
    looks = [
        run.run_id
        for run in read_pending(research_dir)
        if run.trial.strategy == research_id and run.trial.kind is TrialKind.HOLDOUT
    ]
    if not looks:
        return []
    return [
        Denial(
            DenialCode.HOLDOUT_REPEAT,
            f"the holdout of {research_id} is already being looked at (run {', '.join(looks)}); "
            "a further look needs a reason",
        )
    ]


def record_stale(research_dir: Path) -> None:
    """Record the runs whose process is gone and commit them; call under the lock."""
    if recover_stale(research_dir):
        try:
            commit_register(research_dir, REGISTER_PATHS, "trial: record interrupted runs")
        except ResearchRepoError as exc:
            warn_uncommitted("interrupted runs", exc)


def warn_uncommitted(what: str, exc: ResearchRepoError) -> None:
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
    cutoff = decision.effective_metadata.get("holdout_cutoff")
    if cutoff is not None:
        print(
            f"  split:       development .. {cutoff} | holdout {cutoff} .. {window['end'] or 'open'}"
        )
