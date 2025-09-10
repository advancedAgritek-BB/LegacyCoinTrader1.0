import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class FileState:
    path: Path
    inode: Optional[int] = None
    offset: int = 0


@dataclass
class WindowStats:
    start_ts: float
    end_ts: float
    counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    symbols_filtered_passed: int = 0
    errors: List[str] = field(default_factory=list)


class LogMonitor:
    def __init__(self, repo_root: Path, flush_interval: int = 30) -> None:
        self.repo_root = repo_root
        self.logs_dir = repo_root / "crypto_bot" / "logs"
        self.flush_interval = flush_interval
        self.state_by_file: Dict[Path, FileState] = {}
        self.overall_counters: Dict[str, int] = defaultdict(int)
        self.overall_reasons: Dict[str, int] = defaultdict(int)
        self.last_flush_ts: float = time.time()
        self.window = WindowStats(start_ts=self.last_flush_ts, end_ts=self.last_flush_ts)
        self.summary_csv = self.logs_dir / "monitor_funnel.csv"
        self.summary_json = self.logs_dir / "monitor_summary.json"
        self.backlog_path = self.logs_dir / "monitor_backlog.jsonl"
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.re_eval_reason = re.compile(r"\\[EVAL\\] (.+)")
        self.re_trade_blocked = re.compile(r"Trade blocked for .*?: (.+)")
        self.re_symbol_filter_done = re.compile(r"Symbol filtering completed: (\\d+) symbols passed filtering")
        self.re_no_analysis = re.compile(r"No analysis results to act on")
        self.re_all_filtered = re.compile(r"All signals filtered out - nothing actionable")
        self.re_insufficient_hist = re.compile(r"Skipping analysis for .*?: insufficient data \\((\\d+) candles\\)")
        self.re_max_open_trades = re.compile(r"Max open trades reached; skipping remaining signals")
        self.re_existing_position = re.compile(r"Existing position for .*? - skipping")
        self.re_error = re.compile(r" - ERROR - (.+)")

    def _list_log_files(self) -> List[Path]:
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(self.logs_dir.glob("*.log"))
        return files

    def _open_and_seek(self, fs: FileState) -> None:
        try:
            st = fs.path.stat()
        except FileNotFoundError:
            fs.inode = None
            fs.offset = 0
            return
        inode = getattr(st, "st_ino", None)
        if fs.inode is None or fs.inode != inode:
            fs.inode = inode
            fs.offset = st.st_size

    def _read_new_lines(self, fs: FileState) -> List[str]:
        try:
            st = fs.path.stat()
        except FileNotFoundError:
            fs.inode = None
            fs.offset = 0
            return []
        if st.st_size < fs.offset:
            fs.offset = 0
        lines: List[str] = []
        with fs.path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(fs.offset)
            chunk = f.read()
            fs.offset = f.tell()
        if chunk:
            lines = chunk.splitlines()
        return lines

    def _ensure_file_state(self, path: Path) -> FileState:
        fs = self.state_by_file.get(path)
        if fs is None:
            fs = FileState(path=path)
            self.state_by_file[path] = fs
            self._open_and_seek(fs)
        return fs

    def _inc(self, key: str, n: int = 1) -> None:
        self.window.counters[key] += n
        self.overall_counters[key] += n

    def _inc_reason(self, reason: str, n: int = 1) -> None:
        reason_key = reason.strip().lower().replace(" ", "_")
        self.window.reasons[reason_key] += n
        self.overall_reasons[reason_key] += n

    def _parse_line(self, line: str) -> None:
        m = self.re_eval_reason.search(line)
        if m:
            self._inc("eval.events")
            self._inc_reason(m.group(1))
            return

        m = self.re_trade_blocked.search(line)
        if m:
            self._inc("exec.blocked")
            self._inc_reason(m.group(1))
            return

        m = self.re_symbol_filter_done.search(line)
        if m:
            try:
                passed = int(m.group(1))
            except Exception:
                passed = 0
            self.window.symbols_filtered_passed += passed
            self._inc("filter.completed")
            return

        if self.re_no_analysis.search(line):
            self._inc("analysis.no_results")
            return

        if self.re_all_filtered.search(line):
            self._inc("analysis.all_filtered")
            return

        if self.re_max_open_trades.search(line):
            self._inc("exec.blocked_max_open_trades")
            return

        if self.re_existing_position.search(line):
            self._inc("exec.skip_existing_position")
            return

        m = self.re_insufficient_hist.search(line)
        if m:
            self._inc("analysis.skipped_insufficient_history")
            return

        m = self.re_error.search(line)
        if m:
            self.window.errors.append(m.group(1))
            self._inc("errors")
            return

    def _emit_summary(self) -> None:
        now = time.time()
        self.window.end_ts = now
        self._write_csv_summary()
        self._write_json_summary()
        self._update_backlog()
        self.window = WindowStats(start_ts=now, end_ts=now)
        self.last_flush_ts = now

    def _write_csv_summary(self) -> None:
        header = [
            "window_start",
            "window_end",
            "symbols_filtered_passed",
            "eval_events",
            "exec_blocked",
            "analysis_no_results",
            "analysis_all_filtered",
            "exec_blocked_max_open_trades",
            "exec_skip_existing_position",
            "errors",
        ]
        row = [
            int(self.window.start_ts),
            int(self.window.end_ts),
            self.window.symbols_filtered_passed,
            self.window.counters.get("eval.events", 0),
            self.window.counters.get("exec.blocked", 0),
            self.window.counters.get("analysis.no_results", 0),
            self.window.counters.get("analysis.all_filtered", 0),
            self.window.counters.get("exec.blocked_max_open_trades", 0),
            self.window.counters.get("exec.skip_existing_position", 0),
            self.window.counters.get("errors", 0),
        ]
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.summary_csv.exists()
        with self.summary_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)

    def _write_json_summary(self) -> None:
        out = {
            "window": {
                "start": int(self.window.start_ts),
                "end": int(self.window.end_ts),
            },
            "counters": dict(self.window.counters),
            "reasons": dict(self.window.reasons),
            "symbols_filtered_passed": self.window.symbols_filtered_passed,
            "overall": {
                "counters": dict(self.overall_counters),
                "reasons": dict(self.overall_reasons),
            },
            "errors": self.window.errors[-50:],
        }
        self.summary_json.write_text(json.dumps(out, indent=2))

    def _update_backlog(self) -> None:
        issues = self._detect_issues()
        if not issues:
            return
        self.backlog_path.parent.mkdir(parents=True, exist_ok=True)
        with self.backlog_path.open("a", encoding="utf-8") as f:
            for it in issues:
                f.write(json.dumps(it) + "\n")

    def _detect_issues(self) -> List[Dict[str, object]]:
        issues: List[Dict[str, object]] = []
        reasons = self.window.reasons
        ctr = self.window.counters

        def add(issue: str, severity: str, evidence: Dict[str, object], prompt: str, priority: int) -> None:
            issues.append({
                "ts": int(self.window.end_ts),
                "issue": issue,
                "severity": severity,
                "priority": priority,
                "evidence": evidence,
                "fix_prompt": prompt,
            })

        total_blocks = ctr.get("exec.blocked", 0)
        if total_blocks >= 5:
            top = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]
            prompt = (
                "Investigate dominant trade block reasons and propose config or logic changes. "
                "For each reason, suggest a concrete edit with parameters and affected files."
            )
            add(
                "High number of blocked trades",
                "high",
                {"blocked": total_blocks, "top_reasons": top},
                prompt,
                1,
            )

        if reasons.get("not_enough_data_to_trade", 0) >= 3:
            add(
                "Frequent insufficient data",
                "medium",
                {"count": reasons.get("not_enough_data_to_trade", 0)},
                (
                    "Increase OHLCV warmup/lookback in data fetch; ensure cache holds sufficient candles. "
                    "Lower minimum required candles in risk check or prefetch more history."
                ),
                2,
            )

        if reasons.get("sentiment_too_bearish", 0) >= 3:
            add(
                "Sentiment filter blocking trades",
                "medium",
                {"count": reasons.get("sentiment_too_bearish", 0)},
                (
                    "Reduce min_sentiment or min_fng in config; consider bypassing sentiment filter during pump regimes."
                ),
                2,
            )

        if reasons.get("market_volatility_too_low", 0) >= 3:
            add(
                "ATR threshold too strict",
                "medium",
                {"count": reasons.get("market_volatility_too_low", 0)},
                (
                    "Lower min_atr_pct or switch to higher-volatility timeframe; allow smaller ATR environments with reduced size."
                ),
                2,
            )

        if ctr.get("analysis.no_results", 0) + ctr.get("analysis.all_filtered", 0) >= 3:
            add(
                "No actionable analysis results",
                "high",
                {
                    "no_results": ctr.get("analysis.no_results", 0),
                    "all_filtered": ctr.get("analysis.all_filtered", 0),
                },
                (
                    "Lower signal thresholds or include additional strategies; verify symbol_filter allows enough candidates and raise top_n_symbols."
                ),
                1,
            )

        if ctr.get("exec.blocked_max_open_trades", 0) >= 1:
            add(
                "Max open trades limit reached",
                "medium",
                {"count": ctr.get("exec.blocked_max_open_trades", 0)},
                (
                    "Increase max concurrent trades in PositionGuard settings or prioritize higher-score signals only."
                ),
                3,
            )

        if self.window.symbols_filtered_passed == 0 and ctr.get("filter.completed", 0) > 0:
            add(
                "Zero symbols passed filtering",
                "high",
                {"filter_completed": ctr.get("filter.completed", 0)},
                (
                    "Relax symbol_filter parameters (min_volume_usd, volume_percentile) or expand the symbols universe."
                ),
                1,
            )

        if self.window.errors:
            add(
                "Errors observed in logs",
                "high",
                {"examples": self.window.errors[-3:]},
                (
                    "Investigate stack traces and implement defensive checks; add retries with backoff for transient failures."
                ),
                1,
            )

        return issues

    def run(self) -> None:
        print(f"Monitoring logs in {self.logs_dir}")
        while True:
            files = self._list_log_files()
            for path in files:
                fs = self._ensure_file_state(path)
                for line in self._read_new_lines(fs):
                    self._parse_line(line)
            if time.time() - self.last_flush_ts >= self.flush_interval:
                try:
                    self._emit_summary()
                except Exception as exc:
                    # Keep the monitor alive on write errors
                    sys.stderr.write(f"Monitor flush failed: {exc}\n")
            time.sleep(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tail bot logs and compute funnel stats")
    p.add_argument("--interval", type=int, default=30, help="Flush interval seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    monitor = LogMonitor(repo_root=repo_root, flush_interval=args.interval)
    monitor.run()


if __name__ == "__main__":
    main()


