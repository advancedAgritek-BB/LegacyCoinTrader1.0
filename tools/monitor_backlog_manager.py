#!/usr/bin/env python3
"""
Monitor Backlog Manager

This script helps manage the monitor_backlog.jsonl file by:
- Analyzing error patterns
- Clearing old entries
- Generating error reports
- Suggesting configuration updates
"""

import json
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any


class MonitorBacklogManager:
    """Manage monitor_backlog.jsonl file operations."""

    def __init__(self, backlog_file: str = "crypto_bot/logs/monitor_backlog.jsonl"):
        self.backlog_file = Path(backlog_file)
        self.entries = []

    def load_entries(self) -> List[Dict[str, Any]]:
        """Load all entries from the backlog file."""
        if not self.backlog_file.exists():
            print(f"Backlog file {self.backlog_file} does not exist")
            return []

        entries = []
        try:
            with open(self.backlog_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse line: {e}")
            self.entries = entries
            print(f"Loaded {len(entries)} entries from backlog")
            return entries
        except Exception as e:
            print(f"Failed to load backlog file: {e}")
            return []

    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns in the backlog."""
        if not self.entries:
            self.load_entries()

        analysis = {
            'total_entries': len(self.entries),
            'error_types': Counter(),
            'severities': Counter(),
            'time_range': {},
            'common_errors': Counter(),
            'affected_symbols': Counter(),
            'api_endpoints': Counter()
        }

        if not self.entries:
            return analysis

        timestamps = []
        for entry in self.entries:
            timestamps.append(entry.get('ts', 0))
            analysis['error_types'][entry.get('issue', 'unknown')] += 1
            analysis['severities'][entry.get('severity', 'unknown')] += 1

            # Extract common error patterns
            examples = entry.get('evidence', {}).get('examples', [])
            for example in examples:
                # Extract symbol names
                if 'OHLCV' in example and 'on' in example:
                    parts = example.split()
                    for part in parts:
                        if '/' in part and len(part) > 3:
                            analysis['affected_symbols'][part] += 1

                # Extract API endpoints
                if 'https://' in example:
                    # Find URL in the example
                    import re
                    urls = re.findall(r'https://[^\s]+', example)
                    for url in urls:
                        analysis['api_endpoints'][url] += 1

                analysis['common_errors'][example[:100]] += 1

        if timestamps:
            analysis['time_range'] = {
                'earliest': min(timestamps),
                'latest': max(timestamps),
                'duration_hours': (max(timestamps) - min(timestamps)) / 3600
            }

        return analysis

    def clear_old_entries(self, days_old: int = 7) -> int:
        """Clear entries older than specified days."""
        if not self.entries:
            self.load_entries()

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        filtered_entries = [entry for entry in self.entries if entry.get('ts', 0) > cutoff_time]

        removed_count = len(self.entries) - len(filtered_entries)

        if removed_count > 0:
            # Write filtered entries back
            with open(self.backlog_file, 'w') as f:
                for entry in filtered_entries:
                    f.write(json.dumps(entry) + '\n')

            self.entries = filtered_entries
            print(f"Cleared {removed_count} entries older than {days_old} days")

        return removed_count

    def generate_report(self) -> str:
        """Generate a summary report of the backlog."""
        analysis = self.analyze_errors()

        report = []
        report.append("=== Monitor Backlog Analysis Report ===")
        report.append(f"Total Entries: {analysis['total_entries']}")
        report.append("")

        if analysis['time_range']:
            tr = analysis['time_range']
            report.append(f"Time Range: {datetime.fromtimestamp(tr['earliest'])} to {datetime.fromtimestamp(tr['latest'])}")
            report.append(".1f")
            report.append("")

        report.append("Error Types:")
        for error_type, count in analysis['error_types'].most_common(10):
            report.append(f"  {error_type}: {count}")
        report.append("")

        report.append("Severities:")
        for severity, count in analysis['severities'].items():
            report.append(f"  {severity}: {count}")
        report.append("")

        report.append("Most Affected Symbols:")
        for symbol, count in analysis['affected_symbols'].most_common(10):
            report.append(f"  {symbol}: {count}")
        report.append("")

        report.append("Common API Endpoints:")
        for endpoint, count in analysis['api_endpoints'].most_common(5):
            report.append(f"  {endpoint}: {count}")
        report.append("")

        report.append("Most Common Errors:")
        for error, count in analysis['common_errors'].most_common(10):
            report.append(f"  {error}...: {count}")

        return '\n'.join(report)

    def suggest_fixes(self) -> List[str]:
        """Suggest configuration fixes based on error patterns."""
        analysis = self.analyze_errors()
        suggestions = []

        # Check for specific error patterns
        error_types = [error.lower() for error in analysis['error_types'].keys()]

        if any('twitter' in error or 'sentiment' in error for error in error_types):
            suggestions.append("- Twitter sentiment API is using placeholder URL. Set TWITTER_SENTIMENT_URL environment variable to a valid API endpoint")

        if any('ohlcv' in error and 'kraken' in error for error in error_types):
            suggestions.append("- Kraken OHLCV errors detected. Consider:")
            suggestions.append("  * Increasing retry delays in api_error_handling.yaml")
            suggestions.append("  * Adding rate limiting for Kraken API calls")
            suggestions.append("  * Checking Kraken API key permissions")

        if any('alternative' in error and 'sopr' in error for error in error_types):
            suggestions.append("- Alternative.me SOPR endpoint returning 404. The endpoint may be deprecated")
            suggestions.append("  * Alternative endpoint configured: realized-profit-loss")
            suggestions.append("  * Consider using different on-chain metrics")

        if any('too many requests' in error for error in error_types):
            suggestions.append("- Rate limiting detected. Consider:")
            suggestions.append("  * Increasing delays between API calls")
            suggestions.append("  * Implementing circuit breaker pattern")
            suggestions.append("  * Adding request batching")

        if analysis['total_entries'] > 100:
            suggestions.append(f"- High error volume ({analysis['total_entries']} entries). Consider:")
            suggestions.append("  * Reviewing API keys and permissions")
            suggestions.append("  * Checking network connectivity")
            suggestions.append("  * Implementing error alerting")

        return suggestions


def main():
    parser = argparse.ArgumentParser(description="Monitor Backlog Manager")
    parser.add_argument('--file', default='crypto_bot/logs/monitor_backlog.jsonl',
                       help='Path to monitor backlog file')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze error patterns')
    parser.add_argument('--clear-old', type=int,
                       help='Clear entries older than N days')
    parser.add_argument('--report', action='store_true',
                       help='Generate summary report')
    parser.add_argument('--suggest-fixes', action='store_true',
                       help='Suggest configuration fixes')

    args = parser.parse_args()

    manager = MonitorBacklogManager(args.file)

    if args.analyze:
        analysis = manager.analyze_errors()
        print("Analysis complete")
        return

    if args.clear_old:
        removed = manager.clear_old_entries(args.clear_old)
        print(f"Cleared {removed} old entries")
        return

    if args.report:
        report = manager.generate_report()
        print(report)
        return

    if args.suggest_fixes:
        suggestions = manager.suggest_fixes()
        if suggestions:
            print("Suggested Fixes:")
            for suggestion in suggestions:
                print(f"  {suggestion}")
        else:
            print("No specific suggestions available")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
