#!/usr/bin/env python3
"""
Pipeline Log Viewer - Clean view of trading pipeline execution
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def view_pipeline_logs(hours=24, follow=False, filter_phase=None, filter_symbol=None):
    """View pipeline logs with filtering options."""

    # Find the pipeline log file
    log_dir = Path(__file__).parent / "crypto_bot" / "logs"
    pipeline_log = log_dir / "pipeline.log"

    if not pipeline_log.exists():
        print(f"❌ Pipeline log file not found: {pipeline_log}")
        print("The trading engine may not have run yet, or logging is not configured.")
        return

    print(f"📊 Pipeline Log Viewer - Last {hours} hours")
    print(f"📁 Log file: {pipeline_log}")
    print("=" * 80)

    # Calculate cutoff time
    cutoff_time = datetime.now() - timedelta(hours=hours)

    try:
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        filtered_lines = []
        current_cycle = None
        cycle_summary = {}

        for line in lines:
            # Parse timestamp
            try:
                # Extract timestamp from log format: "2025-09-19 22:19:01"
                timestamp_str = line[:19]
                log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                if log_time < cutoff_time:
                    continue

            except (ValueError, IndexError):
                # If we can't parse timestamp, include the line
                pass

            # Apply filters
            if filter_phase and f"Phase: {filter_phase}" not in line:
                continue
            if filter_symbol and filter_symbol not in line:
                continue

            # Extract cycle information for summary
            if "[Cycle #" in line and "🚀 Starting trading pipeline" in line:
                # Extract cycle number
                try:
                    cycle_part = line.split("[Cycle #")[1].split(",")[0]
                    current_cycle = int(cycle_part)
                    cycle_summary[current_cycle] = {
                        'start_time': log_time,
                        'phases': [],
                        'symbols': set(),
                        'status': 'running'
                    }
                except (ValueError, IndexError):
                    pass

            elif current_cycle and "[Cycle #" in line:
                # Track symbols and phases
                if "Symbol:" in line:
                    symbol_part = line.split("Symbol: ")[1].split("]")[0]
                    cycle_summary[current_cycle]['symbols'].add(symbol_part)
                if "Phase:" in line:
                    phase_part = line.split("Phase: ")[1].split(",")[0].split("]")[0]
                    if phase_part not in cycle_summary[current_cycle]['phases']:
                        cycle_summary[current_cycle]['phases'].append(phase_part)

            elif current_cycle and "🎉 Pipeline completed" in line:
                cycle_summary[current_cycle]['status'] = 'completed'

            filtered_lines.append(line.rstrip())

        # Display cycle summary first
        if cycle_summary:
            print("📈 CYCLE SUMMARY")
            print("-" * 40)
            for cycle_id, info in sorted(cycle_summary.items(), reverse=True):
                status_icon = "✅" if info['status'] == 'completed' else "🔄"
                symbols_count = len(info['symbols'])
                phases_str = ", ".join(info['phases'][-3:])  # Show last 3 phases
                print(f"{status_icon} Cycle #{cycle_id}: {symbols_count} symbols, phases: {phases_str}")
            print()

        # Display filtered log lines
        if filtered_lines:
            print("📋 RECENT LOG ENTRIES")
            print("-" * 40)
            for line in filtered_lines[-50:]:  # Show last 50 entries
                # Clean up the log line for better readability
                if " | {" in line:
                    # Remove the JSON part for cleaner display
                    clean_line = line.split(" | {")[0]
                    print(clean_line)
                else:
                    print(line)
        else:
            print("ℹ️  No log entries found in the specified time range.")

        print("\n" + "=" * 80)
        print(f"📊 Total entries shown: {len(filtered_lines)}")
        if cycle_summary:
            active_cycles = sum(1 for c in cycle_summary.values() if c['status'] == 'running')
            completed_cycles = sum(1 for c in cycle_summary.values() if c['status'] == 'completed')
            print(f"🔄 Active cycles: {active_cycles}, Completed cycles: {completed_cycles}")

    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def main():
    parser = argparse.ArgumentParser(description="View trading pipeline logs")
    parser.add_argument("-H", "--hours", type=int, default=24,
                       help="Number of hours to look back (default: 24)")
    parser.add_argument("-f", "--follow", action="store_true",
                       help="Follow the log file (not implemented yet)")
    parser.add_argument("-p", "--phase", type=str,
                       help="Filter by specific phase (e.g., discovery, execution)")
    parser.add_argument("-s", "--symbol", type=str,
                       help="Filter by specific symbol")

    args = parser.parse_args()

    view_pipeline_logs(
        hours=args.hours,
        follow=args.follow,
        filter_phase=args.phase,
        filter_symbol=args.symbol
    )

if __name__ == "__main__":
    main()
