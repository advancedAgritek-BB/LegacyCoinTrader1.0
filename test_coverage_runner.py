#!/usr/bin/env python3
"""
Test Coverage Runner and Analysis Tool

This script provides comprehensive test coverage analysis and reporting 
for the Legacy Coin Trader application.
"""

import subprocess
import sys
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class CoverageAnalyzer:
    """Analyze and report test coverage metrics."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.coverage_file = self.base_dir / "coverage.xml"
        self.html_dir = self.base_dir / "htmlcov"
        
    def run_tests_with_coverage(self, target_coverage: int = 70, 
                              include_slow: bool = False,
                              module_filter: str = None) -> Tuple[bool, Dict]:
        """Run tests with coverage analysis."""
        print("🧪 Running tests with coverage analysis...")
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=crypto_bot",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            f"--cov-fail-under={target_coverage}",
            "--cov-branch",
            "-v"
        ]
        
        # Add filters
        if not include_slow:
            cmd.extend(["-m", "not slow"])
            
        if module_filter:
            cmd.extend(["-k", module_filter])
            
        # Add test paths
        cmd.extend(["tests/", "crypto_bot/tools/"])
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.base_dir
            )
            
            success = result.returncode == 0
            
            return success, {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False, {"error": str(e)}
    
    def analyze_coverage_gaps(self) -> List[Dict]:
        """Analyze coverage gaps from XML report."""
        if not self.coverage_file.exists():
            print("⚠️  Coverage XML file not found. Run tests first.")
            return []
            
        try:
            tree = ET.parse(self.coverage_file)
            root = tree.getroot()
            
            gaps = []
            
            for package in root.findall(".//package"):
                package_name = package.get("name", "")
                
                for class_elem in package.findall("classes/class"):
                    filename = class_elem.get("filename", "")
                    coverage_rate = float(class_elem.get("line-rate", "0"))
                    
                    if coverage_rate < 0.7:  # Less than 70% coverage
                        lines_covered = int(class_elem.get("lines-covered", "0"))
                        lines_valid = int(class_elem.get("lines-valid", "1"))
                        
                        gaps.append({
                            "file": filename,
                            "package": package_name,
                            "coverage_rate": coverage_rate,
                            "lines_covered": lines_covered,
                            "lines_valid": lines_valid,
                            "lines_missing": lines_valid - lines_covered
                        })
            
            return sorted(gaps, key=lambda x: x["coverage_rate"])
            
        except Exception as e:
            print(f"❌ Error analyzing coverage: {e}")
            return []
    
    def find_untested_modules(self) -> List[str]:
        """Find modules with no tests."""
        crypto_bot_dir = self.base_dir / "crypto_bot"
        tests_dir = self.base_dir / "tests"
        
        untested = []
        
        for py_file in crypto_bot_dir.rglob("*.py"):
            if py_file.name.startswith("test_") or py_file.name == "__init__.py":
                continue
                
            # Check if corresponding test exists
            relative_path = py_file.relative_to(crypto_bot_dir)
            test_name = f"test_{py_file.stem}.py"
            
            test_paths = [
                tests_dir / test_name,
                tests_dir / relative_path.parent / test_name
            ]
            
            if not any(test_path.exists() for test_path in test_paths):
                untested.append(str(relative_path))
        
        return sorted(untested)
    
    def generate_coverage_report(self) -> Dict:
        """Generate comprehensive coverage report."""
        print("📊 Generating coverage report...")
        
        gaps = self.analyze_coverage_gaps()
        untested = self.find_untested_modules()
        
        # Calculate overall stats
        total_files = len(gaps) if gaps else 0
        critical_gaps = [g for g in gaps if g["coverage_rate"] < 0.3]
        moderate_gaps = [g for g in gaps if 0.3 <= g["coverage_rate"] < 0.7]
        
        report = {
            "summary": {
                "total_files_analyzed": total_files,
                "critical_gaps": len(critical_gaps),
                "moderate_gaps": len(moderate_gaps),
                "untested_modules": len(untested)
            },
            "critical_gaps": critical_gaps[:10],  # Top 10 worst
            "moderate_gaps": moderate_gaps[:15],  # Top 15 moderate
            "untested_modules": untested[:20],    # Top 20 untested
            "recommendations": self._generate_recommendations(gaps, untested)
        }
        
        return report
    
    def _generate_recommendations(self, gaps: List[Dict], 
                                untested: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if untested:
            recommendations.append(
                f"🎯 Create tests for {len(untested)} untested modules, "
                f"starting with core modules like {', '.join(untested[:3])}"
            )
        
        critical_files = [g["file"] for g in gaps if g["coverage_rate"] < 0.3]
        if critical_files:
            recommendations.append(
                f"🚨 Immediately improve coverage for critical files: "
                f"{', '.join(critical_files[:3])}"
            )
        
        if gaps:
            high_impact = [g for g in gaps if g["lines_missing"] > 100]
            if high_impact:
                recommendations.append(
                    f"📈 Focus on high-impact files with 100+ missing lines: "
                    f"{', '.join([g['file'] for g in high_impact[:3]])}"
                )
        
        recommendations.extend([
            "🔧 Fix import and module path issues in existing tests",
            "🧪 Add integration tests for end-to-end workflows", 
            "⚡ Create performance tests for critical trading functions",
            "🛡️ Add error handling and edge case tests"
        ])
        
        return recommendations
    
    def print_report(self, report: Dict):
        """Print formatted coverage report."""
        print("\n" + "="*80)
        print("📋 TEST COVERAGE ANALYSIS REPORT")
        print("="*80)
        
        summary = report["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   • Total files analyzed: {summary['total_files_analyzed']}")
        print(f"   • Critical coverage gaps (<30%): {summary['critical_gaps']}")
        print(f"   • Moderate coverage gaps (30-70%): {summary['moderate_gaps']}")
        print(f"   • Untested modules: {summary['untested_modules']}")
        
        if report["critical_gaps"]:
            print(f"\n🚨 CRITICAL COVERAGE GAPS:")
            for gap in report["critical_gaps"]:
                coverage_pct = gap["coverage_rate"] * 100
                print(f"   • {gap['file']}: {coverage_pct:.1f}% "
                      f"({gap['lines_missing']} lines missing)")
        
        if report["untested_modules"]:
            print(f"\n🎯 UNTESTED MODULES (first 10):")
            for module in report["untested_modules"][:10]:
                print(f"   • {module}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🌐 Detailed HTML report: file://{self.html_dir.absolute()}/index.html")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Coverage Analysis Tool")
    parser.add_argument("--target", type=int, default=70,
                       help="Target coverage percentage (default: 70)")
    parser.add_argument("--include-slow", action="store_true",
                       help="Include slow tests")
    parser.add_argument("--module", type=str,
                       help="Filter tests by module/pattern")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report without running tests")
    
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer()
    
    if not args.report_only:
        print("🚀 Starting test coverage analysis...")
        success, result = analyzer.run_tests_with_coverage(
            target_coverage=args.target,
            include_slow=args.include_slow,
            module_filter=args.module
        )
        
        if not success:
            print(f"⚠️  Tests failed, but generating coverage report anyway...")
            if "stdout" in result:
                print("Test output:", result["stdout"][-500:])  # Last 500 chars
    
    # Generate and display report
    report = analyzer.generate_coverage_report()
    analyzer.print_report(report)
    
    # Save report to JSON
    report_file = Path("coverage_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file.absolute()}")


if __name__ == "__main__":
    main()
