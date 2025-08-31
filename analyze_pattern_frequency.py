#!/usr/bin/env python3
"""
Pattern Frequency Analysis Script

This script analyzes the pattern_frequency.csv file to determine:
1. Which strategies are actually being used
2. Pattern distribution across regimes
3. Potential problems with strategy utilization
4. Recommendations for optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternFrequencyAnalyzer:
    def __init__(self, csv_path="crypto_bot/logs/pattern_frequency.csv"):
        self.csv_path = Path(csv_path)
        self.data = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and preprocess the pattern frequency data."""
        logger.info(f"Loading data from {self.csv_path}")
        
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return False
            
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.data)} records")
            
            # Convert timestamp to datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            # Remove header row if it exists
            if self.data.iloc[0]['regime'] == 'regime':
                self.data = self.data.iloc[1:].reset_index(drop=True)
                
            # Remove any rows with invalid regime or pattern
            self.data = self.data[
                (self.data['regime'] != 'regime') & 
                (self.data['pattern'] != 'pattern') &
                (self.data['regime'].notna()) &
                (self.data['pattern'].notna())
            ]
            
            logger.info(f"Cleaned data: {len(self.data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_regime_distribution(self):
        """Analyze the distribution of regimes."""
        logger.info("Analyzing regime distribution...")
        
        regime_counts = self.data['regime'].value_counts()
        regime_percentages = (regime_counts / len(self.data) * 100).round(2)
        
        self.analysis_results['regime_distribution'] = {
            'counts': regime_counts.to_dict(),
            'percentages': regime_percentages.to_dict(),
            'total_records': len(self.data)
        }
        
        logger.info("Regime distribution:")
        for regime, count in regime_counts.items():
            pct = regime_percentages[regime]
            logger.info(f"  {regime}: {count} ({pct}%)")
    
    def analyze_pattern_distribution(self):
        """Analyze the distribution of patterns."""
        logger.info("Analyzing pattern distribution...")
        
        pattern_counts = self.data['pattern'].value_counts()
        pattern_percentages = (pattern_counts / len(self.data) * 100).round(2)
        
        self.analysis_results['pattern_distribution'] = {
            'counts': pattern_counts.to_dict(),
            'percentages': pattern_percentages.to_dict()
        }
        
        logger.info("Top 10 patterns:")
        for pattern, count in pattern_counts.head(10).items():
            pct = pattern_percentages[pattern]
            logger.info(f"  {pattern}: {count} ({pct}%)")
    
    def analyze_regime_pattern_combinations(self):
        """Analyze combinations of regimes and patterns."""
        logger.info("Analyzing regime-pattern combinations...")
        
        combo_counts = self.data.groupby(['regime', 'pattern']).size().sort_values(ascending=False)
        
        self.analysis_results['regime_pattern_combinations'] = {
            'top_combinations': combo_counts.head(20).to_dict(),
            'total_unique_combinations': len(combo_counts)
        }
        
        logger.info("Top 10 regime-pattern combinations:")
        for (regime, pattern), count in combo_counts.head(10).items():
            logger.info(f"  {regime} + {pattern}: {count}")
    
    def analyze_strategy_utilization(self):
        """Analyze which strategies are actually being used based on regime mapping."""
        logger.info("Analyzing strategy utilization...")
        
        # Strategy mapping from config
        strategy_mapping = {
            'bounce': ['bounce_scalper'],
            'breakout': ['breakout_bot'],
            'mean-reverting': ['mean_bot', 'dip_hunter', 'stat_arb_bot'],
            'scalp': ['micro_scalp', 'micro_scalp_bot'],
            'sideways': ['grid', 'maker_spread', 'range_arb_bot'],
            'trending': ['trend', 'trend_bot', 'momentum_bot', 'lstm_bot'],
            'volatile': ['sniper_bot', 'sniper_solana', 'flash_crash_bot', 'meme_wave_bot', 'hft_engine']
        }
        
        # Count regime occurrences
        regime_counts = self.data['regime'].value_counts()
        
        # Calculate strategy utilization
        strategy_usage = {}
        for regime, strategies in strategy_mapping.items():
            if regime in regime_counts:
                count = regime_counts[regime]
                strategy_usage[regime] = {
                    'count': count,
                    'strategies': strategies,
                    'strategy_count': len(strategies)
                }
        
        self.analysis_results['strategy_utilization'] = strategy_usage
        
        logger.info("Strategy utilization by regime:")
        for regime, info in strategy_usage.items():
            logger.info(f"  {regime}: {info['count']} occurrences -> {info['strategies']}")
    
    def identify_potential_problems(self):
        """Identify potential problems with strategy usage."""
        logger.info("Identifying potential problems...")
        
        problems = []
        
        # Check for regime imbalance
        regime_counts = self.data['regime'].value_counts()
        total = len(self.data)
        
        # Check if any regime is over-represented (>50%)
        for regime, count in regime_counts.items():
            percentage = (count / total) * 100
            if percentage > 50:
                problems.append({
                    'type': 'regime_imbalance',
                    'regime': regime,
                    'percentage': percentage,
                    'description': f"{regime} regime dominates with {percentage:.1f}% of all patterns"
                })
        
        # Check for pattern over-reliance
        pattern_counts = self.data['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            percentage = (count / total) * 100
            if percentage > 30:
                problems.append({
                    'type': 'pattern_over_reliance',
                    'pattern': pattern,
                    'percentage': percentage,
                    'description': f"Pattern '{pattern}' over-relied upon with {percentage:.1f}% frequency"
                })
        
        # Check for unused strategies
        strategy_mapping = {
            'bounce': ['bounce_scalper'],
            'breakout': ['breakout_bot'],
            'mean-reverting': ['mean_bot', 'dip_hunter', 'stat_arb_bot'],
            'scalp': ['micro_scalp', 'micro_scalp_bot'],
            'sideways': ['grid', 'maker_spread', 'range_arb_bot'],
            'trending': ['trend', 'trend_bot', 'momentum_bot', 'lstm_bot'],
            'volatile': ['sniper_bot', 'sniper_solana', 'flash_crash_bot', 'meme_wave_bot', 'hft_engine']
        }
        
        for regime, strategies in strategy_mapping.items():
            if regime not in regime_counts or regime_counts[regime] < 100:
                for strategy in strategies:
                    problems.append({
                        'type': 'unused_strategy',
                        'strategy': strategy,
                        'regime': regime,
                        'count': regime_counts.get(regime, 0),
                        'description': f"Strategy '{strategy}' for '{regime}' regime has only {regime_counts.get(regime, 0)} occurrences"
                    })
        
        # Check for unknown regimes
        known_regimes = set(strategy_mapping.keys())
        unknown_regimes = set(regime_counts.index) - known_regimes
        for regime in unknown_regimes:
            problems.append({
                'type': 'unknown_regime',
                'regime': regime,
                'count': regime_counts[regime],
                'description': f"Unknown regime '{regime}' detected with {regime_counts[regime]} occurrences"
            })
        
        self.analysis_results['problems'] = problems
        
        logger.info(f"Identified {len(problems)} potential problems:")
        for problem in problems:
            logger.info(f"  {problem['type']}: {problem['description']}")
    
    def generate_recommendations(self):
        """Generate recommendations based on the analysis."""
        logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Check regime balance
        regime_counts = self.data['regime'].value_counts()
        total = len(self.data)
        
        for regime, count in regime_counts.items():
            percentage = (count / total) * 100
            if percentage < 5:
                recommendations.append({
                    'type': 'increase_regime_usage',
                    'regime': regime,
                    'current_percentage': percentage,
                    'description': f"Consider increasing usage of '{regime}' regime (currently {percentage:.1f}%)"
                })
            elif percentage > 60:
                recommendations.append({
                    'type': 'reduce_regime_dominance',
                    'regime': regime,
                    'current_percentage': percentage,
                    'description': f"Consider reducing dominance of '{regime}' regime (currently {percentage:.1f}%)"
                })
        
        # Check pattern diversity
        pattern_counts = self.data['pattern'].value_counts()
        unique_patterns = len(pattern_counts)
        
        if unique_patterns < 10:
            recommendations.append({
                'type': 'increase_pattern_diversity',
                'current_patterns': unique_patterns,
                'description': f"Consider increasing pattern diversity (currently {unique_patterns} unique patterns)"
            })
        
        # Check for high-strength patterns
        high_strength = self.data[self.data['strength'] > 0.8]
        if len(high_strength) < len(self.data) * 0.1:
            recommendations.append({
                'type': 'improve_pattern_strength',
                'high_strength_percentage': (len(high_strength) / len(self.data)) * 100,
                'description': f"Only {len(high_strength) / len(self.data) * 100:.1f}% of patterns have high strength (>0.8)"
            })
        
        self.analysis_results['recommendations'] = recommendations
        
        logger.info(f"Generated {len(recommendations)} recommendations:")
        for rec in recommendations:
            logger.info(f"  {rec['type']}: {rec['description']}")
    
    def create_visualizations(self, output_dir="analysis_output"):
        """Create visualizations of the analysis."""
        logger.info("Creating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Regime distribution pie chart
        plt.figure(figsize=(10, 8))
        regime_counts = self.data['regime'].value_counts()
        plt.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        plt.title('Regime Distribution')
        plt.savefig(output_path / 'regime_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Pattern distribution bar chart
        plt.figure(figsize=(12, 8))
        pattern_counts = self.data['pattern'].value_counts().head(15)
        pattern_counts.plot(kind='bar')
        plt.title('Top 15 Pattern Distribution')
        plt.xlabel('Pattern')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'pattern_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Regime-pattern heatmap
        plt.figure(figsize=(12, 8))
        pivot_table = self.data.groupby(['regime', 'pattern']).size().unstack(fill_value=0)
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Regime-Pattern Heatmap')
        plt.tight_layout()
        plt.savefig(output_path / 'regime_pattern_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Strength distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['strength'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Pattern Strength Distribution')
        plt.xlabel('Strength')
        plt.ylabel('Frequency')
        plt.axvline(self.data['strength'].mean(), color='red', linestyle='--', label=f"Mean: {self.data['strength'].mean():.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'strength_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def export_analysis_report(self, output_file="pattern_analysis_report.json"):
        """Export the analysis results to a JSON file."""
        logger.info(f"Exporting analysis report to {output_file}")
        
        # Add metadata
        self.analysis_results['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'data_file': str(self.csv_path),
            'total_records': len(self.data),
            'date_range': {
                'start': self.data['timestamp'].min().isoformat(),
                'end': self.data['timestamp'].max().isoformat()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis report exported to {output_file}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting full pattern frequency analysis...")
        
        if not self.load_data():
            return False
        
        self.analyze_regime_distribution()
        self.analyze_pattern_distribution()
        self.analyze_regime_pattern_combinations()
        self.analyze_strategy_utilization()
        self.identify_potential_problems()
        self.generate_recommendations()
        self.create_visualizations()
        self.export_analysis_report()
        
        logger.info("Analysis complete!")
        return True

def main():
    """Main function to run the analysis."""
    analyzer = PatternFrequencyAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
