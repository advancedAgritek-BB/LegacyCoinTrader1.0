#!/usr/bin/env python3
"""
Asset Score Generator Tool

This tool generates sample asset scores for demonstration purposes
and populates the asset_scores.json file that the Scans page reads.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import LOG_DIR


def generate_sample_scores() -> Dict[str, float]:
    """Generate sample asset scores for demonstration."""
    
    # Common cryptocurrency symbols
    symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD", "AVAX/USD",
        "ADA/USD", "DOT/USD", "LINK/USD", "UNI/USD", "AAVE/USD",
        "COMP/USD", "MKR/USD", "SNX/USD", "YFI/USD", "CRV/USD",
        "BAL/USD", "REN/USD", "KNC/USD", "ZRX/USD", "BAND/USD"
    ]
    
    scores = {}
    for symbol in symbols:
        # Generate realistic-looking scores between -0.5 and 2.0
        # Positive scores indicate good performance, negative indicate poor performance
        score = random.uniform(-0.5, 2.0)
        scores[symbol] = round(score, 4)
    
    # Sort by score (highest first)
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_scores


def save_asset_scores(scores: Dict[str, float]) -> None:
    """Save asset scores to the asset_scores.json file."""
    
    score_file = LOG_DIR / "asset_scores.json"
    
    # Ensure the directory exists
    score_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the scores
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"Asset scores saved to: {score_file}")
    print(f"Generated {len(scores)} asset scores")


def main():
    """Main function to generate and save asset scores."""
    
    print("Generating sample asset scores...")
    
    try:
        # Generate sample scores
        scores = generate_sample_scores()
        
        # Save to file
        save_asset_scores(scores)
        
        # Display top 10 scores
        print("\nTop 10 Asset Scores:")
        print("-" * 30)
        for i, (symbol, score) in enumerate(list(scores.items())[:10], 1):
            print(f"{i:2d}. {symbol:<12} {score:>8.4f}")
        
        print(f"\nTotal assets scored: {len(scores)}")
        print("The Scans page should now display these scores!")
        
    except Exception as e:
        print(f"Error generating asset scores: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
