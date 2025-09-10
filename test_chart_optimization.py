#!/usr/bin/env python3
"""
Test script to verify the new batch chart loading optimization for Open Positions cards.
This tests the new /api/batch-chart-data endpoint and compares performance with individual requests.
"""

import requests
import time
import json
from typing import List, Dict

def test_batch_chart_data_api(symbols: List[str]) -> Dict:
    """Test the new batch chart data API endpoint."""
    print(f"🔄 Testing batch chart data API for {len(symbols)} symbols: {symbols}")
    
    start_time = time.time()
    
    try:
        # Test the new batch endpoint
        response = requests.get('http://localhost:8000/api/batch-chart-data', 
                              params={'symbols[]': symbols})
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch API successful in {duration:.2f}s")
            print(f"   - Fetched: {data.get('fetched_count', 0)} symbols")
            print(f"   - Cached: {data.get('cached_count', 0)} symbols")
            print(f"   - Total symbols: {len(data.get('chart_data', {}))}")
            return {
                'success': True,
                'duration': duration,
                'fetched_count': data.get('fetched_count', 0),
                'cached_count': data.get('cached_count', 0),
                'total_symbols': len(data.get('chart_data', {}))
            }
        else:
            print(f"❌ Batch API failed: {response.status_code} - {response.text}")
            return {'success': False, 'error': f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Batch API error: {e}")
        return {'success': False, 'error': str(e)}

def test_individual_chart_data_api(symbols: List[str]) -> Dict:
    """Test the old individual chart data API endpoint for comparison."""
    print(f"🔄 Testing individual chart data API for {len(symbols)} symbols")
    
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    
    for symbol in symbols:
        try:
            response = requests.get('http://localhost:8000/api/trend-data', 
                                  params={'symbol': symbol})
            
            if response.status_code == 200:
                successful_requests += 1
            else:
                failed_requests += 1
                print(f"   ❌ Failed for {symbol}: {response.status_code}")
                
        except Exception as e:
            failed_requests += 1
            print(f"   ❌ Error for {symbol}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Individual API completed in {duration:.2f}s")
    print(f"   - Successful: {successful_requests}")
    print(f"   - Failed: {failed_requests}")
    
    return {
        'success': successful_requests > 0,
        'duration': duration,
        'successful_requests': successful_requests,
        'failed_requests': failed_requests
    }

def test_open_positions_api() -> List[str]:
    """Test the open positions API to get real symbols."""
    print("🔄 Testing open positions API to get real symbols...")
    
    try:
        response = requests.get('http://localhost:8000/api/open-positions')
        
        if response.status_code == 200:
            positions = response.json()
            symbols = [pos['symbol'] for pos in positions if 'symbol' in pos]
            print(f"✅ Found {len(symbols)} open positions: {symbols}")
            return symbols
        else:
            print(f"❌ Open positions API failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Open positions API error: {e}")
        return []

def main():
    """Main test function."""
    print("🚀 Chart Loading Optimization Test")
    print("=" * 50)
    
    # Test with real open positions first
    real_symbols = test_open_positions_api()
    
    if real_symbols:
        print(f"\n📊 Testing with {len(real_symbols)} real symbols...")
        
        # Test batch API
        batch_result = test_batch_chart_data_api(real_symbols)
        
        # Test individual API for comparison
        individual_result = test_individual_chart_data_api(real_symbols)
        
        # Compare results
        print(f"\n📈 Performance Comparison:")
        print(f"   Batch API:     {batch_result['duration']:.2f}s")
        print(f"   Individual API: {individual_result['duration']:.2f}s")
        
        if batch_result['success'] and individual_result['success']:
            speedup = individual_result['duration'] / batch_result['duration']
            print(f"   Speedup:       {speedup:.1f}x faster")
            
            if speedup > 2:
                print("   ✅ Significant performance improvement achieved!")
            elif speedup > 1.5:
                print("   ✅ Good performance improvement achieved!")
            else:
                print("   ⚠️  Minimal performance improvement")
    
    # Test with mock symbols if no real positions
    if not real_symbols:
        print(f"\n📊 Testing with mock symbols...")
        mock_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'MATIC/USD', 'ALGO/USD']
        
        batch_result = test_batch_chart_data_api(mock_symbols)
        individual_result = test_individual_chart_data_api(mock_symbols)
        
        print(f"\n📈 Performance Comparison:")
        print(f"   Batch API:     {batch_result['duration']:.2f}s")
        print(f"   Individual API: {individual_result['duration']:.2f}s")
        
        if batch_result['success'] and individual_result['success']:
            speedup = individual_result['duration'] / batch_result['duration']
            print(f"   Speedup:       {speedup:.1f}x faster")
    
    print(f"\n✅ Chart optimization test completed!")

if __name__ == "__main__":
    main()
