#!/usr/bin/env python3
"""
Comprehensive test of Raydium integration and comparison with other APIs.
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any

async def test_all_apis():
    """Test all available Solana APIs for pool discovery."""
    print("🚀 Comprehensive Solana API Comparison Test")
    print("=" * 60)

    results = {}

    # Test Raydium API
    print("\n🔍 Testing Raydium API...")
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.raydium.io/pairs', timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raydium_pools = len(data) if isinstance(data, list) else 0

                    # Extract some tokens
                    tokens = []
                    seen = set()
                    sorted_pools = sorted(data, key=lambda x: x.get('liquidity', 0), reverse=True)

                    for pool in sorted_pools[:50]:
                        pair_id = pool.get('pair_id', '')
                        if '-' in pair_id:
                            token_a, token_b = pair_id.split('-', 1)
                            for token in [token_a, token_b]:
                                if token and token not in ['So11111111111111111111111111111111111111112', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'] and token not in seen:
                                    tokens.append(token)
                                    seen.add(token)
                                    if len(tokens) >= 20:
                                        break
                            if len(tokens) >= 20:
                                break

                    results['Raydium'] = {
                        'pools': raydium_pools,
                        'tokens': len(tokens),
                        'sample_tokens': tokens[:5],
                        'response_time': time.time() - start_time,
                        'status': '✅ Working'
                    }
                else:
                    results['Raydium'] = {'status': f'❌ HTTP {resp.status}', 'response_time': time.time() - start_time}
    except Exception as e:
        results['Raydium'] = {'status': f'❌ Error: {str(e)[:50]}...', 'response_time': time.time() - start_time}

    # Test Orca API
    print("\n🔍 Testing Orca API...")
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://www.orca.so/api/pools', timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    orca_pools = len(data) if isinstance(data, list) else 0

                    tokens = []
                    if isinstance(data, list):
                        for pool in data[:20]:
                            token_a = pool.get('tokenA', {}).get('mint') or pool.get('tokenAMint')
                            token_b = pool.get('tokenB', {}).get('mint') or pool.get('tokenBMint')
                            for token in [token_a, token_b]:
                                if token:
                                    tokens.append(token)

                    results['Orca'] = {
                        'pools': orca_pools,
                        'tokens': len(tokens),
                        'sample_tokens': tokens[:3],
                        'response_time': time.time() - start_time,
                        'status': '✅ Working' if orca_pools > 0 else '⚠️ Limited data'
                    }
                else:
                    results['Orca'] = {'status': f'❌ HTTP {resp.status}', 'response_time': time.time() - start_time}
    except Exception as e:
        results['Orca'] = {'status': f'❌ Error: {str(e)[:50]}...', 'response_time': time.time() - start_time}

    # Test Jupiter API
    print("\n🔍 Testing Jupiter API...")
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://token.jup.ag/all', timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    jupiter_tokens = len(data) if isinstance(data, list) else 0

                    solana_tokens = [t for t in data if isinstance(t, dict) and t.get('chainId') == 101] if isinstance(data, list) else []

                    results['Jupiter'] = {
                        'pools': 'N/A (token list)',
                        'tokens': len(solana_tokens),
                        'sample_tokens': [t.get('address', '') for t in solana_tokens[:3] if t.get('address')],
                        'response_time': time.time() - start_time,
                        'status': '✅ Working' if jupiter_tokens > 0 else '❌ No data'
                    }
                else:
                    results['Jupiter'] = {'status': f'❌ HTTP {resp.status}', 'response_time': time.time() - start_time}
    except Exception as e:
        results['Jupiter'] = {'status': f'❌ Error: {str(e)[:50]}...', 'response_time': time.time() - start_time}

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("📊 SOLANA API COMPARISON RESULTS")
    print("=" * 80)

    print(f"{'API':<15} {'Status':<20} {'Pools':<15} {'Tokens':<10} {'Response Time':<15}")
    print("-" * 80)

    for api_name, data in results.items():
        status = data.get('status', '❓ Unknown')
        response_time = data.get('response_time', 0)
        pools = data.get('pools', 'N/A')
        tokens = data.get('tokens', 0)

        print(f"{api_name:<15} {status:<20} {str(pools):<15} {tokens:<10} {response_time:.2f}s")
    print("-" * 80)

    # Summary and recommendations
    print("\n🎯 SUMMARY & RECOMMENDATIONS")
    print("-" * 80)

    working_apis = [api for api, data in results.items() if '✅' in data.get('status', '')]

    print(f"✅ Working APIs: {', '.join(working_apis)}")
    print(f"📊 Total APIs tested: {len(results)}")

    if 'Raydium' in working_apis:
        raydium_data = results['Raydium']
        print("\n🎯 RECOMMENDATION: Use Raydium as primary source")
        print(f"   • {raydium_data['pools']:,} pools available")
        print(f"   • {raydium_data['tokens']} tokens extracted")
        print(f"   • Response time: {raydium_data['response_time']:.2f}s")
        print("   • Free and reliable!")

    print("\n🔄 Fallback order:")
    priority = ['Raydium', 'Orca', 'Jupiter']
    for i, api in enumerate(priority, 1):
        if api in working_apis:
            print(f"   {i}. {api} ✅")
        else:
            print(f"   {i}. {api} ❌")

    print("\n💡 The scanner automatically uses the best available source!")
    return results

if __name__ == "__main__":
    asyncio.run(test_all_apis())
