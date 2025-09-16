from __future__ import annotations

"""Utilities for scanning new Solana tokens."""

import asyncio
import logging
import os
import time
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import aiohttp

logger = logging.getLogger(__name__)


_JUPITER_TOKEN_LIST_URL = "https://token.jup.ag/all"
_SYMBOL_TO_MINT_CACHE: Dict[str, str] = {}
_SYMBOL_TO_MINT_MISS_TS: Dict[str, float] = {}
_SYMBOL_TO_MINT_CACHE_TTL = 900  # seconds to wait before retrying unresolved symbols
_STATIC_SYMBOL_TO_MINT: Dict[str, str] = {
    "SOL": "So11111111111111111111111111111111111111112",
    "WSOL": "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "PYTH": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "MSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
    "JTO": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
    "JLP": "27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4",
    "SAMO": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
}


def _normalize_symbol(symbol: str) -> str:
    """Normalize a token symbol for mapping lookups."""
    return str(symbol or "").upper().strip()


def _load_symbol_to_mint_overrides(cfg: Mapping[str, object]) -> Dict[str, str]:
    """Extract custom symbol-to-mint overrides from the configuration."""
    overrides = cfg.get("symbol_to_mint_map")
    result: Dict[str, str] = {}
    if isinstance(overrides, Mapping):
        for symbol, mint in overrides.items():
            normalized = _normalize_symbol(symbol)
            if not normalized or not mint:
                continue
            result[normalized] = str(mint)
    return result


async def _fetch_symbol_mints_from_jupiter(symbols: Set[str]) -> Dict[str, str]:
    """Fetch mint addresses for the provided symbols using Jupiter's token list."""
    if not symbols:
        return {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(_JUPITER_TOKEN_LIST_URL, timeout=15) as resp:
                if resp.status != 200:
                    logger.debug("Jupiter token list request failed with status %s", resp.status)
                    return {}
                data = await resp.json()
    except Exception as exc:
        logger.debug("Failed to fetch Jupiter token list: %s", exc)
        return {}

    if not isinstance(data, list):
        logger.debug("Unexpected Jupiter token list payload: %s", type(data))
        return {}

    mapping: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        symbol_value = entry.get("symbol")
        address = entry.get("address")
        chain_id = entry.get("chainId")
        if not symbol_value or not address or chain_id != 101:
            continue
        normalized = _normalize_symbol(symbol_value)
        if normalized in symbols and normalized not in mapping:
            mapping[normalized] = str(address)
        if len(mapping) == len(symbols):
            break

    return mapping


async def _get_symbol_to_mint_map(
    cfg: Mapping[str, object],
    symbols: Iterable[str],
) -> Dict[str, str]:
    """Resolve mint addresses for the provided symbols."""
    normalized_symbols = {_normalize_symbol(symbol) for symbol in symbols if symbol}
    mapping = dict(_STATIC_SYMBOL_TO_MINT)
    mapping.update(_load_symbol_to_mint_overrides(cfg))

    now = time.time()
    remaining: List[str] = []
    for symbol in normalized_symbols:
        if symbol in mapping:
            continue
        cached = _SYMBOL_TO_MINT_CACHE.get(symbol)
        if cached:
            mapping[symbol] = cached
            continue
        miss_ts = _SYMBOL_TO_MINT_MISS_TS.get(symbol)
        if miss_ts and now - miss_ts < _SYMBOL_TO_MINT_CACHE_TTL:
            continue
        remaining.append(symbol)

    if remaining:
        fetched = await _fetch_symbol_mints_from_jupiter(set(remaining))
        resolved: Set[str] = set()
        if fetched:
            for symbol, mint in fetched.items():
                mapping[symbol] = mint
                _SYMBOL_TO_MINT_CACHE[symbol] = mint
                resolved.add(symbol)

        unresolved = set(remaining) - resolved
        if unresolved:
            miss_time = time.time()
            for symbol in unresolved:
                _SYMBOL_TO_MINT_MISS_TS[symbol] = miss_time
            logger.debug(
                "Missing Solana mint mapping for symbols: %s",
                ", ".join(sorted(unresolved)),
            )

    return mapping


async def get_solana_new_tokens(cfg: Mapping[str, object]) -> List[str]:
    """Return a list of new token mint addresses using multiple API sources."""

    limit = int(cfg.get("limit", 20))
    key = os.getenv("HELIUS_KEY", "")
    url = str(cfg.get("url", ""))

    # Replace API key placeholder
    if url and "${HELIUS_KEY}" in url:
        url = url.replace("${HELIUS_KEY}", key)
    if url and "YOUR_KEY" in url:
        url = url.replace("YOUR_KEY", key)

    # Method 1: Try Helius dex.getNewPools (premium)
    if url:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "dex.getNewPools",
                    "params": {
                        "protocols": ["raydium"],
                        "limit": limit
                    }
                }
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "error" not in data:
                            result = data.get("result", [])
                            results: List[str] = []
                            if isinstance(result, list):
                                for pool in result:
                                    if isinstance(pool, Mapping):
                                        token_a = pool.get("tokenA") or pool.get("tokenAMint")
                                        token_b = pool.get("tokenB") or pool.get("tokenBMint")
                                        if token_a and isinstance(token_a, str):
                                            results.append(token_a)
                                        if token_b and isinstance(token_b, str):
                                            results.append(token_b)
                            if results:
                                logger.info(f"Helius returned {len(results)} tokens")
                                return list(set(results))[:limit]
        except Exception as exc:
            logger.debug(f"Helius dex.getNewPools failed: {exc}")

    # Method 2: Try Raydium API (free alternative)
    logger.info("Trying Raydium API for pool discovery")
    try:
        raydium_url = "https://api.raydium.io/pairs"
        async with aiohttp.ClientSession() as session:
            async with session.get(raydium_url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and data:
                        results: List[str] = []
                        # Sort by liquidity and get recent pools
                        sorted_pools = sorted(data, key=lambda x: x.get("liquidity", 0), reverse=True)

                        for pool in sorted_pools[:limit * 2]:  # Get more to filter
                            # Extract token mints from pair_id (format: TOKENA-TOKENB)
                            pair_id = pool.get("pair_id", "")
                            if "-" in pair_id and len(pair_id.split("-")) >= 2:
                                token_a, token_b = pair_id.split("-", 1)
                                # Filter out WSOL duplicates and common tokens
                                common_tokens = ["So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]
                                if token_a and token_a not in common_tokens:
                                    results.append(token_a)
                                if token_b and token_b not in common_tokens:
                                    results.append(token_b)

                        if results:
                            logger.info(f"Raydium API returned {len(results)} tokens")
                            return list(set(results))[:limit]
    except Exception as exc:
        logger.debug(f"Raydium API failed: {exc}")

    # Method 3: Try Orca API
    logger.info("Trying Orca API for pool discovery")
    try:
        orca_url = "https://www.orca.so/api/pools"
        async with aiohttp.ClientSession() as session:
            async with session.get(orca_url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results: List[str] = []
                    if isinstance(data, list):
                        for pool in data[:limit * 2]:
                            if isinstance(pool, Mapping):
                                token_a = pool.get("tokenA") or pool.get("tokenAMint")
                                token_b = pool.get("tokenB") or pool.get("tokenBMint")
                                if token_a and isinstance(token_a, str):
                                    results.append(token_a)
                                if token_b and isinstance(token_b, str):
                                    results.append(token_b)
                    if results:
                        logger.info(f"Orca API returned {len(results)} tokens")
                        return list(set(results))[:limit]
    except Exception as exc:
        logger.debug(f"Orca API failed: {exc}")

    # Method 4: Try Jupiter API (different endpoint)
    logger.info("Trying Jupiter API for token discovery")
    try:
        jupiter_url = "https://token.jup.ag/all"
        async with aiohttp.ClientSession() as session:
            async with session.get(jupiter_url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and data:
                        # Filter for Solana tokens and sort by recent activity
                        solana_tokens = [token for token in data if token.get("chainId") == 101]
                        # Sort by daily volume or other activity metric if available
                        results = [token.get("address") for token in solana_tokens[:limit] if token.get("address")]
                        if results:
                            logger.info(f"Jupiter API returned {len(results)} tokens")
                            return results
    except Exception as exc:
        logger.debug(f"Jupiter API failed: {exc}")

    # Method 5: Try pump.fun API with wallet evaluation (if API key is available)
    pump_key = os.getenv("PUMPFUN_API_KEY") or cfg.get("pump_fun_api_key")
    if pump_key and pump_key != "YOUR_KEY":  # Make sure we have a real key
        logger.info("Trying pump.fun API for new launches with wallet evaluation")
        try:
            pump_url = f"https://client-api.prod.pump.fun/v1/launches?api-key={pump_key}&limit={limit * 2}"  # Get more for filtering
            async with aiohttp.ClientSession() as session:
                async with session.get(pump_url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            # Evaluate each launch based on wallet credibility
                            evaluated_launches = await evaluate_pump_fun_launches(data, session)
                            # Filter launches based on credibility score
                            credible_launches = [launch for launch in evaluated_launches if launch.get("credibility_score", 0) >= 60]
                            results = [launch["mint"] for launch in credible_launches[:limit] if launch.get("mint")]

                            if results:
                                logger.info(f"pump.fun API returned {len(results)} credible tokens after wallet evaluation")
                                return results
        except Exception as exc:
            logger.debug(f"pump.fun API failed: {exc}")

    # Final fallback: Return popular tokens
    logger.warning("All API methods failed, returning popular tokens")
    fallback_tokens = [
        "So11111111111111111111111111111111111111112",  # Wrapped SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        "mSoLzYCxHdYgdzU16g5QSh3iqbVNLAcP3H5AujTZg",  # Marinade SOL
        "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7AR",  # stSOL
    ]
    return fallback_tokens[:limit]


async def evaluate_pump_fun_launches(launches: List[Mapping], session: aiohttp.ClientSession) -> List[Dict]:
    """Evaluate pump.fun launches based on wallet credibility and history.

    Returns list of launches with added credibility scores.
    """
    evaluated_launches = []

    for launch in launches:
        try:
            launch_data = dict(launch)  # Copy the launch data

            # Extract key information
            mint = launch.get("mint") or launch.get("tokenMint") or launch.get("token_mint")
            creator_wallet = launch.get("creator") or launch.get("developer") or launch.get("wallet")
            bonding_curve = launch.get("bonding_curve", {})
            market_cap = launch.get("market_cap", 0)
            replies = launch.get("replies", 0)
            timestamp = launch.get("created_timestamp") or launch.get("timestamp")

            if not mint or not creator_wallet:
                launch_data["credibility_score"] = 0
                launch_data["evaluation_reason"] = "Missing mint or creator wallet"
                evaluated_launches.append(launch_data)
                continue

            # Calculate base credibility score
            credibility_score = 50  # Start with neutral score
            evaluation_factors = []

            # Factor 1: Bonding curve progress (more complete = more credible)
            curve_progress = bonding_curve.get("progress", 0)
            if curve_progress > 80:
                credibility_score += 15
                evaluation_factors.append("High bonding curve completion")
            elif curve_progress > 50:
                credibility_score += 5
                evaluation_factors.append("Moderate bonding curve completion")
            else:
                credibility_score -= 10
                evaluation_factors.append("Low bonding curve completion")

            # Factor 2: Community engagement (replies indicate interest)
            if replies > 100:
                credibility_score += 10
                evaluation_factors.append("High community engagement")
            elif replies > 20:
                credibility_score += 5
                evaluation_factors.append("Moderate community engagement")
            else:
                evaluation_factors.append("Low community engagement")

            # Factor 3: Market cap (reasonable initial market cap)
            if 1000 <= market_cap <= 50000:  # Sweet spot for legitimate launches
                credibility_score += 10
                evaluation_factors.append("Reasonable initial market cap")
            elif market_cap < 500:
                credibility_score -= 15
                evaluation_factors.append("Very low market cap (suspicious)")
            elif market_cap > 100000:
                credibility_score -= 5
                evaluation_factors.append("Very high market cap (might be pre-pumped)")

            # Factor 4: Creator wallet analysis
            wallet_score, wallet_factors = await evaluate_creator_wallet(creator_wallet, session)
            credibility_score += wallet_score
            evaluation_factors.extend(wallet_factors)

            # Factor 5: Time-based analysis (avoid very new launches that might be scams)
            if timestamp:
                from datetime import datetime
                try:
                    # Convert timestamp if it's a string
                    if isinstance(timestamp, str):
                        # Handle different timestamp formats
                        if timestamp.isdigit():
                            launch_time = datetime.fromtimestamp(int(timestamp))
                        else:
                            launch_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        launch_time = datetime.fromtimestamp(timestamp)

                    time_diff = datetime.now() - launch_time
                    hours_old = time_diff.total_seconds() / 3600

                    if hours_old < 1:
                        credibility_score -= 20
                        evaluation_factors.append("Very new launch (< 1 hour)")
                    elif hours_old < 6:
                        credibility_score -= 5
                        evaluation_factors.append("Recent launch (< 6 hours)")
                    elif hours_old > 24:
                        credibility_score += 5
                        evaluation_factors.append("Established launch (> 24 hours)")
                except Exception as e:
                    logger.debug(f"Could not parse timestamp: {e}")

            # Ensure score stays within bounds
            credibility_score = max(0, min(100, credibility_score))

            launch_data.update({
                "credibility_score": credibility_score,
                "evaluation_factors": evaluation_factors,
                "evaluation_reason": f"Score: {credibility_score}/100",
                "mint": mint,
                "creator_wallet": creator_wallet
            })

            evaluated_launches.append(launch_data)

        except Exception as e:
            logger.debug(f"Error evaluating launch {launch.get('mint', 'unknown')}: {e}")
            launch_data = dict(launch)
            launch_data["credibility_score"] = 0
            launch_data["evaluation_reason"] = f"Evaluation error: {e}"
            evaluated_launches.append(launch_data)

    # Sort by credibility score (highest first)
    evaluated_launches.sort(key=lambda x: x.get("credibility_score", 0), reverse=True)
    return evaluated_launches


async def evaluate_creator_wallet(wallet_address: str, session: aiohttp.ClientSession) -> Tuple[int, List[str]]:
    """Evaluate a creator wallet's credibility based on their launch history.

    Returns (score_adjustment, evaluation_factors)
    """
    try:
        # Check if this wallet has been involved in other launches
        history_score = 0
        factors = []

        # Try to get wallet's token creation history
        # This is a simplified version - in production you'd want more comprehensive analysis

        # Factor 1: Check wallet age and activity (using basic heuristics)
        # For now, we'll use simple heuristics based on wallet address patterns
        if len(wallet_address) == 44 and wallet_address.endswith("pump"):  # Common pump.fun pattern
            history_score += 10
            factors.append("Uses standard pump.fun wallet pattern")

        # Factor 2: Check for suspicious patterns
        suspicious_patterns = ["rug", "scam", "fake", "test"]
        wallet_lower = wallet_address.lower()
        if any(pattern in wallet_lower for pattern in suspicious_patterns):
            history_score -= 10
            factors.append("Wallet contains suspicious keywords")

        # Factor 3: Basic wallet validation
        if not wallet_address.startswith("111111") and len(wallet_address) >= 40:
            history_score += 15
            factors.append("Valid wallet format")
        elif len(wallet_address) >= 32:  # More lenient for test wallets
            history_score += 5
            factors.append("Reasonable wallet format")
        else:
            history_score -= 5
            factors.append("Invalid or suspicious wallet format")

        # Factor 4: Try to get basic wallet info (if API allows)
        try:
            # This would be enhanced with actual wallet analysis APIs
            # For now, we'll use basic heuristics
            wallet_info_url = f"https://public-api.solscan.io/account/{wallet_address}"
            async with session.get(wallet_info_url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("account"):
                        history_score += 10
                        factors.append("Wallet has transaction history")
                    else:
                        history_score -= 5
                        factors.append("Wallet appears new or inactive")
        except Exception:
            factors.append("Could not verify wallet history")

        return history_score, factors

    except Exception as e:
        logger.debug(f"Error evaluating wallet {wallet_address}: {e}")
        return -10, ["Wallet evaluation failed"]


async def get_sentiment_enhanced_tokens(
    cfg: Mapping[str, object], 
    min_galaxy_score: float = 60.0,
    min_sentiment: float = 0.6,
    limit: int = 20
) -> List[Tuple[str, Dict]]:
    """
    Get Solana tokens with enhanced sentiment data from LunarCrush.
    
    Returns list of (mint_address, sentiment_data) tuples for tokens that
    meet the sentiment criteria for potential early entries.
    """
    try:
        from crypto_bot.sentiment_filter import get_lunarcrush_client, SentimentDirection
        
        client = get_lunarcrush_client()

        trending_tokens = await client.get_trending_solana_tokens(limit=limit * 2)
        if not trending_tokens:
            logger.info("No trending Solana tokens returned by LunarCrush")
            return []

        symbol_to_mint = await _get_symbol_to_mint_map(cfg, (symbol for symbol, _ in trending_tokens))

        enhanced_results: List[Tuple[str, Dict]] = []
        seen_mints: Set[str] = set()

        for symbol, sentiment_data in trending_tokens:
            if (sentiment_data.sentiment_direction != SentimentDirection.BULLISH or
                    sentiment_data.galaxy_score < min_galaxy_score or
                    sentiment_data.sentiment < min_sentiment):
                continue

            mint_address = symbol_to_mint.get(_normalize_symbol(symbol))
            if not mint_address:
                logger.debug("Skipping %s due to missing Solana mint mapping", symbol)
                continue

            if mint_address in seen_mints:
                continue

            sentiment_dict = {
                "symbol": symbol,
                "galaxy_score": sentiment_data.galaxy_score,
                "alt_rank": sentiment_data.alt_rank,
                "sentiment": sentiment_data.sentiment,
                "sentiment_direction": sentiment_data.sentiment_direction.value,
                "social_mentions": sentiment_data.social_mentions,
                "social_volume": sentiment_data.social_volume,
                "bullish_strength": sentiment_data.bullish_strength,
                "last_updated": sentiment_data.last_updated,
            }

            enhanced_results.append((mint_address, sentiment_dict))
            seen_mints.add(mint_address)

            if len(enhanced_results) >= limit:
                break

        logger.info(f"Found {len(enhanced_results)} sentiment-enhanced Solana tokens")
        return enhanced_results

    except Exception as exc:
        logger.error(f"Failed to get sentiment-enhanced tokens: {exc}")
        return []


async def score_token_by_sentiment(symbol: str) -> Optional[Dict]:
    """
    Score a token based on LunarCrush sentiment metrics.
    
    Returns sentiment scoring dict or None if token not found/error.
    """
    try:
        from crypto_bot.sentiment_filter import get_lunarcrush_client, SentimentDirection
        
        client = get_lunarcrush_client()
        sentiment_data = await client.get_sentiment(symbol)
        
        # Calculate composite score (0-100)
        galaxy_weight = 0.4
        sentiment_weight = 0.3
        social_weight = 0.2
        rank_weight = 0.1
        
        # Normalize alt_rank (lower is better, so invert)
        rank_score = max(0, 100 - (sentiment_data.alt_rank / 10))
        
        # Normalize social metrics (log scale to handle large variations)
        import math
        social_score = min(100, math.log10(max(1, sentiment_data.social_mentions)) * 10)
        
        composite_score = (
            sentiment_data.galaxy_score * galaxy_weight +
            sentiment_data.sentiment * 100 * sentiment_weight +
            social_score * social_weight +
            rank_score * rank_weight
        )
        
        return {
            "composite_score": composite_score,
            "galaxy_score": sentiment_data.galaxy_score,
            "sentiment": sentiment_data.sentiment,
            "sentiment_direction": sentiment_data.sentiment_direction.value,
            "alt_rank": sentiment_data.alt_rank,
            "social_mentions": sentiment_data.social_mentions,
            "social_volume": sentiment_data.social_volume,
            "bullish_strength": sentiment_data.bullish_strength,
            "recommendation": _get_recommendation(sentiment_data, composite_score)
        }
        
    except Exception as exc:
        logger.warning(f"Failed to score token {symbol} by sentiment: {exc}")
        return None


def _get_recommendation(sentiment_data, composite_score: float) -> str:
    """Get trading recommendation based on sentiment analysis."""
    from crypto_bot.sentiment_filter import SentimentDirection
    
    if composite_score >= 80 and sentiment_data.sentiment_direction == SentimentDirection.BULLISH:
        return "STRONG_BUY"
    elif composite_score >= 60 and sentiment_data.sentiment_direction == SentimentDirection.BULLISH:
        return "BUY"
    elif composite_score >= 40 and sentiment_data.sentiment_direction != SentimentDirection.BEARISH:
        return "HOLD"
    elif sentiment_data.sentiment_direction == SentimentDirection.BEARISH:
        return "AVOID"
    else:
        return "NEUTRAL"
