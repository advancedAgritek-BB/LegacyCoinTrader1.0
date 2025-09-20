from __future__ import annotations

"""Utilities for scanning new Solana tokens."""

import asyncio
import logging
import os
from typing import Dict, List, Mapping, Optional, Tuple

import aiohttp

from .token_registry import get_jupiter_registry, index_registry_by_symbol

logger = logging.getLogger(__name__)


async def get_solana_new_tokens(cfg: Mapping[str, object]) -> List[str]:
    """Return a list of new token mint addresses using multiple API sources."""

    limit_candidate = (
        cfg.get("limit")
        or cfg.get("max_tokens_per_scan")
        or cfg.get("max_tokens")
        or cfg.get("solana_scanner_limit")
        or cfg.get("scanner_limit")
    )
    try:
        limit = max(1, int(limit_candidate)) if limit_candidate is not None else 20
    except (TypeError, ValueError):
        limit = 20

    helius_key = str(cfg.get("helius_key") or os.getenv("HELIUS_KEY", "")).strip()
    url = str(
        cfg.get("url")
        or cfg.get("helius_endpoint")
        or cfg.get("helius_url")
        or os.getenv("HELIUS_ENDPOINT", "")
    ).strip()

    if url and "${HELIUS_KEY}" in url:
        url = url.replace("${HELIUS_KEY}", helius_key)
    if url and helius_key and "YOUR_KEY" in url:
        url = url.replace("YOUR_KEY", helius_key)

    discovered: List[str] = []

    if url and helius_key:
        before = len(discovered)
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "dex.getNewPools",
                    "params": {"protocols": ["raydium"], "limit": limit},
                }
                async with session.post(url, json=payload, timeout=10) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as exc:
            logger.debug("Helius dex.getNewPools failed: %s", exc)
        else:
            result = data.get("result", []) if isinstance(data, Mapping) else []
            for pool in result:
                if not isinstance(pool, Mapping):
                    continue
                token_a = pool.get("tokenA") or pool.get("tokenAMint")
                token_b = pool.get("tokenB") or pool.get("tokenBMint")
                if isinstance(token_a, str) and token_a:
                    discovered.append(token_a)
                if isinstance(token_b, str) and token_b:
                    discovered.append(token_b)
            added = len(discovered) - before
            if added:
                logger.info("Helius returned %s tokens", added)

    logger.info("Trying Raydium API for pool discovery")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.raydium.io/pairs", timeout=15) as resp:
                resp.raise_for_status()
                payload = await resp.json()
    except Exception as exc:
        logger.debug("Raydium API failed: %s", exc)
    else:
        if isinstance(payload, list):
            before = len(discovered)
            sorted_pools = sorted(
                (pool for pool in payload if isinstance(pool, Mapping)),
                key=lambda pool: float(pool.get("liquidity", 0) or 0.0),
                reverse=True,
            )
            ws_ignored = {
                "So11111111111111111111111111111111111111112",
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            }
            for pool in sorted_pools[: limit * 2]:
                pair_id = str(pool.get("pair_id") or "")
                if "-" not in pair_id:
                    continue
                token_a, token_b = pair_id.split("-", 1)
                if token_a and token_a not in ws_ignored:
                    discovered.append(token_a)
                if token_b and token_b not in ws_ignored:
                    discovered.append(token_b)
            added = len(discovered) - before
            if added:
                logger.info("Raydium API returned %s tokens", added)

    logger.info("Trying Orca API for pool discovery")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.orca.so/api/pools", timeout=10) as resp:
                resp.raise_for_status()
                payload = await resp.json()
    except Exception as exc:
        logger.debug("Orca API failed: %s", exc)
    else:
        if isinstance(payload, list):
            before = len(discovered)
            for pool in payload[: limit * 2]:
                if not isinstance(pool, Mapping):
                    continue
                token_a = pool.get("tokenA") or pool.get("tokenAMint")
                token_b = pool.get("tokenB") or pool.get("tokenBMint")
                if isinstance(token_a, str) and token_a:
                    discovered.append(token_a)
                if isinstance(token_b, str) and token_b:
                    discovered.append(token_b)
            added = len(discovered) - before
            if added:
                logger.info("Orca API returned %s tokens", added)

    logger.info("Trying Jupiter API for token discovery")
    try:
        registry = await get_jupiter_registry()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Jupiter API failed: %s", exc)
    else:
        if registry:
            before = len(discovered)
            solana_index = index_registry_by_symbol(registry, chain_id=101)
            for entries in solana_index.values():
                for entry in entries:
                    address = entry.get("address")
                    if isinstance(address, str) and address:
                        discovered.append(address)
            added = len(discovered) - before
            if added:
                logger.info("Jupiter registry contributed %s tokens", added)

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
                            results = [launch["mint"] for launch in credible_launches[: limit * 2] if launch.get("mint")]

                            if results:
                                before = len(discovered)
                                discovered.extend(results)
                                added = len(discovered) - before
                                logger.info(
                                    "pump.fun API returned %s credible tokens after wallet evaluation",
                                    added,
                                )
        except Exception as exc:
            logger.debug(f"pump.fun API failed: {exc}")

    unique_tokens: List[str] = []
    seen: set[str] = set()
    for token in discovered:
        if not isinstance(token, str):
            continue
        cleaned = token.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_tokens.append(cleaned)
        if len(unique_tokens) >= limit:
            break

    if unique_tokens:
        return unique_tokens

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
        
        # Get trending tokens with sentiment data
        trending_tokens = await client.get_trending_solana_tokens(limit=limit * 2)
        
        enhanced_results = []
        seen_mints: set[str] = set()
        for mint_address, sentiment_data, metadata in trending_tokens:
            if not isinstance(mint_address, str) or not mint_address:
                continue
            if mint_address in seen_mints:
                continue
            # Filter for strong bullish sentiment
            if (sentiment_data.sentiment_direction == SentimentDirection.BULLISH and
                sentiment_data.galaxy_score >= min_galaxy_score and
                sentiment_data.sentiment >= min_sentiment):
                symbol = str(metadata.get("symbol") or "")
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
                    "mint": mint_address,
                    "name": metadata.get("name"),
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
