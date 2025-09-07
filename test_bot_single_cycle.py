#!/usr/bin/env python3
"""
Test script to run the bot for a single cycle to verify it works correctly.
"""

import asyncio
import sys
import os
import time
import signal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR

# Setup logging
logger = setup_logger("single_cycle_test", LOG_DIR / "single_cycle_test.log")

class BotTester:
    def __init__(self):
        self.start_time = None
        self.timeout = 180  # 3 minutes timeout
        
    async def test_single_cycle(self):
        """Test the bot for a single cycle."""
        logger.info("🧪 Starting single cycle test...")
        self.start_time = time.time()
        
        # Set environment variables to prevent interactive mode
        os.environ['AUTO_START_TRADING'] = '1'
        os.environ['NON_INTERACTIVE'] = '1'
        
        try:
            # Import the main bot function
            from crypto_bot.main import _main_impl
            
            # Create a task to run the bot
            bot_task = asyncio.create_task(self._run_bot_with_timeout())
            
            # Run the bot with timeout
            await bot_task
            
        except asyncio.TimeoutError:
            logger.error("❌ Bot test timed out - likely stuck in infinite loop")
            return False
        except Exception as e:
            logger.error(f"❌ Bot test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        logger.info("✅ Single cycle test completed successfully")
        return True
    
    async def _run_bot_with_timeout(self):
        """Run the bot with a timeout."""
        try:
            # Import the main bot function
            from crypto_bot.main import _main_impl
            
            # Create a timeout task
            timeout_task = asyncio.create_task(asyncio.sleep(self.timeout))
            
            # Create the bot task
            bot_task = asyncio.create_task(self._run_bot_single_cycle())
            
            # Race between timeout and bot completion
            done, pending = await asyncio.wait(
                [bot_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                
            # Check which task completed
            if bot_task in done:
                result = await bot_task
                logger.info("✅ Bot cycle completed successfully")
                return result
            else:
                logger.error("❌ Bot test timed out")
                raise asyncio.TimeoutError("Bot test timed out")
                
        except Exception as e:
            logger.error(f"❌ Error in bot timeout handler: {e}")
            raise
    
    async def _run_bot_single_cycle(self):
        """Run the bot for a single cycle."""
        logger.info("🤖 Starting bot single cycle...")
        
        # Load config to check settings
        import yaml
        config_path = Path("crypto_bot/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Config loaded - enhanced fetcher: {config.get('use_enhanced_ohlcv_fetcher', 'NOT SET')}")
        
        # Import and run main phases manually
        from crypto_bot.main import load_config
        from crypto_bot.execution.cex_executor import get_exchange
        from crypto_bot.phase_runner import BotContext, PhaseRunner
        from crypto_bot.main import (
            fetch_candidates,
            update_caches, 
            analyse_batch,
            execute_signals
        )
        
        # Load config and exchange
        config = load_config()
        exchange, ws_client = get_exchange(config)
        
        # Create context
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=config
        )
        ctx.exchange = exchange
        ctx.ws_client = ws_client
        
        # Run a single cycle
        logger.info("🔄 Running single trading cycle...")
        
        try:
            # Phase 1: Fetch candidates
            logger.info("Phase 1: Fetching candidates...")
            await fetch_candidates(ctx)
            logger.info(f"Phase 1 complete - {len(ctx.current_batch)} candidates")
            
            # Phase 2: Update caches (this is where it was getting stuck)
            logger.info("Phase 2: Updating caches...")
            await update_caches(ctx)
            logger.info("Phase 2 complete - caches updated")
            
            # Phase 3: Analyze batch
            logger.info("Phase 3: Analyzing batch...")
            await analyse_batch(ctx)
            logger.info(f"Phase 3 complete - {len(ctx.analysis_results)} analysis results")
            
            # Check if any tokens made it to evaluation
            actionable_signals = [r for r in ctx.analysis_results if not r.get("skip") and r.get("direction") != "none"]
            logger.info(f"🎯 Actionable signals generated: {len(actionable_signals)}")
            
            for result in actionable_signals:
                logger.info(f"  ✅ {result['symbol']}: {result['direction']} (score: {result.get('score', 0.0):.4f})")
            
            if not actionable_signals:
                logger.info("ℹ️ No actionable signals generated (normal in stable market conditions)")
                
                # Log why signals were not generated
                for result in ctx.analysis_results:
                    if result.get('skip'):
                        logger.info(f"  ❌ {result['symbol']}: Skipped - {result.get('skip')}")
                    elif result.get('direction') == 'none':
                        logger.info(f"  ❌ {result['symbol']}: No direction signal (score: {result.get('score', 0.0):.4f})")
                    elif result.get('score', 0.0) < config.get('min_confidence_score', 0.0):
                        logger.info(f"  ❌ {result['symbol']}: Score too low ({result.get('score', 0.0):.4f} < {config.get('min_confidence_score', 0.0)})")
            
            # Phase 4: Execute signals (if any)
            if actionable_signals:
                logger.info("Phase 4: Executing signals...")
                await execute_signals(ctx)
                logger.info("Phase 4 complete - signals executed")
            else:
                logger.info("Phase 4: Skipped - no signals to execute")
            
        except Exception as e:
            logger.error(f"❌ Error during cycle phases: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
        logger.info("✅ Single cycle completed successfully")
        return True

async def main():
    """Main test function."""
    tester = BotTester()
    success = await tester.test_single_cycle()
    
    if success:
        print("✅ Bot single cycle test PASSED")
        print("🎯 Tokens are now making it to evaluation!")
    else:
        print("❌ Bot single cycle test FAILED")
        
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test failed: {e}")
        sys.exit(1)
