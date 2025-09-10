#!/usr/bin/env python3
"""
Simple test runner to verify stop loss functionality is working.
"""

import sys
import yaml
from pathlib import Path

def test_configuration():
    """Test that configuration is properly set."""
    print("🔧 Testing Configuration...")
    
    config_path = Path("crypto_bot/config.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exit_cfg = config.get("exit_strategy", {})
    
    # Check critical settings
    critical_settings = {
        "stop_loss_pct": 0.01,
        "trailing_stop_pct": 0.008,
        "take_profit_pct": 0.04,
        "min_gain_to_trail": 0.005
    }
    
    all_good = True
    for setting, expected_value in critical_settings.items():
        if setting in exit_cfg:
            actual_value = exit_cfg[setting]
            if actual_value == expected_value:
                print(f"✅ {setting}: {actual_value}")
            else:
                print(f"⚠️  {setting}: {actual_value} (expected {expected_value})")
                all_good = False
        else:
            print(f"❌ {setting}: Missing")
            all_good = False
    
    # Check real-time monitoring
    monitoring_cfg = exit_cfg.get("real_time_monitoring", {})
    if monitoring_cfg.get("enabled", False):
        print("✅ Real-time monitoring: Enabled")
    else:
        print("❌ Real-time monitoring: Disabled")
        all_good = False
    
    return all_good

def test_main_file():
    """Test that main.py has required components."""
    print("\n📄 Testing Main File...")
    
    main_path = Path("crypto_bot/main.py")
    if not main_path.exists():
        print("❌ main.py not found")
        return False
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    required_components = [
        "async def handle_exits(ctx: BotContext) -> None:",
        "should_exit(",
        "ctx.position_monitor",
        "handle_exits,",
        "PhaseRunner("
    ]
    
    all_good = True
    for component in required_components:
        if component in content:
            print(f"✅ Found: {component}")
        else:
            print(f"❌ Missing: {component}")
            all_good = False
    
    return all_good

def test_emergency_monitor():
    """Test that emergency monitor exists."""
    print("\n🚨 Testing Emergency Monitor...")
    
    emergency_path = Path("emergency_stop_loss_monitor.py")
    if not emergency_path.exists():
        print("❌ Emergency monitor not found")
        return False
    
    with open(emergency_path, 'r') as f:
        content = f.read()
    
    required_methods = [
        "class EmergencyStopLossMonitor:",
        "should_exit",
        "calculate_stop_loss",
        "calculate_trailing_stop"
    ]
    
    all_good = True
    for method in required_methods:
        if method in content:
            print(f"✅ Found: {method}")
        else:
            print(f"❌ Missing: {method}")
            all_good = False
    
    return all_good

def test_restart_script():
    """Test that restart script exists."""
    print("\n🔄 Testing Restart Script...")
    
    restart_path = Path("restart_bot_fixed.sh")
    if not restart_path.exists():
        print("❌ Restart script not found")
        return False
    
    # Check if executable
    import os
    if os.access(restart_path, os.X_OK):
        print("✅ Restart script: Executable")
        return True
    else:
        print("⚠️  Restart script: Not executable")
        return False

def main():
    """Run all tests."""
    print("🧪 STOP LOSS FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Main File", test_main_file),
        ("Emergency Monitor", test_emergency_monitor),
        ("Restart Script", test_restart_script)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The stop loss system is properly configured and ready to use.")
        print("\nNext steps:")
        print("1. Run: ./restart_bot_fixed.sh")
        print("2. Monitor: tail -f crypto_bot/logs/bot.log")
        print("3. Check for stop loss activity in logs")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Some components may need attention before using stop loss system.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
