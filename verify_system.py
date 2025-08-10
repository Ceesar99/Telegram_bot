#!/usr/bin/env python3
"""
System Verification Script
Tests that the unified trading system can be imported and initialized
"""

import sys
import traceback

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing system imports...")
    
    try:
        # Test core imports
        print("  Testing core components...")
        import signal_engine
        import risk_manager
        import performance_tracker
        print("    ✅ Core components imported")
        
        # Test institutional imports
        print("  Testing institutional components...")
        import enhanced_signal_engine
        from portfolio.institutional_risk_manager import InstitutionalRiskManager
        from execution.smart_order_router import SmartOrderRouter
        print("    ✅ Institutional components imported")
        
        # Test unified system
        print("  Testing unified system...")
        import unified_trading_system
        print("    ✅ Unified system imported")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_system_modes():
    """Test that the system can be initialized in different modes"""
    print("\n🔍 Testing system modes...")
    
    try:
        from unified_trading_system import UnifiedTradingSystem
        
        # Test original mode
        print("  Testing original mode...")
        system = UnifiedTradingSystem(mode="original")
        print(f"    ✅ Original mode: {system.mode.name}")
        
        # Test institutional mode
        print("  Testing institutional mode...")
        system = UnifiedTradingSystem(mode="institutional")
        print(f"    ✅ Institutional mode: {system.mode.name}")
        
        # Test hybrid mode
        print("  Testing hybrid mode...")
        system = UnifiedTradingSystem(mode="hybrid")
        print(f"    ✅ Hybrid mode: {system.mode.name}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Mode testing failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("🚀 Unified Trading System - Verification Script")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test system modes
    modes_ok = test_system_modes()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION RESULTS")
    print("=" * 50)
    
    if imports_ok and modes_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The unified trading system is ready to use")
        print("\n🚀 To start trading, run:")
        print("   python3 unified_trading_system.py")
        print("\n📚 For more options, see QUICK_START.md")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check the error messages above and fix any issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())