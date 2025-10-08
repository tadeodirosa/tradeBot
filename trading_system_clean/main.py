#!/usr/bin/env python3
"""
Crypto Analysis & Backtesting Suite
===================================

Unified command-line interface for the crypto analysis and backtesting project.

Usage:
    python main.py <command> [options]

Commands:
    analyze     - Run analysis on crypto assets
    backtest    - Execute backtesting strategies
    discover    - Discover new trading opportunities
    download    - Download market data
    optimize    - Run optimization and grid searches

Examples:
    python main.py analyze --symbol BTCUSDT --timeframe 4h
    python main.py backtest --symbol BTCUSDT --strategy meanreversion
    python main.py discover --top50 --binance
    python main.py download --symbols BTC,ETH --timeframes 1h,4h
    python main.py optimize --strategy meanreversion --asset BTCUSDT
"""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))

def setup_analyzer_commands(subparsers):
    """Setup analyze command and subcommands"""
    analyze_parser = subparsers.add_parser('analyze', help='Run crypto analysis')
    analyze_subparsers = analyze_parser.add_subparsers(dest='analyze_command')
    
    # Single asset analysis
    single_parser = analyze_subparsers.add_parser('single', help='Analyze single asset')
    single_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    single_parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')
    
    # Batch analysis
    batch_parser = analyze_subparsers.add_parser('batch', help='Batch analyze multiple assets')
    batch_parser.add_argument('--top', type=int, default=20, help='Analyze top N assets')
    batch_parser.add_argument('--candidates', help='CSV file with candidate assets')

def setup_backtest_commands(subparsers):
    """Setup backtest command and subcommands"""
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--symbol', required=True, help='Trading symbol')
    backtest_parser.add_argument('--timeframe', default='4h', help='Timeframe')
    backtest_parser.add_argument('--strategy', default='meanreversion', help='Strategy to test')
    backtest_parser.add_argument('--days', type=int, default=180, help='Days of historical data')

def setup_discover_commands(subparsers):
    """Setup discover command and subcommands"""
    discover_parser = subparsers.add_parser('discover', help='Discover trading opportunities')
    discover_parser.add_argument('--top50', action='store_true', help='Discover from top 50')
    discover_parser.add_argument('--binance', action='store_true', help='Focus on Binance listings')
    discover_parser.add_argument('--screen', action='store_true', help='Run screening pipeline')

def setup_download_commands(subparsers):
    """Setup download command and subcommands"""
    download_parser = subparsers.add_parser('download', help='Download market data')
    download_parser.add_argument('--symbols', help='Comma-separated symbols')
    download_parser.add_argument('--timeframes', default='1h,4h', help='Comma-separated timeframes')
    download_parser.add_argument('--top50', action='store_true', help='Download top 50 assets')

def setup_optimize_commands(subparsers):
    """Setup optimize command and subcommands"""
    optimize_parser = subparsers.add_parser('optimize', help='Run optimization')
    optimize_parser.add_argument('--strategy', default='meanreversion', help='Strategy to optimize')
    optimize_parser.add_argument('--symbol', help='Specific symbol to optimize')
    optimize_parser.add_argument('--gridsearch', action='store_true', help='Run grid search')

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Crypto Analysis & Backtesting Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command groups
    setup_analyzer_commands(subparsers)
    setup_backtest_commands(subparsers)
    setup_discover_commands(subparsers)
    setup_download_commands(subparsers)
    setup_optimize_commands(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Route to appropriate handler
        if args.command == 'analyze':
            handle_analyze(args)
        elif args.command == 'backtest':
            handle_backtest(args)
        elif args.command == 'discover':
            handle_discover(args)
        elif args.command == 'download':
            handle_download(args)
        elif args.command == 'optimize':
            handle_optimize(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        sys.exit(1)

def handle_analyze(args):
    """Handle analyze commands"""
    logger.info(f"Running analysis with command: {args.analyze_command}")
    
    if args.analyze_command == 'single':
        # Run single asset analysis
        import subprocess
        result = subprocess.run([
            sys.executable, 
            'analysis/analyzer_v10.py', 
            f"{args.symbol}_{args.timeframe}"
        ])
        return result.returncode
        
    elif args.analyze_command == 'batch':
        # Run batch analysis
        if args.candidates:
            logger.info(f"Running batch analysis with candidates file: {args.candidates}")
        else:
            logger.info(f"Running batch analysis on top {args.top} assets")
        # Run appropriate batch script
        import subprocess
        subprocess.run([sys.executable, 'scripts/analysis/analiza_top20_batch.py'])
    
def handle_backtest(args):
    """Handle backtest commands"""
    logger.info(f"Running backtest for {args.symbol} on {args.timeframe}")
    
    # Run backtester directly
    import subprocess
    result = subprocess.run([
        sys.executable, 
        'backtesting/backtester.py', 
        f"{args.symbol}_{args.timeframe}"
    ])
    return result.returncode

def handle_discover(args):
    """Handle discover commands"""
    logger.info("Running discovery pipeline")
    
    import subprocess
    if args.top50:
        subprocess.run([sys.executable, 'scripts/analysis/pipeline_top50_binance.py'])
    elif args.binance:
        subprocess.run([sys.executable, 'scripts/discovery/descubre_nuevas_coins_binance.py'])
    elif args.screen:
        subprocess.run([sys.executable, 'scripts/discovery/screen_binance_usdt.py'])

def handle_download(args):
    """Handle download commands"""
    logger.info("Running data download")
    
    import subprocess
    if args.top50:
        subprocess.run([sys.executable, 'scripts/download/download_top50_ohlcv.py'])
    else:
        subprocess.run([sys.executable, 'scripts/download/download_binance_ohlcv_h1_h4.py'])

def handle_optimize(args):
    """Handle optimize commands"""
    logger.info(f"Running optimization for {args.strategy}")
    
    import subprocess
    if args.gridsearch:
        subprocess.run([sys.executable, 'scripts/optimization/gridsearch_meanrev.py'])
    else:
        subprocess.run([sys.executable, 'scripts/optimization/backtest_downloaded_ohlcv.py'])

if __name__ == '__main__':
    main()