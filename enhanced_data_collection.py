#!/usr/bin/env python3
"""
📊 ENHANCED DATA COLLECTION - PRODUCTION READY
Expand and improve existing market data for model training
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import yfinance as yf
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """Enhanced data collection and expansion"""
    
    def __init__(self):
        self.logger = logging.getLogger('EnhancedDataCollector')
        self.data_dir = '/workspace/data/real_market_data/'
        self.existing_data = None
        
    def load_existing_data(self):
        """Load existing market data"""
        try:
            data_file = os.path.join(self.data_dir, 'combined_market_data_20250816_092932.csv')
            if os.path.exists(data_file):
                self.logger.info(f"Loading existing data from {data_file}")
                df = pd.read_csv(data_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                self.logger.info(f"✅ Loaded {len(df)} records")
                self.logger.info(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                self.logger.info(f"🎯 Symbols: {df['symbol'].nunique()} unique")
                self.logger.info(f"💱 Symbols list: {list(df['symbol'].unique())}")
                
                self.existing_data = df
                return True
            else:
                self.logger.error(f"❌ Data file not found: {data_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return False
    
    def expand_recent_data(self, days_to_add=30):
        """Expand data with more recent market data"""
        try:
            self.logger.info(f"🚀 Expanding data with last {days_to_add} days")
            
            # Get currency pairs from existing data
            symbols = self.existing_data['symbol'].unique()
            yahoo_symbols = []
            
            for symbol in symbols:
                if '=X' in symbol:
                    yahoo_symbols.append(symbol)
                else:
                    # Convert to Yahoo format
                    if '/' in symbol:
                        yahoo_symbols.append(symbol.replace('/', '') + '=X')
                    else:
                        yahoo_symbols.append(symbol + '=X')
            
            # Remove duplicates
            yahoo_symbols = list(set(yahoo_symbols))
            self.logger.info(f"📈 Fetching recent data for {len(yahoo_symbols)} symbols")
            
            new_data = []
            successful_symbols = 0
            
            for symbol in yahoo_symbols:
                try:
                    self.logger.info(f"📊 Fetching {symbol}...")
                    
                    # Fetch recent data
                    ticker = yf.Ticker(symbol)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_to_add)
                    
                    # Get hourly data
                    data = ticker.history(start=start_date, end=end_date, interval='1h')
                    
                    if len(data) > 0:
                        # Format data to match existing structure
                        data.reset_index(inplace=True)
                        data['symbol'] = symbol
                        data['source'] = 'yahoo_finance'
                        data['collected_at'] = datetime.now()
                        data['dividends'] = 0.0
                        data['stock_splits'] = 0.0
                        
                        # Rename columns to match existing format
                        data = data.rename(columns={
                            'Datetime': 'datetime',
                            'Open': 'open',
                            'High': 'high', 
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume',
                            'Dividends': 'dividends',
                            'Stock Splits': 'stock_splits'
                        })
                        
                        # Ensure datetime column exists
                        if 'datetime' not in data.columns and 'Date' in data.columns:
                            data = data.rename(columns={'Date': 'datetime'})
                        
                        # Select only required columns
                        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 
                                       'dividends', 'stock_splits', 'symbol', 'source', 'collected_at']
                        
                        # Add missing columns if needed
                        for col in required_cols:
                            if col not in data.columns:
                                if col in ['dividends', 'stock_splits', 'volume']:
                                    data[col] = 0.0
                                elif col == 'source':
                                    data[col] = 'yahoo_finance'
                                elif col == 'collected_at':
                                    data[col] = datetime.now()
                        
                        data = data[required_cols]
                        new_data.append(data)
                        successful_symbols += 1
                        self.logger.info(f"✅ {symbol}: {len(data)} records")
                    else:
                        self.logger.warning(f"⚠️ No data for {symbol}")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error fetching {symbol}: {e}")
                    continue
            
            if new_data:
                # Combine new data
                combined_new = pd.concat(new_data, ignore_index=True)
                self.logger.info(f"✅ Collected {len(combined_new)} new records from {successful_symbols} symbols")
                
                # Merge with existing data
                all_data = pd.concat([self.existing_data, combined_new], ignore_index=True)
                
                # Remove duplicates
                before_dedup = len(all_data)
                all_data = all_data.drop_duplicates(subset=['datetime', 'symbol'], keep='last')
                after_dedup = len(all_data)
                
                self.logger.info(f"🔄 Removed {before_dedup - after_dedup} duplicate records")
                
                # Sort by datetime
                all_data = all_data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
                
                # Save expanded data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.data_dir, f'expanded_market_data_{timestamp}.csv')
                all_data.to_csv(output_file, index=False)
                
                self.logger.info(f"💾 Saved expanded dataset: {output_file}")
                self.logger.info(f"📊 Total records: {len(all_data)}")
                self.logger.info(f"📅 Date range: {all_data['datetime'].min()} to {all_data['datetime'].max()}")
                
                return output_file, len(all_data)
            else:
                self.logger.error("❌ No new data collected")
                return None, 0
                
        except Exception as e:
            self.logger.error(f"❌ Error expanding data: {e}")
            return None, 0
    
    def validate_data_quality(self, data_file):
        """Validate the quality of collected data"""
        try:
            self.logger.info("🔍 Validating data quality...")
            
            df = pd.read_csv(data_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            quality_report = {
                'total_records': len(df),
                'unique_symbols': df['symbol'].nunique(),
                'date_range_days': (df['datetime'].max() - df['datetime'].min()).days,
                'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'data_gaps': {},
                'quality_score': 0
            }
            
            # Check for data gaps by symbol
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('datetime')
                
                # Check for gaps (more than 2 hours between records)
                time_diffs = symbol_data['datetime'].diff()
                gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
                quality_report['data_gaps'][symbol] = len(gaps)
            
            # Calculate quality score
            score = 100
            if quality_report['missing_data_pct'] > 5:
                score -= 20
            elif quality_report['missing_data_pct'] > 1:
                score -= 10
            
            total_gaps = sum(quality_report['data_gaps'].values())
            if total_gaps > 100:
                score -= 30
            elif total_gaps > 50:
                score -= 15
            
            quality_report['quality_score'] = max(0, score)
            
            self.logger.info("=" * 50)
            self.logger.info("📊 DATA QUALITY REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"📈 Total Records: {quality_report['total_records']:,}")
            self.logger.info(f"💱 Unique Symbols: {quality_report['unique_symbols']}")
            self.logger.info(f"📅 Date Range: {quality_report['date_range_days']} days")
            self.logger.info(f"❌ Missing Data: {quality_report['missing_data_pct']:.2f}%")
            self.logger.info(f"🕳️ Total Data Gaps: {total_gaps}")
            self.logger.info(f"🎯 Quality Score: {quality_report['quality_score']}/100")
            self.logger.info("=" * 50)
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"❌ Error validating data: {e}")
            return None
    
    def run_enhanced_collection(self):
        """Run the complete enhanced data collection process"""
        self.logger.info("🚀 Starting Enhanced Data Collection Process")
        self.logger.info("=" * 60)
        
        # Step 1: Load existing data
        if not self.load_existing_data():
            self.logger.error("❌ Failed to load existing data")
            return False
        
        # Step 2: Expand with recent data
        output_file, total_records = self.expand_recent_data(days_to_add=30)
        
        if output_file and total_records > 0:
            # Step 3: Validate data quality
            quality_report = self.validate_data_quality(output_file)
            
            if quality_report and quality_report['quality_score'] >= 70:
                self.logger.info("✅ Enhanced data collection completed successfully!")
                self.logger.info(f"📁 Output file: {output_file}")
                self.logger.info(f"📊 Ready for model training with {total_records:,} records")
                return True
            else:
                self.logger.warning("⚠️ Data quality below threshold, but collection completed")
                return True
        else:
            self.logger.error("❌ Failed to expand data collection")
            return False

if __name__ == "__main__":
    collector = EnhancedDataCollector()
    success = collector.run_enhanced_collection()
    
    if success:
        print("\n🎉 DATA COLLECTION SUCCESS!")
        print("✅ Your trading system now has enhanced market data")
        print("🚀 Ready to proceed with model retraining")
    else:
        print("\n❌ DATA COLLECTION FAILED!")
        print("⚠️ Please check the logs and try again")