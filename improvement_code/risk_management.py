
# RISK MANAGEMENT - CRITICAL FOR WIN RATE

def calculate_position_size(self, account_balance, risk_per_trade, stop_loss_pips):
    """Calculate optimal position size based on risk"""
    
    # Risk only 1-2% per trade
    risk_amount = account_balance * (risk_per_trade / 100)
    position_size = risk_amount / (stop_loss_pips * 10)
    
    # Maximum position size (5% of account)
    max_position = account_balance * 0.05
    
    return min(position_size, max_position)

def calculate_dynamic_stop_loss(self, entry_price, direction, atr_multiplier=2):
    """Calculate dynamic stop loss based on ATR"""
    
    atr = self.calculate_atr(20)
    
    if direction == 'BUY':
        stop_loss = entry_price - (atr * atr_multiplier)
    else:
        stop_loss = entry_price + (atr * atr_multiplier)
    
    return stop_loss

def check_drawdown_limits(self, current_balance, peak_balance, max_drawdown=15):
    """Check if drawdown exceeds limits"""
    
    drawdown = ((peak_balance - current_balance) / peak_balance) * 100
    
    if drawdown > max_drawdown:
        return False, f"Drawdown limit exceeded: {drawdown:.2f}%"
    
    return True, f"Drawdown acceptable: {drawdown:.2f}%"

def check_daily_loss_limit(self, daily_pnl, account_balance, max_daily_loss=5):
    """Check daily loss limits"""
    
    daily_loss_percentage = (abs(daily_pnl) / account_balance) * 100
    
    if daily_loss_percentage > max_daily_loss:
        return False, f"Daily loss limit exceeded: {daily_loss_percentage:.2f}%"
    
    return True, f"Daily loss acceptable: {daily_loss_percentage:.2f}%"
