import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
class Analytics:
    """Contains methods for calculating performance metrics"""
    
    @staticmethod
    def calculate_service_level(immediate_fulfilled, total_demand):
        return immediate_fulfilled / total_demand if total_demand > 0 else 1.0
    
    @staticmethod
    def calculate_inventory_turns(data):
        """Robust turns calculation"""
        if data.empty or 'inventory' not in data.columns or 'demand' not in data.columns:
            return 0
        
        # Handle zero-inventory scenarios
        clean_inventory = data['inventory'].replace(0, np.nan)
        
        with np.errstate(invalid='ignore'):
            avg_inventory = clean_inventory.mean(skipna=True)
            if pd.isna(avg_inventory) or avg_inventory == 0:
                return 0
                
            return data['demand'].sum() / avg_inventory
        
    @staticmethod
    def calculate_cost_breakdown(data):
        return {
            'holding': data['holding_cost'].iloc[-1],
            'shortage': data['shortage_cost'].iloc[-1],
            'ordering': data['ordering_cost'].iloc[-1],
            'total': data['holding_cost'].iloc[-1] + 
                    data['shortage_cost'].iloc[-1] + 
                    data['ordering_cost'].iloc[-1]
        }
    
    @staticmethod
    def generate_forecast(data, periods=14):
        """Safe demand forecasting"""
        if len(data) < 14 or 'demand' not in data.columns:
            return pd.Series()  # Return empty series instead of None
            
        clean_demand = data['demand'].dropna()
        if len(clean_demand) < 14:
            return pd.Series()
            
        # Handle constant data cases
        if clean_demand.nunique() == 1:
            return pd.Series([clean_demand.iloc[0]] * periods)
            
        # Use context manager for error handling
        with np.errstate(all='ignore'):
            model = ExponentialSmoothing(
                clean_demand,
                seasonal='add',
                seasonal_periods=7
            ).fit()
            return model.forecast(periods)