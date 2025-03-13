DEFAULT_PARAMS = {
    # Simulation parameters
    'duration': 30,
    'production_interval': 1.0,
    'production_batch_size': 500,
    
    # Supplier parameters
    'supplier_reliability': 0.9,
    'disruption_interval': 100,
    'disruption_probability': 0.3,
    'reliability_reduction': 0.5,
    'min_downtime': 2,
    'max_downtime': 5,
    
    # Inventory parameters
    'safety_stock_factor': 2.0,
    'policy_type': 'sS',
    'sS_buffer': 200,
    
    # Cost parameters
    'ordering_cost': 100.0,
    'holding_cost': 0.5,
    'shortage_cost': 5.0,
    
    # Inventory capacities
    'initial_warehouse': 500,
    'warehouse_capacity': 2000,
    'initial_retailer': 100,
    'retailer_capacity': 500,
    
    # Logistics parameters
    'shipment_delay': 2,
    
    # Demand parameters
    'average_daily_demand': 100,
    'linear_growth': 2.0,
    'seasonal_amplitude': 15.0,
    'random_std': 5.0,
    'demand_multiplier': 1.0
}