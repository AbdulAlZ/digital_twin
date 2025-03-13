import simpy
import numpy as np
import pandas as pd
from entities.retailer import Retailer
from entities.warehouse import Warehouse
from entities.supplier import Supplier
import random
import ast

class DigitalTwinInventory:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.all_lead_times = []
        self.data = []
        self.ml_dataset = []
        self.supplier_production_history = []
        self.inventory_age_timestamps = {}
        self.order_lead_times = []
        self.initialized_from_state = False

        if self.params is None:
            raise ValueError("ERROR: Simulation parameters 'params' cannot be None!")

        # Initialize warehouse first, then retailer (avoid passing None)
        self.warehouse = Warehouse(env, None, params)  
        self.retailer = Retailer(env, params, self.warehouse, self)  
        self.warehouse.retailer = self.retailer  # Set the retailer in warehouse

        self.supplier = Supplier(env, params, self.warehouse)

        self.total_demand = 0
        self.cumulative_costs = {
            'holding': 0.0,
            'shortage': 0.0,
            'ordering': 0.0
        }
        
        self.env.process(self.demand_generator())
        self.env.process(self.disruption_manager())
        # Start processes only if not loading from state
        if not self.initialized_from_state:
            self.env.process(self.demand_generator())
            self.env.process(self.disruption_manager())


    def load_initial_state(self, initial_df):
        """Load historical data from CSV or previous simulation safely."""
        print("\n" + "="*40)
        print("Loading initial simulation state...")

        # Validate input dataframe
        required_cols = {'time', 'inventory', 'warehouse_inventory', 'demand'}
        missing_cols = required_cols - set(initial_df.columns)
        if missing_cols:
            raise ValueError(f"ERROR: Missing required columns: {missing_cols}")

        # Ensure 'time' column is numeric to prevent errors in run_simulation()
        initial_df['time'] = pd.to_numeric(initial_df['time'], errors='coerce').fillna(0)

        # Load last state from historical data
        last_state = initial_df.iloc[-1].to_dict()

        # Set core inventory levels safely
        current_inventory = self.retailer.inventory.level
        inventory_difference = last_state['inventory'] - current_inventory

        if inventory_difference > 0:
            self.retailer.inventory.put(inventory_difference)  # Add stock
        elif inventory_difference < 0:
            self.retailer.inventory.get(abs(inventory_difference))  # Remove stock

        self.retailer.backorders = last_state.get('backorders', 0)

        # Correct warehouse inventory update
        current_warehouse_inventory = self.warehouse.inventory.level
        warehouse_difference = last_state['warehouse_inventory'] - current_warehouse_inventory

        if warehouse_difference > 0:
            self.warehouse.inventory.put(warehouse_difference)  # Add stock
        elif warehouse_difference < 0:
            self.warehouse.inventory.get(abs(warehouse_difference))  # Remove stock

        # Ensure cumulative cost values are correctly loaded
        self.total_demand = initial_df['demand'].sum()
        self.cumulative_costs = {
            'holding': last_state.get('cumulative_holding', initial_df['holding_cost'].sum()),
            'shortage': last_state.get('cumulative_shortage', initial_df['shortage_cost'].sum()),
            'ordering': last_state.get('cumulative_ordering', initial_df['ordering_cost'].sum())
        }

        # Fix pipeline column safely (ensure proper decoding and list conversion)
        def safe_eval(value):
            try:
                return ast.literal_eval(value) if isinstance(value, str) and value.startswith("[") else []
            except (SyntaxError, ValueError, UnicodeDecodeError):
                return []  # If corrupted, return an empty list

        if 'pipeline' in last_state:
            last_state['pipeline'] = safe_eval(last_state['pipeline'])
            self.warehouse.pipeline = [
                {**order, 'created': float(order['created']), 'arrives': float(order['arrives'])}
                for order in last_state['pipeline']
            ]

        # Set simulation clock correctly
        time_offset = last_state['time'] - self.env.now
        if time_offset > 0:
            self.env.timeout(time_offset)  # Advance simulation time

        # Load historical lead times safely
        if 'avg_lead_time' in initial_df.columns:
            self.all_lead_times = initial_df['avg_lead_time'].dropna().tolist()

        # Load historical data for ML features safely
        self.data = initial_df.to_dict('records')

        self.initialized_from_state = True
        print("✅ Successfully loaded initial state!")
        print(f"📌 Current simulation time: {self.env.now}")
        print(f"📦 Retailer inventory: {self.retailer.inventory.level}")
        print(f"🏢 Warehouse inventory: {self.warehouse.inventory.level}")
        print("="*40 + "\n")



    def receive_shipment(self, quantity):
        """Track lead times when shipments arrive"""
        current_time = self.env.now

        print(f"DEBUG [{current_time}]: Checking for completed orders...")

        if not hasattr(self.warehouse, 'pipeline') or not self.warehouse.pipeline:
            print(f"DEBUG [{current_time}]: No orders in pipeline.")
            return
        
        completed_orders = []

        for order in list(self.warehouse.pipeline):
            lead_time = order['arrives'] - order['created']
            print(f"✅ DEBUG [{current_time}]: Order Created: {order['created']}, Arrives: {order['arrives']}, Lead Time: {lead_time}")

            if order['arrives'] <= current_time:
                if lead_time > 0:
                    self.all_lead_times.append(lead_time)
                    print(f"✅ DEBUG [{current_time}]: Order Delivered! Lead Time → {lead_time}")
                    completed_orders.append(order)
        
        for order in completed_orders:
            self.warehouse.pipeline.remove(order)


    def demand_generator(self):
        """Generate daily demand with safeguards"""
        while True:
            day = self.env.now
            
            # Calculate base demand components
            trend = self.params['linear_growth'] * day
            seasonality = self.params['seasonal_amplitude'] * np.sin(2 * np.pi * (day % 7)/7)
            noise = np.abs(np.random.normal(0, self.params['random_std']))  # Absolute noise
            
            base_demand = (
                self.params['average_daily_demand'] +
                trend +
                seasonality +
                noise
            )
            
            # Ensure positive demand with variance
            demand = max(1, int(base_demand * self.params['demand_multiplier']))  # Minimum demand=1
            demand += np.random.poisson(lam=3)
            
            self.total_demand += demand
            
            yield self.env.process(self.retailer.process_sale(demand))
            self.record_snapshot(demand)
            yield self.env.timeout(1)

    def disruption_manager(self):
        """Handle supplier reliability disruptions"""
        while True:
            yield self.env.timeout(np.random.exponential(max(1, self.params['disruption_interval'])))
            if np.random.random() < self.params['disruption_probability']:
                yield self.env.process(self.apply_disruption())

    def apply_disruption(self):
        """Apply temporary reliability reduction"""
        original = self.supplier.current_reliability
        self.supplier.current_reliability *= max(0.1, self.params['reliability_reduction'])
        
        duration = np.random.uniform(
            max(1, self.params['min_downtime']),
            max(2, self.params['max_downtime'])
        )
        yield self.env.timeout(duration)
        self.supplier.current_reliability = original

    def order_stock(self):
        if self.retailer.backorders > 0:
            avg_demand = np.mean([d['demand'] for d in self.simulation.data[-7:]]) if len(self.simulation.data) >= 7 else self.params['average_daily_demand']
            order_quantity = max(avg_demand * 2, self.retailer.backorders)  # ✅ Order only 2x avg demand instead of full backorders

            print(f"DEBUG [{self.env.now}]: Ordering {order_quantity} to clear {self.retailer.backorders} backorders")

            yield self.env.process(self.ship(order_quantity))

    def calculate_reorder_point(self, demand_history=None):
        lead_time = self.params['shipment_delay'] + self.params['production_interval']
        
        if demand_history and len(demand_history) >= 7:
            avg_demand = np.mean(demand_history)
            std_demand = np.std(demand_history)
            safety_stock = self.params['safety_stock_factor'] * std_demand * np.sqrt(lead_time)
        else:
            avg_demand = self.params['average_daily_demand']
            safety_stock = self.params['safety_stock_factor'] * 1.2  # ✅ Increase safety stock

        return int(avg_demand * lead_time + safety_stock)

    def record_snapshot(self, demand):
        """Record daily system state with numerical validation"""
        # Initialize with safe defaults
        current_time = self.env.now
        current_lead_times = []

        # Add current pipeline state to data
        pipeline_data = [{
            'created': o['created'],
            'arrives': o['arrives'],
            'quantity': o['quantity'],
            'status': 'in_transit' if self.env.now < o['arrives'] else 'delivered'
        } for o in self.warehouse.pipeline]

        
        # Validate and calculate lead times
        for order in self.warehouse.pipeline:
            try:
                created = order['created']
                arrives = order['arrives']
                if arrives > created:  # Only valid future orders
                    current_lead_times.append(arrives - created)
            except KeyError:
                continue

        # Safe average calculation
        with np.errstate(invalid='ignore'):
            avg_lead_time = float(np.nanmean(current_lead_times)) if current_lead_times else 0.0
            current_lead_time = float(current_lead_times[-1]) if current_lead_times else 0.0

        # ML features with validation
        with np.errstate(divide='ignore', invalid='ignore'):
            # Demand std (require ≥2 points)
            demand_window = [d.get('demand', 0) for d in self.data[-7:]] if self.data else []
            demand_std = float(np.nanstd(demand_window, ddof=1)) if len(demand_window) >= 2 else 0.0

            # Supplier reliability (30-day window)
            reliability_window = [d.get('supplier_reliability', 1.0) for d in self.data[-30:]]
            supplier_reliability = float(np.nanmean(reliability_window)) if reliability_window else 1.0

        # Inventory calculations with zero protection
        inventory_level = max(0, self.retailer.inventory.level)
        backorders = max(0, self.retailer.backorders)
        warehouse_inventory = max(0, self.warehouse.inventory.level)

        # Cost calculations
        daily_holding = float(inventory_level * self.params['holding_cost'])
        daily_shortage = float(backorders * self.params['shortage_cost'])
        
        # Cumulative costs with NaN protection
        self.cumulative_costs['holding'] = float(np.nansum([
            self.cumulative_costs.get('holding', 0), daily_holding
        ]))
        self.cumulative_costs['shortage'] = float(np.nansum([
            self.cumulative_costs.get('shortage', 0), daily_shortage
        ]))
        self.cumulative_costs['ordering'] = float(np.nansum([
            self.cumulative_costs.get('ordering', 0), self.retailer.daily_ordering_cost
        ]))

        # Append data with type validation
        self.data.append({
            'time': float(current_time),
            'inventory': float(inventory_level),
            'backorders': float(backorders),
            'warehouse_inventory': float(warehouse_inventory),
            'demand': float(demand),
            'pipeline': pipeline_data,
            'holding_cost': daily_holding,
            'shortage_cost': daily_shortage,
            'ordering_cost': float(self.retailer.daily_ordering_cost),
            'service_level': float(random.uniform(0.7, 1.0)),
            'inventory_turns': float(self.calculate_turns()),
            'supplier_reliability': float(self.supplier.current_reliability),
            'inventory_age_days': float(self.retailer.average_inventory_age),
            'in_transit': float(sum(o.get('quantity', 0) for o in self.warehouse.pipeline)),
            'safety_stock': float(self.warehouse.calculate_reorder_point(
                [d.get('demand', 0) for d in self.data[-7:]] if self.data else []
            )),
            'cycle_time': float(current_time - (self.data[-1]['time'] if self.data else 0)),
            'total_immediate_fulfilled': float(self.retailer.total_immediate_fulfilled),
            'total_backorder_fulfilled': float(self.retailer.total_backorder_fulfilled),
            'cumulative_total_cost': float(
                self.cumulative_costs['holding']/1000 +
                self.cumulative_costs['shortage']/1000 +
                self.cumulative_costs['ordering']/1000
            ),
            'supplier_production': float(
                self.supplier.last_production_qty 
                if hasattr(self.supplier, 'last_production_qty') 
                else 0
            ),
            'avg_lead_time': avg_lead_time,
            'current_lead_time': current_lead_time,
            'demand_std_7d': demand_std,
            'inventory_turnover_ratio': float(self.calculate_turns()),
            'supplier_reliability_30d': supplier_reliability
        })

        # Reset daily ordering cost with validation
        self.retailer.daily_ordering_cost = max(0.0, float(self.retailer.daily_ordering_cost))
        print(f"DEBUG [{self.env.now}]: Total Immediate Fulfill: {self.retailer.total_immediate_fulfilled}, Total Demand: {self.total_demand}")
        print(f"DEBUG [{self.env.now}]: Warehouse Pipeline →", self.warehouse.pipeline)
        print(f"DEBUG [{self.env.now}]: Immediate Fulfill: {self.retailer.total_immediate_fulfilled}, Backorders: {self.retailer.backorders}, Total Demand: {self.total_demand}")











    def calculate_service_level(self):
        """Service level calculation with safeguards"""
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.total_demand <= 0:
                return 1.0
            return min(1.0, max(0.0, self.retailer.total_immediate_fulfilled / self.total_demand))

    def calculate_turns(self):
        """Robust inventory turns calculation"""
        if len(self.data) < 7:
            return 0.0
        
        window = self.data[-7:]
        inventory_values = [d.get('inventory', 0) for d in window]
        demand_values = [d.get('demand', 0) for d in window]
        
        # Handle empty data case
        if not inventory_values or not demand_values:
            return 0.0
        
        # Calculate using numpy's nan-safe functions
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_inventory = np.nanmean(inventory_values)
            total_demand = np.nansum(demand_values)
            
            if np.isnan(avg_inventory) or avg_inventory <= 0:
                return 0.0
                
            turns = total_demand / avg_inventory
        
        return float(turns) if not np.isnan(turns) and not np.isinf(turns) else 0.0

def run_simulation(params, initial_state=None):
    """Run simulation with optional initial state"""
    env = simpy.Environment()
    sim = DigitalTwinInventory(env, params)
    
    if initial_state is not None:
        print("DEBUG: inside initial state")
        sim.load_initial_state(initial_state)
        # Add duration to existing time
        run_until = initial_state['time'].max() + params['duration']
    else:
        print("DEBUG: outside initial state")
        run_until = params['duration']
    
    env.run(until=run_until)
    return sim