import simpy
import numpy as np
from collections import deque

class Retailer:
    def __init__(self, env, params, warehouse, simulation):
        self.env = env
        self.params = params
        self.warehouse = warehouse
        self.simulation = simulation
        
        self.inventory = simpy.Container(
            env,
            init=params['initial_retailer'],
            capacity=params['retailer_capacity']
        )
        
        self.inventory_age = deque()  # Track age of stock batches
        self.last_stock_added = None  # Timestamp of last stock addition

        self.backorders = 0
        self.total_immediate_fulfilled = 0
        self.total_backorder_fulfilled = 0
        self.daily_ordering_cost = 0.0
        
        self.env.process(self.inventory_policy())

    def inventory_policy(self):
        while True:
            yield self.env.process(self.check_incoming_shipments())
            
            if self.params['policy_type'] == 'sS':
                yield self.env.process(self.s_S_policy())
            elif self.params['policy_type'] == 'ROP':
                yield self.env.process(self.ROP_policy())
                
            yield self.env.timeout(1)

    def check_incoming_shipments(self):
        """Processes incoming shipments and fulfills backorders."""
        current_time = self.env.now
        completed_orders = []

        for order in self.warehouse.pipeline:
            if order['arrives'] <= current_time:
                print(f"✅ DEBUG [{current_time}]: Order Arrived! Processing {order['quantity']} units.")
                yield self.receive_shipment(order['quantity'])
                completed_orders.append(order)

        # ✅ Remove completed shipments from pipeline
        for order in completed_orders:
            self.warehouse.pipeline.remove(order)


    def s_S_policy(self):
        """(s, S) policy implementation"""
        demand_history = [d['demand'] for d in self.simulation.data[-7:]] if self.simulation.data else []
        s = self.warehouse.calculate_reorder_point(demand_history)
        S = s + self.params['sS_buffer']
        effective_inventory = self.get_effective_inventory()
        
        if effective_inventory < s:
            order_qty = max(0, S - effective_inventory)
            if order_qty > 0:
                self.daily_ordering_cost += self.params['ordering_cost']
                yield self.env.process(self.warehouse.ship(order_qty))

    def ROP_policy(self):
        """Reorder Point policy implementation"""
        demand_history = [d['demand'] for d in self.simulation.data[-7:]] if self.simulation.data else []
        rop = self.warehouse.calculate_reorder_point(demand_history)
        effective_inventory = self.get_effective_inventory()
        
        if effective_inventory < rop or self.backorders > 0:

            D = np.mean(demand_history) if demand_history else self.params['average_daily_demand']
            annual_demand = D * 365
            annual_holding_cost = self.params['holding_cost'] * 365
            eoq = int(np.sqrt((2 * annual_demand * self.params['ordering_cost']) / annual_holding_cost))
            order_qty = max(eoq, rop - effective_inventory)
            
            if order_qty > 0:
                self.daily_ordering_cost += self.params['ordering_cost']
                yield self.env.process(self.warehouse.ship(order_qty))

    def get_effective_inventory(self):
        """Calculate available inventory including only realistic in-transit stock"""
        current_time = self.env.now
        incoming = sum(
            o['quantity'] for o in self.warehouse.pipeline
            if o['arrives'] <= current_time + self.params['shipment_delay']
        )
        return max(0, self.inventory.level + incoming - self.backorders)  # ✅ Prevent overestimation

    def add_backorder(self, quantity, demand=0):
        """Adds backorders when demand cannot be met."""
        self.backorders += quantity
        print(f"DEBUG: Backorder Added: {quantity}, New Backorder Total: {self.backorders}")
        self.simulation.record_snapshot(demand)  # ✅ Now 'demand' is passed explicitly


    def process_sale(self, demand):
        available = self.inventory.level
        immediate_fulfill = min(available, demand)  # Fulfill as much as available

        shortage = demand - immediate_fulfill

        print(f"DEBUG [{self.env.now}]: Demand: {demand}, Available: {available}, Immediate Fulfill: {immediate_fulfill}, Shortage: {shortage}")

        if immediate_fulfill > 0:
            yield self.inventory.get(immediate_fulfill)
            self._update_inventory_age(immediate_fulfill)
            self.total_immediate_fulfilled += immediate_fulfill

        if immediate_fulfill < demand:  # Only backorder what isn't fulfilled
            self.add_backorder(demand - immediate_fulfill, demand)
            print(f"DEBUG [{self.env.now}]: Backorder Added: {demand - immediate_fulfill}, New Backorder Total: {self.backorders}")

        print(f"DEBUG [{self.env.now}]: Inventory AFTER fulfilling demand: {self.inventory.level}")

        yield self.env.timeout(0)



    def receive_shipment(self, quantity):
        """Process incoming stock and fulfill backorders first"""
        current_time = self.env.now
        print(f"DEBUG [{current_time}]: Receiving shipment of {quantity}")

        if self.backorders > 0:
            fulfilled = min(quantity, self.backorders)
            self.total_backorder_fulfilled += fulfilled
            self.backorders -= fulfilled
            quantity -= fulfilled
            print(f"DEBUG [{current_time}]: Backorders Fulfilled: {fulfilled}, Remaining Backorders: {self.backorders}")

        if quantity > 0:
            available_space = self.inventory.capacity - self.inventory.level
            qty_to_add = min(quantity, available_space)

            if qty_to_add > 0:
                yield self.inventory.put(qty_to_add)
                self._record_inventory_age(qty_to_add)
                print(f"DEBUG [{current_time}]: Stored {qty_to_add} in Retailer Inventory, New Level: {self.inventory.level}")
            else:
                print(f"⚠️ DEBUG [{current_time}]: No space to store stock. Skipping inventory addition.")

        yield self.env.timeout(0)




    def _record_inventory_age(self, quantity):
        """Record timestamp for new stock additions"""
        now = self.env.now
        self.inventory_age.extend([now]*quantity)
        self.last_stock_added = now

    def _update_inventory_age(self, quantity):
        """Remove oldest stock from age tracking"""
        for _ in range(quantity):
            try:
                self.inventory_age.popleft()
            except IndexError:
                break

    @property
    def average_inventory_age(self):
        """Calculate average age of current inventory"""
        if not self.inventory_age:
            return 0
        current_time = self.env.now
        return np.mean([current_time - ts for ts in self.inventory_age])   
    
    def calculate_order_quantity(self):
        """Add randomness to order quantities"""
        base_order = self.policy.calculate_order(
            self.inventory.level,
            self.warehouse.inventory.level,
            self.backorders
        )
        
        # Add 10% random variation to prevent identical orders
        variation = np.random.uniform(-0.1, 0.1) * base_order
        return max(0, int(base_order + variation))