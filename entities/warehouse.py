import simpy
import numpy as np

class Warehouse:
    def __init__(self, env, retailer, params):
        self.env = env
        self.retailer = retailer
        self.params = params
        self.inventory = simpy.Container(env, init=params['initial_warehouse'], capacity=params['warehouse_capacity'])
        self.backorders = 0  # Tracks unfulfilled demand
        self.shipment_queue = []
        self.pipeline = []  # ✅ FIX: Initialize pipeline attribute

        # Start processes
        self.env.process(self.delivery_check_process())

    def replenish(self, quantity):
        """Replenishes the warehouse inventory correctly."""
        qty_to_add = min(quantity, self.params['warehouse_capacity'] - self.inventory.level)
        if qty_to_add > 0:
            yield self.inventory.put(qty_to_add)  # ✅ Correct way to add stock

    def ship(self, quantity):
        """Ships products to the retailer while handling shortages properly."""
        available_to_ship = min(quantity, self.inventory.level)  # ✅ Ship only what is available
        shortage = quantity - available_to_ship

        if available_to_ship > 0:
            yield self.inventory.get(available_to_ship)  # ✅ Deduct only available stock

        if shortage > 0:
            self.retailer.add_backorder(shortage)  # ✅ Track unfulfilled demand

    def delivery_check_process(self):
        """Checks for completed deliveries and moves them to the retailer."""
        while True:
            yield self.env.timeout(1)  # Daily check
            
            completed_orders = []
            for order in self.pipeline:
                if order['arrives'] <= self.env.now:
                    print(f"✅ DEBUG [{self.env.now}]: Order Delivered! Lead Time → {order['arrives'] - order['created']}")
                    completed_orders.append(order)
                    yield self.env.process(self.retailer.receive_shipment(order['quantity']))

            # ✅ Remove completed orders from pipeline
            for order in completed_orders:
                self.pipeline.remove(order)

    def calculate_reorder_point(self, demand_history):
        """Calculate the reorder point based on demand history and lead time."""
        if not demand_history:
            avg_demand = self.params['average_daily_demand']
        else:
            avg_demand = sum(demand_history) / len(demand_history)
        
        lead_time = self.params.get('shipment_delay', 2)  # Default lead time if not set
        safety_stock = self.params.get('safety_stock_factor', 2.0) * avg_demand

        reorder_point = int(avg_demand * lead_time + safety_stock)
        return reorder_point
