import simpy
import numpy as np

class Supplier:
    def __init__(self, env, params, warehouse):
        self.env = env
        self.params = params
        self.warehouse = warehouse
        self.current_reliability = params['supplier_reliability']
        self.env.process(self.production_process())

    def production_process(self):
        """Production process with reliability checks"""
        while True:
            yield self.env.timeout(
                max(0.1, np.random.exponential(self.params['production_interval']))
            )
            if np.random.random() <= self.current_reliability:
                batch = min(
                    self.params['production_batch_size'],
                    self.warehouse.inventory.capacity - 
                    self.warehouse.inventory.level
                )
                if batch > 0:
                    yield self.env.process(self.warehouse.replenish(batch))

    def process_order(self, order):
        """Add lead time variability"""
        base_lead_time = self.params['shipment_delay']
        
        # Add random delay (0-2 days)
        delay_variation = np.random.randint(0, 3)
        yield self.env.timeout(base_lead_time + delay_variation)
        
        # Add reliability check after delay
        if np.random.random() > self.current_reliability:
            self.env.process(self.handle_failed_shipment(order))
        else:
            self.warehouse.receive_shipment(order)
    
    def produce(self):
        while True:
            if self.warehouse.inventory.level < self.warehouse.inventory.capacity * 0.2:  # ✅ Increase if below 20% stock
                batch_size = self.params['production_batch_size'] * 1.5  # ✅ Boost production
            else:
                batch_size = self.params['production_batch_size']

            yield self.env.process(self.warehouse.replenish(batch_size))
            print(f"DEBUG [{self.env.now}]: Supplier Produced {batch_size} units")
            yield self.env.timeout(self.params['production_interval'])
