import numpy as np

class InventoryPolicy:
    """Contains implementations of different inventory policies"""
    
    @staticmethod
    def sS_policy(effective_inventory, reorder_point, buffer_stock):
        """
        (s, S) Policy: Order up to S when inventory falls below s
        """
        if effective_inventory < reorder_point:
            return max(0, buffer_stock - effective_inventory)
        return 0

    @staticmethod
    def ROP_policy(effective_inventory, reorder_point, demand, params):
        """
        Reorder Point Policy: Order EOQ when inventory falls below ROP
        """
        if effective_inventory < reorder_point:
            # Annualize demand and holding cost
            annual_demand = demand * 365
            annual_holding_cost = params['holding_cost'] * 365
            
            # Calculate EOQ
            eoq = np.sqrt(
                (2 * annual_demand * params['ordering_cost']) / 
                annual_holding_cost
            )
            return int(max(eoq, reorder_point - effective_inventory))
        return 0

    @classmethod
    def get_policy(cls, policy_name):
        """Factory method for policy selection"""
        return {
            'sS': cls.s_S_policy,
            'ROP': cls.ROP_policy
        }.get(policy_name, cls.s_S_policy)