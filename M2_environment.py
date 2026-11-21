import numpy as np

class CustomerDemandEnv:
    def __init__(self, product_config, seed=42):
        self.config = product_config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.unit_cost = product_config["unit_cost"]
        self.base_p0 = product_config.get("base_p0", (product_config["min_price"] + product_config["max_price"]) / 2.0)
        self.elasticity = product_config.get("price_elasticity", 1.5)
        self.competitor_price = product_config.get("competitor_price", self.base_p0)

    def generate_context(self, day_of_week):
        """
        Generates realistic visitor demographic & environment context vector:
        - Device: Mobile (60%) vs Desktop (40%)
        - User Tier: Metro (55%) vs Tier-2/3 (45%)
        - Channel: Organic (50%), Paid Ad (30%), Direct (20%)
        - Weekend flag
        """
        is_mobile = 1.0 if self.rng.random() < 0.60 else 0.0
        is_metro = 1.0 if self.rng.random() < 0.55 else 0.0
        is_weekend = 1.0 if day_of_week in [5, 6] else 0.0
        channel_rand = self.rng.random()
        is_paid = 1.0 if channel_rand < 0.30 else 0.0

        # Context vector: [Intercept, Is_Mobile, Is_Metro, Is_Weekend]
        context_vector = np.array([1.0, is_mobile, is_metro, is_weekend])
        context_dict = {
            "device": "Mobile" if is_mobile else "Desktop",
            "tier": "Metro" if is_metro else "Tier-2/3",
            "is_weekend": bool(is_weekend),
            "channel": "Paid Ad" if is_paid else "Organic"
        }
        return context_vector, context_dict

    def _calculate_conversion_prob(self, price, day_of_week, context_vec=None, competitor_price=None, shock_multiplier=1.0):
        """
        Microeconomic Logistic Demand function incorporating:
        1. Base Price Sensitivity
        2. Contextual shifts (Mobile users more sensitive, Metro users higher willingness-to-pay)
        3. Competitor relative price ratio
        4. Non-stationary market shocks
        """
        p0 = self.base_p0
        comp = competitor_price if competitor_price is not None else self.competitor_price
        
        # Contextual adjustment to midpoint reservation price
        context_shift = 0.0
        if context_vec is not None:
            # context_vec: [1.0, is_mobile, is_metro, is_weekend]
            context_shift += (context_vec[2] * 0.12 * p0)  # Metro buyers tolerate +12% price
            context_shift -= (context_vec[1] * 0.08 * p0)  # Mobile shoppers are 8% more price-sensitive
            context_shift += (context_vec[3] * 0.10 * p0)  # Weekend buyers have +10% higher conversion willingness

        adj_p0 = p0 + context_shift
        
        # Sensitivity scaling
        k = (self.elasticity / (self.config["max_price"] - self.config["min_price"])) * 4.0
        
        # Competitor penalty/advantage: if price > competitor, conversion probability drops
        if comp is not None:
            comp_ratio = (price - comp) / max(comp, 1.0)
            comp_penalty = 1.5 * comp_ratio
        else:
            comp_penalty = 0.0 # No competitor present; penalty disabled

        # Logistic demand equation
        exponent = k * (price - adj_p0) + comp_penalty
        prob = 1.0 / (1.0 + np.exp(np.clip(exponent, -10.0, 10.0)))
        
        # Apply market shock (e.g. flash sale traffic surge or demand drop)
        final_prob = np.clip(prob * shock_multiplier, 0.02, 0.98)
        return float(final_prob)

    def simulate_step(self, price, day_of_week, context_vec=None, competitor_price=None, shock_multiplier=1.0):
        prob = self._calculate_conversion_prob(price, day_of_week, context_vec, competitor_price, shock_multiplier)
        bought = int(self.rng.random() < prob)
        reward = (price - self.unit_cost) if bought else 0.0
        return bought, reward, prob, {}