import numpy as np

class ProductCatalog:
    PRESETS = {
        "Wireless Earbuds": {
            "name": "Wireless Earbuds",
            "unit_cost": 1200.0,
            "min_price": 1500.0,
            "max_price": 3500.0,
            "initial_stock": 500,
            "price_elasticity": 1.8,
            "competitor_price": 2400.0,
            "base_p0": 2200.0
        },
        "Smart Watch": {
            "name": "Smart Watch",
            "unit_cost": 4500.0,
            "min_price": 5500.0,
            "max_price": 12000.0,
            "initial_stock": 300,
            "price_elasticity": 1.5,
            "competitor_price": 8500.0,
            "base_p0": 7800.0
        },
        "Limited Edition Sneakers": {
            "name": "Limited Edition Sneakers",
            "unit_cost": 4000.0,
            "min_price": 7000.0,
            "max_price": 20000.0,
            "initial_stock": 100,
            "price_elasticity": 0.9,  # Inelastic (luxury/hype)
            "competitor_price": 14500.0,
            "base_p0": 12500.0
        },
        "Noise-Cancelling Headphones": {
            "name": "Noise-Cancelling Headphones",
            "unit_cost": 6500.0,
            "min_price": 8000.0,
            "max_price": 18000.0,
            "initial_stock": 250,
            "price_elasticity": 2.1,
            "competitor_price": 13000.0,
            "base_p0": 11500.0
        }
    }

    @classmethod
    def get_preset_names(cls):
        return list(cls.PRESETS.keys())

    @classmethod
    def load_preset(cls, name):
        cfg = cls.PRESETS[name].copy()
        cfg["price_arms"] = np.round(
            np.linspace(cfg["min_price"], cfg["max_price"], 5), -1
        ).tolist()
        return cfg

    @classmethod
    def create_custom_product(cls, name, unit_cost, min_price, max_price, initial_stock, price_elasticity, competitor_price=None):
          
        arms = np.round(np.linspace(min_price, max_price, 5), -1).tolist()
        return {
            "name": name,
            "unit_cost": float(unit_cost),
            "min_price": float(min_price),
            "max_price": float(max_price),
            "initial_stock": int(initial_stock),
            "price_elasticity": float(price_elasticity),
            "competitor_price": float(competitor_price) if competitor_price is not None else None,
            "base_p0": float((min_price + max_price) / 2.0),
            "price_arms": arms
        }