import pandas as pd

class SimulationExporter:
    def __init__(self, log_history, config=None):
        self.log_history = log_history
        self.config = config or {}

    def to_dataframe(self):
        if not self.log_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.log_history)
        # Add metadata configuration attributes
        for key, val in self.config.items():
            df[f"config_{key}"] = val
        return df