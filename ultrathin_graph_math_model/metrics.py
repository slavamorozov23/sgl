import time

class MetricsCollector:
    """Collects and aggregates latency metrics for FFN, MLA, and Gate operations."""
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize metrics storage
        self.metrics = {
            'ffn': {'time': 0.0, 'calls': 0},
            'mla': {'time': 0.0, 'calls': 0},
            'gate': {'time': 0.0, 'calls': 0},
        }
        # Temporary storage for start times
        self._current = {}

    def start(self, name: str):
        """Mark the start of an operation."""
        self._current[name] = time.perf_counter()

    def end(self, name: str):
        """Mark the end of an operation and accumulate."""
        start = self._current.pop(name, None)
        if start is None:
            return
        elapsed = time.perf_counter() - start
        self.metrics[name]['time'] += elapsed
        self.metrics[name]['calls'] += 1

    def get_and_reset(self):
        """
        Retrieve current metrics and reset counters.
        Returns a dict with keys:
            'ffn_time', 'ffn_calls',
            'mla_time', 'mla_calls',
            'gate_time', 'gate_calls'
        """
        data = {}
        for name, d in self.metrics.items():
            data[f"{name}_time"] = d['time']
            data[f"{name}_calls"] = d['calls']
        # Reset for next window
        self.reset()
        return data

# Global singleton
metrics = MetricsCollector()
