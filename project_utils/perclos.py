import time

class PERCLOS:
    def __init__(self, window_seconds=60, threshold=0.4):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.eyes_closed_times = []

    def update(self, eyes_closed: bool):
        current_time = time.time()
        self.eyes_closed_times.append((current_time, eyes_closed))
        self._prune_old_entries(current_time)

    def _prune_old_entries(self, current_time):
        self.eyes_closed_times = [
            (t, closed) for t, closed in self.eyes_closed_times
            if current_time - t <= self.window_seconds
        ]

    def get_score(self) -> float:
        if not self.eyes_closed_times:
            return 0.0
        closed_count = sum(1 for _, closed in self.eyes_closed_times if closed)
        total_count = len(self.eyes_closed_times)
        return closed_count / total_count if total_count > 0 else 0.0

    def is_drowsy(self) -> bool:
        return self.get_score() > self.threshold
