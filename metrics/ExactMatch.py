from metrics.MetricsBase import Metric


class ExactMatch(Metric):
    def name():
        return "Exact Match"

    def evaluate(predictions, targets)