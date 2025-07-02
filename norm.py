import numpy as np


class MeanStdNormalizer:
    def __init__(self):
        """Initialize mean and standard deviation normalizer"""
        self.mean = None
        self.std = None

    def fit(self, data):
        """Compute mean and standard deviation from data"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        """Normalize data using precomputed mean and standard deviation"""
        return (data - self.mean) / (self.std + 1e-8)  # Avoid division by zero

    def fit_transform(self, data):
        """Compute statistics and transform data in one step"""
        self.fit(data)
        return self.transform(data)


class MinMaxNormalizer:
    def __init__(self):
        """Initialize min-max normalizer"""
        self.min = None
        self.max = None

    def fit(self, data):
        """Compute minimum and maximum values from data"""
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        """Normalize data to [0, 1] range using precomputed min and max"""
        return (data - self.min) / (self.max - self.min + 1e-8)  # Avoid division by zero

    def fit_transform(self, data):
        """Compute min/max and transform data in one step"""
        self.fit(data)
        return self.transform(data)


class LogNormalizer:
    def __init__(self):
        """Initialize logarithmic normalizer"""
        self.min = None
        self.max = None

    def fit(self, data):
        """Compute min and max after logarithmic transformation"""
        log_data = np.log(1 + np.abs(data))  # Log transformation
        self.min = np.min(log_data, axis=0)
        self.max = np.max(log_data, axis=0)

    def transform(self, data):
        """Apply logarithmic transformation and normalize"""
        log_data = np.log(1 + np.abs(data))  # Log transformation
        return (log_data - self.min) / (self.max - self.min + 1e-8)  # Avoid division by zero

    def fit_transform(self, data):
        """Compute log-transformed statistics and transform data"""
        self.fit(data)
        return self.transform(data)


class DynamicNormalizer:
    def __init__(self, window_size):
        """Initialize dynamic normalizer with sliding window"""
        self.window_size = window_size
        self.window = []

    def update(self, data):
        """Update sliding window with new data point"""
        self.window.append(data)
        if len(self.window) > self.window_size:
            self.window.pop(0)  # Remove oldest data point

    def normalize(self, data):
        """Normalize data using statistics from sliding window"""
        if len(self.window) == 0:
            return data  # Return original data if window is empty
        mean = np.mean(self.window, axis=0)
        std = np.std(self.window, axis=0)
        return (data - mean) / (std + 1e-8)  # Avoid division by zero