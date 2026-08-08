"""
Monitoring utilities for the House Price Prediction API.
"""

import time
from collections import defaultdict


# Store basic API monitoring information
request_count = 0
total_response_time = 0.0
endpoint_counts = defaultdict(int)


def record_request(endpoint: str, response_time: float):
    """
    Record information about an API request.
    """
    global request_count
    global total_response_time

    request_count += 1
    total_response_time += response_time
    endpoint_counts[endpoint] += 1


def get_metrics():
    """
    Return current API monitoring metrics.
    """

    if request_count > 0:
        average_response_time = total_response_time / request_count
    else:
        average_response_time = 0.0

    return {
        "total_requests": request_count,
        "average_response_time_seconds": round(
            average_response_time, 4
        ),
        "endpoint_requests": dict(endpoint_counts),
    }