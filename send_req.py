# TODO: Remove this when PRing, just a convenient local test file

import requests
from concurrent.futures import ThreadPoolExecutor

base_url = "http://localhost:8000/generate"
payload = {"text": "Name the capital of the United States.", "sampling_params": {"max_completion_tokens": 50}}

def send_request():
    response = requests.post(base_url, json=payload)
    return response.text

n = 1

# Use ThreadPoolExecutor to send two requests concurrently
with ThreadPoolExecutor(max_workers=n) as executor:
    # Submit two tasks to send requests
    futures = []
    for _ in range(n):
        futures.append(executor.submit(send_request))

    # Wait for the results and print them
    for i, f in enumerate(futures):
        print(f"Response {i}: {f.result()}")
