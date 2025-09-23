import concurrent.futures
import json
import logging
import pytest
import requests

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

from ...datasets import Dataset, DatasetRequest, DatasetTypes, AvailableDatasets
from ...model_management import AccuracyValidationException, ModelConfig
from ...server_management import ServerConfig


logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    max_completion_tokens: int = 250
    temperature: float = 1.0


@dataclass
class AccuracyResults:
    total_prompts: int
    total_correct: int
    total_incorrect: int
    accuracy: float

    def __str__(self) -> str:
        return (
            "Results:\n"
            f"\tTotal Prompts: {self.total_prompts}\n"
            f"\tTotal Correct: {self.total_correct}\n"
            f"\tTotal Incorrect: {self.total_incorrect}\n"
            f"\tAccuracy: {self.accuracy:.2f}%\n"
        )


BASIC = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.BASIC,
    dataset_path=Path(__file__).parent / "datasets.json",
)

CHUNKED_PREFILL = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.CHUNKED_PREFILL,
    dataset_path=Path(__file__).parent / "datasets.json",
)

PREFIX_MATCHING = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.PREFIX_MATCHING,
    dataset_path=Path(__file__).parent / "datasets.json",
)

ALL = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.ALL,
    dataset_path=Path(__file__).parent / "datasets.json",
)


class TestLLMAccuracy:
    def _check_health(self, base_url: str):
        resp = requests.get(f"{base_url}/health")
        assert resp.status_code == 200

    def _validate_response(self, dataset: Dataset, results: List[Dict]):
        total_prompts = dataset.size
        total_correct = 0
        total_incorrect = 0

        for result in results:
            responses = result["responses"]
            for response in responses:
                prompt = response["prompt"]

                expected_generation = dataset.get_expected_generation(prompt)
                actual_generation = response["responses"][0]["text"]

                if expected_generation != actual_generation:
                    logger.error(
                        f"Mismatch for prompt: {prompt}\n"
                        f"\tExpected: {expected_generation}\n"
                        f"\tActual: {actual_generation}\n"
                    )
                    total_incorrect += 1
                    continue

                total_correct += 1

        return AccuracyResults(
            total_prompts=total_prompts,
            total_correct=total_correct,
            total_incorrect=total_incorrect,
            accuracy=(total_correct / total_prompts) * 100,
        )

    def _send_request(
        self, base_url: str, prompts_batch: List[str], sampling_params: SamplingParams
    ):
        endpoint = f"{base_url}/generate"
        payload = {
            "text": prompts_batch,
            "sampling_params": asdict(sampling_params),
        }

        logger.info(f"Sending request with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return json.loads(response.text)

    def _request_loop(
        self,
        base_url: str,
        dataset: Dataset,
        sampling_params: SamplingParams,
        num_workers,
    ):
        batch_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {}
            for batch in dataset:
                future = executor.submit(
                    self._send_request, base_url, list(batch.keys()), sampling_params
                )
                future_to_batch[future] = batch

            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_batch)
            ):
                batch = future_to_batch[future]
                try:
                    result_dict = future.result()
                    logger.info(f"Batch {i} - Prompts:")
                    for prompt in batch.keys():
                        logger.info(prompt)

                    logger.info(f"Batch {i} - Responses:")
                    logger.info(json.dumps(result_dict, indent=2))
                    batch_responses.append(result_dict)
                    print("-" * 80)
                except Exception as e:
                    logger.error(f"Batch {i} generated an exception: {e}")
                    print("-" * 80)

            return batch_responses

    @pytest.mark.parametrize(
        "dataset_request,batch_size,num_workers",
        [
            # Run against basic, short test prompts (basic accuracy)
            (BASIC, 4, 2),
            # Run against prompts optimized for `chunked_prefill`
            (CHUNKED_PREFILL, 4, 2),
            # Run against prompts optimized for `prefix_matching`
            (PREFIX_MATCHING, 4, 2),
            # Test against large set of prompts
            # (make sure we don't "drift" over time)
            (ALL, 4, 2),
        ],
        ids=[
            "BASIC-bs4-n2",
            "CHUNKED_PREFILL-bs4-n2",
            "PREFIX_MATCHING-bs4-n2",
            "ALL-bs4-n2",
        ],
    )
    @pytest.mark.parametrize(
        "model_artifacts,server",
        [
            (
                ModelConfig.get(name="meta_llama3.1_8b_instruct"),
                {"prefix_sharing_algorithm": "none"},
            ),  # noqa: E501
            (
                ModelConfig.get(name="meta_llama3.1_8b_instruct"),
                {"prefix_sharing_algorithm": "trie"},
            ),  # noqa: E501
        ],
        ids=[
            "meta_llama3.1_8b_instruct-no_prefix_sharing",
            "meta_llama3.1_8b_instruct-trie_prefix_sharing",
        ],
        indirect=True,
    )
    def test_accuracy(
        self,
        dataset_request: DatasetRequest,
        batch_size: int,
        num_workers: int,
        server,
    ):
        process, port, config = server
        assert process.poll() is None, "Server process terminated unexpectedly."

        base_url = f"http://localhost:{port}"
        self._check_health(base_url)

        dataset_instance = Dataset(dataset_request, batch_size=batch_size)
        sampling_params = SamplingParams()

        server_results = self._request_loop(
            base_url, dataset_instance, sampling_params, num_workers
        )
        accuracy_results = self._validate_response(dataset_instance, server_results)

        logger.info(
            f"Dataset: {dataset_request.dataset.name} | "
            f"Batch Size: {batch_size} | "
            f"Num Workers: {num_workers}\n"
            f"{accuracy_results}"
        )

        if accuracy_results.accuracy < 100.0:
            logger.error(f"Accuracy below 100%: {accuracy_results.accuracy:.2f}%")
            raise AccuracyValidationException(
                f"Accuracy below 100%: {accuracy_results.accuracy:.2f}%"
            )
