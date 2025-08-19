"""
Simple smoke tests to:
- ensure the full fastapi server works
- ensure the smoke test model works so we know it's not a model issue when another test using this model fails.
"""

import json
import logging
import pytest
import requests
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from ..server_management import ServerConfig
from shortfin_apps.llm.components.io_struct import (
    PromptResponse,
    GeneratedResponse,
    GenerateReqOutput,
)
import urllib3

logger = logging.getLogger(__name__)

from ..model_management import AccuracyValidationException, ModelConfig


pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        (ModelConfig.get(name="tinystories_llama2_25m"), {"prefix_sharing": "none"}),
        (
            ModelConfig.get(name="tinystories_llama2_25m"),
            {
                "prefix_sharing": "none",
                "use_beam_search": True,
                "num_beams": 2,
            },
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m_gpu_argmax"),
            {"prefix_sharing": "none"},
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m_gpu_topk_k4"),
            {"prefix_sharing": "none"},
        ),
        (ModelConfig.get(name="tinystories_llama2_25m"), {"prefix_sharing": "trie"}),
        (
            ModelConfig.get(name="tinystories_llama2_25m"),
            {
                "prefix_sharing": "trie",
                "use_beam_search": True,
                "num_beams": 2,
            },
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m_gpu_argmax"),
            {"prefix_sharing": "trie"},
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m_gpu_topk_k4"),
            {"prefix_sharing": "trie"},
        ),
    ],
    ids=[
        "tinystories_llama2_25m_none",
        "tinystories_llama2_25m_none_beam_search_2_beams",
        "tinystories_llama2_25m_gpu_argmax_none",
        "tinystories_llama2_25m_gpu_topk_k4_none",
        "tinystories_llama2_25m_trie",
        "tinystories_llama2_25m_trie_beam_search_2_beams",
        "tinystories_llama2_25m_gpu_argmax_trie",
        "tinystories_llama2_25m_gpu_topk_k4_trie",
    ],
    indirect=True,
)


# goldens are generated in: https://colab.research.google.com/drive/1pFiyvyIxk1RsHnw5gTk_gu9QiQNy9gfW?usp=sharing
GOLDEN_PROMPT = "Once upon a time"
GOLDEN_RESPONSE = {
    ", there was a little girl named Lily. She loved to play with"
}  # this assumes purely deterministic greedy search

GOLDEN_BEAM_SEARCH_RESPONSE = {
    ", there was a little girl named Lily. She loved to play with",
    ", there was a little girl named Lily. She had a big,",
}  # this assumes purely deterministic beam search with 2 beams


class TestLLMServer:
    """Test suite for LLM server functionality."""

    def test_basic_generation(
        self,
        server: tuple[Any, int, ServerConfig],
    ) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port, config = server
        assert process.poll() is None, "Server process terminated unexpectedly"
        prompt = GOLDEN_PROMPT
        expected_response = (
            GOLDEN_RESPONSE
            if not config.use_beam_search
            else GOLDEN_BEAM_SEARCH_RESPONSE
        )

        response = self._generate(prompt, port)
        response = json.loads(response)
        req_output = GenerateReqOutput(**response)

        for prompt_response in req_output.responses:
            prompt_response = PromptResponse(**prompt_response)
            assert len(prompt_response.responses) == config.num_beams
            for generated_response in prompt_response.responses:
                generated_response = GeneratedResponse(**generated_response)
                response_text = generated_response.text
                if response_text not in expected_response:
                    raise AccuracyValidationException(
                        expected=f"{expected_response}...",
                        actual=response_text,
                        message=f"Generation did not match expected pattern.\nExpected to be one of: {expected_response}\nActual response: '{response_text}'",
                    )

    @pytest.mark.parametrize(
        "concurrent_requests",
        [
            2,
            4,
        ],
    )
    def test_concurrent_generation(
        self,
        server: tuple[Any, int, ServerConfig],
        concurrent_requests: int,
    ) -> None:
        """Tests concurrent text generation requests.

        Args:
            server: Tuple of (process, port) from server fixture
            concurrent_requests: Number of concurrent requests to test
        """
        process, port, config = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        prompt = GOLDEN_PROMPT
        expected_response = (
            GOLDEN_RESPONSE
            if not config.use_beam_search
            else GOLDEN_BEAM_SEARCH_RESPONSE
        )

        def _generate_task(prompt: str, port: int):
            return self._generate(prompt, port)

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(_generate_task, prompt, port)
                for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                response = future.result()
                response = json.loads(response)
                req_output = GenerateReqOutput(**response)

                for prompt_response in req_output.responses:
                    prompt_response = PromptResponse(**prompt_response)
                    assert len(prompt_response.responses) == config.num_beams

                    for generated_response in prompt_response.responses:
                        generated_response = GeneratedResponse(**generated_response)
                        generated_text = generated_response.text
                        if generated_text not in expected_response:
                            raise AccuracyValidationException(
                                expected=f"{expected_response}...",
                                actual=response,
                                message=f"Concurrent generation did not match expected pattern.\nExpected to start with: {expected_response}\nActual response: {response}",
                            )

    # -------- Test switching generation strategies from client ---------- #
    def test_single_greedy_switch(
        self,
        server: tuple[Any, int, ServerConfig],
    ):
        """Tests switching to single-beam greedy generation.

        Args:
            server: Tuple of (process, port, config) from server fixture
        """
        process, port, _ = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        # Test greedy generation
        prompt = GOLDEN_PROMPT
        sampling_params = {
            "max_completion_tokens": 15,
            "temperature": 0.7,
            "num_beams": 1,
            "use_beam_search": False,
        }
        response = self._generate(prompt, port, sampling_params=sampling_params)

        response = json.loads(response)
        req_output = GenerateReqOutput(**response)

        for prompt_response in req_output.responses:
            prompt_response = PromptResponse(**prompt_response)
            assert len(prompt_response.responses) == 1
            for generated_response in prompt_response.responses:
                generated_response = GeneratedResponse(**generated_response)
                response_text = generated_response.text
                if response_text not in GOLDEN_RESPONSE:
                    raise AccuracyValidationException(
                        expected=f"{GOLDEN_RESPONSE}...",
                        actual=response_text,
                        message=f"Greedy generation did not match expected pattern.\nExpected to be one of: {GOLDEN_RESPONSE}\nActual response: {response_text}",
                    )

    def test_beam_search_switch(
        self,
        request: pytest.FixtureRequest,
        server: tuple[Any, int, ServerConfig],
    ):
        """Tests switching to beam search generation.

        Args:
            request: Pytest request object for accessing test metadata
            server: Tuple of (process, port, config) from server fixture
        """
        test_id = request.node.callspec.id
        if "gpu_argmax" in test_id:
            pytest.skip(
                "Beam search with 2 beams isn't compatible with logits returned by GPU argmax model."
            )

        process, port, _ = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        # Test beam search generation
        num_beams = 2
        sampling_params = {
            "max_completion_tokens": 15,
            "temperature": 0.7,
            "num_beams": num_beams,
            "use_beam_search": True,
        }
        prompt = GOLDEN_PROMPT

        response = self._generate(prompt, port, sampling_params=sampling_params)
        response = json.loads(response)
        req_output = GenerateReqOutput(**response)

        for prompt_response in req_output.responses:
            prompt_response = PromptResponse(**prompt_response)
            assert len(prompt_response.responses) == num_beams
            for generated_response in prompt_response.responses:
                generated_response = GeneratedResponse(**generated_response)
                response_text = generated_response.text
                if response_text not in GOLDEN_BEAM_SEARCH_RESPONSE:
                    raise AccuracyValidationException(
                        expected=f"{GOLDEN_BEAM_SEARCH_RESPONSE}...",
                        actual=response_text,
                        message=f"Beam search generation did not match expected pattern.\nExpected to be one of: {GOLDEN_BEAM_SEARCH_RESPONSE}\nActual response: {response_text}",
                    )

    # -------- End Test switching generation strategies from client ---------- #

    def _generate(
        self,
        prompt: str | list[int],
        port: int,
        input_ids: bool = False,
        sampling_params: dict[str, Any] = {
            "max_completion_tokens": 15,
            "temperature": 0.7,
        },
    ) -> str:
        """Helper method to make generation request to server.

        Args:
            prompt: Input text prompt
            port: Server port number

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If request fails
            AccuracyValidationException: If response format is invalid
        """
        payload = {
            "sampling_params": sampling_params,
            "rid": uuid.uuid4().hex,
            "stream": False,
        }
        if input_ids:
            payload["input_ids"] = prompt
        else:
            payload["text"] = prompt
        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,  # Add reasonable timeout
        )
        response.raise_for_status()
        return response.text
