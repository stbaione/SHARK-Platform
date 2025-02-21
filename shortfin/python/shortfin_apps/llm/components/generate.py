# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import io
import json
import logging
import numpy as np
import pdb

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .io_struct import GenerateReqInput
from .messages import InferenceExecRequest, InferencePhase
from .service import GenerateService
from .tokenizer import Encoding

logger = logging.getLogger(__name__)


class GenerateItemProcess(sf.Process):
    """Process instantiated for each generation sequence.

    This process breaks the sequence into individual inference and sampling
    steps, submitting them to the batcher and marshaling incremental/final
    results.
    """

    def __init__(
        self,
        client: "ClientGenerateBatchProcess",
        gen_req: GenerateReqInput,
        index: int,
        input_token_ids: list[list[int]],
        max_completion_tokens: int,
        eos_token_id: int,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.input_token_ids = input_token_ids
        self.n_beams = self.client.model_params.n_beams
        self.result_token_ids: list[list[int]] = [[] for _ in range(self.n_beams)]
        self.max_completion_tokens = max_completion_tokens
        self.eos_token_id = eos_token_id

        self.streamed_tokens_index = 0

    def topk(self, logits: sfnp.device_array, k: int, axis: int):
        # TODO: Move this implementation to the sfnp.array
        partitioned_indices = np.argpartition(logits, -k, axis=axis)
        axis_len = logits.shape[axis]
        idx_topk = [slice(None)] * logits.ndim
        idx_topk[axis] = slice(axis_len - k, axis_len)

        topk_indices = partitioned_indices[tuple(idx_topk)]
        topk_values = np.take_along_axis(logits, topk_indices, axis=axis)
        sort_order = np.argsort(-topk_values, axis=axis)
        sorted_values = np.take_along_axis(topk_values, sort_order, axis=axis)
        sorted_indices = np.take_along_axis(topk_indices, sort_order, axis=axis)

        return sorted_values, sorted_indices

    def log_softmax(self, logits: sfnp.device_array):
        # TODO: Move this to sfnp.array
        c = logits.max()
        logsumexp = np.log(np.exp(logits - c).sum())
        return logits - c - logsumexp

    async def run(self):
        exec = InferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=self.input_token_ids,
            rid=self.gen_req.rid,
            n_beams=self.n_beams,
        )
        try:
            self.client.batcher.submit(exec)
            await exec.done

            # Prefill result.
            token = sfnp.argmax(exec.result_logits)
            token_int = token.items[0]
            exec.input_token_ids.append(token_int)
            # TODO: Start positions should be a list
            exec.start_position = len(exec.input_token_ids) - 1
            self.client.stream_results(self)

            for beam in self.result_token_ids:
                beam.append(token_int)
            exec.replicate_for_beam_search()

            # Decode loop.
            n_beams = self.n_beams
            for i in range(self.max_completion_tokens):
                exec.reset(InferencePhase.DECODE)
                self.client.batcher.submit(exec)
                await exec.done
                # Greedy selection
                if n_beams == 1:
                    token = sfnp.argmax(exec.result_logits)
                    token_int = token.items[0]
                    self.result_token_ids[0].append(token_int)
                    if token_int == self.eos_token_id:
                        break
                    exec.input_token_ids[0].append(token_int)
                    exec.start_position += 1
                else:
                    logits = np.array(exec.result_logits)
                    # Apply log-softmax to raw logits
                    logits = self.log_softmax(logits)
                    # Find top-k scoring logits per beam
                    values, tokens = self.topk(logits, n_beams, axis=-1)
                    values, tokens = np.squeeze(values, 1), np.squeeze(tokens, 1)
                    values += exec.cumulative_log_probs
                    values_flat = values.flatten()
                    top_indices = np.argsort(values_flat)[-n_beams:][::-1]

                    new_beams = []
                    new_log_probs = []
                    for idx in top_indices:
                        old_beam_idx = idx // n_beams
                        token_idx = idx % n_beams

                        chosen_log_prob = values[old_beam_idx, token_idx]
                        token_int = tokens[old_beam_idx, token_idx]

                        if token_int == self.eos_token_id:
                            exec.completed_beams.add(old_beam_idx)
                            if len(exec.completed_beams) == n_beams:
                                break
                            continue

                        new_beam_tokens = exec.input_token_ids[old_beam_idx] + [
                            token_int
                        ]
                        new_log_prob = (
                            exec.cumulative_log_probs[old_beam_idx][0] + chosen_log_prob
                        )

                        new_beams.append(new_beam_tokens)
                        new_log_probs.append([new_log_prob])

                    exec.input_token_ids = new_beams
                    exec.cumulative_log_probs = new_log_probs
                    exec.start_position += 1
                    self.result_token_ids = exec.input_token_ids
                self.client.stream_results(self)

        finally:
            exec.free_cache_pages()

    # TODO: Modify this for beam case
    def append_token(self, token: int):
        self.result_token_ids.append(token)
        self.client.stream_results(self)


class ClientGenerateBatchProcess(sf.Process):
    """Process instantiated for handling a batch from a client.

    This takes care of several responsibilities:

    * Tokenization / Detokenization
    * Splitting the batch into GenerateItemProcesses
    * Streaming responses
    * Final responses
    """

    __slots__ = [
        "batcher",
        "complete_infeed",
        "gen_req",
        "model_params",
        "responder",
        "tokenizer",
    ]

    def __init__(
        self,
        service: GenerateService,
        gen_req: GenerateReqInput,
        responder: FastAPIResponder,
    ):
        super().__init__(fiber=service.main_fiber)
        self.model_params = service.model_params
        self.gen_req = gen_req
        self.responder = responder
        self.tokenizer = service.tokenizer
        self.batcher = service.batcher
        self.complete_infeed = self.system.create_queue()

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)
        streaming = self.gen_req.stream
        if streaming:
            self.responder.start_response()

        try:
            # Launch all individual generate processes and wait for them to finish.
            gen_processes = []
            input_ids = self.gen_req.input_ids
            is_pretokenized = input_ids is not None
            # TODO: We should send this to an executor and await the results.
            if is_pretokenized:
                input_batch = [input_ids] if self.gen_req.is_single else input_ids
            else:
                input_batch = self.tokenize()
            for index, input_tokens in enumerate(input_batch):
                max_completion_tokens = (
                    self.gen_req.sampling_params["max_completion_tokens"]
                    if self.gen_req.is_single
                    else self.gen_req.sampling_params[index]["max_completion_tokens"]
                )
                gen_process = GenerateItemProcess(
                    self,
                    self.gen_req,
                    index,
                    input_tokens if is_pretokenized else input_tokens.ids,
                    max_completion_tokens=max_completion_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_processes.append(gen_process)
                gen_process.launch()

            await asyncio.gather(*gen_processes)

            if streaming:
                logger.debug("Responding to streaming batch")
                self.responder.stream_part(b"data: [DONE]\n\n")
                self.responder.stream_part(None)
            else:
                logging.debug("Responding to one shot batch")
                out = io.BytesIO()
                result_tokens = [p.result_token_ids for p in gen_processes][0]
                if self.gen_req.return_input_ids:
                    if self.gen_req.is_single:
                        result_tokens = result_tokens[0]
                    out.write(bytes(json.dumps(result_tokens), "utf-8"))
                else:
                    pdb.set_trace()
                    result_texts = self.tokenizer.decode(result_tokens)
                    for result_text in result_texts:
                        out.write(b"data: ")
                        out.write(result_text.encode())
                        out.write(b"\n\n")
                self.responder.send_response(out.getvalue())
        finally:
            self.responder.ensure_response()

    def stream_results(self, gen_process: GenerateItemProcess):
        if not self.gen_req.stream:
            return
        out = io.BytesIO()
        result_tokens = gen_process.result_token_ids[
            gen_process.streamed_tokens_index :
        ]
        rid = (
            gen_process.gen_req.rid
            if gen_process.gen_req.is_single
            else gen_process.gen_req.rid[gen_process.index]
        )
        if not self.gen_req.return_input_ids:
            (result_text,) = self.tokenizer.decode([result_tokens])
            out.write(f"data({rid}): ".encode())
            out.write(result_text.encode())
            out.write(b"\n\n")
        else:
            out.write(f"data({rid}): ".encode())
            out.write(str(result_tokens[0]).encode())
            out.write(b"\n\n")
        self.responder.stream_part(out.getvalue())
        gen_process.streamed_tokens_index += len(result_tokens)

    def tokenize(self) -> list[Encoding]:
        gen_req = self.gen_req
        if gen_req.text is not None:
            if self.gen_req.is_single:
                texts = [self.gen_req.text]
                logger.debug("Encoding single request")
            else:
                texts = self.gen_req.text
                logger.debug("Encoding batch of %d", len(texts))
            encodings = self.tokenizer.encode(texts)
            logger.debug("Generated encodings: %r", encodings)
            return encodings
        else:
            raise ValueError("Cannot tokenize 'None' value")
