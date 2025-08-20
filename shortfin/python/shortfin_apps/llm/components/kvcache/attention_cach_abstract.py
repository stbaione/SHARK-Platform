# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base classes for attention cache components. This module defines the abstract base classes
for attention cache components used in the ShortFin LLM framework. These classes provide a
foundation for implementing various types of attention caches, including those that has distributed storage support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


class CacheStoreAbstract(ABC):
    """
    Abstract base class for attention cache storage.
    This class defines the interface for storing and retrieving attention cache data.
    """

    pass


@dataclass
class CacheInfo:
    """
    cache information with some metadata about its contents.
    """

    num_tokens: int
    ## slot_ids stores the index in the allocated block for each token.
    ## block_ids stores the allocated block ids, one block id corresponds to number of tokens defined by tokens_per_page.
    slot_ids: list[int]
    page_ids: list[int]
    pool: CacheStoreAbstract


@dataclass
class CacheStoreConfig:
    """
    Configuration for the cache store.
    This class holds the hyperparameters and settings for the cache store.
    """

    max_size: int = 1000  # Maximum number of items in the cache
    eviction_policy: str = "LRU"  # Eviction policy (e.g., LRU, FIFO)
    storage_type: str = "in_memory"  # Type of storage (e.g., in_memory, disk)


class AttentionCacheAbstract(ABC):
    """
    Abstract base class for attention cache components.
    This class defines the interface for attention cache components used in the ShortFin LLM framework.
    """

    @abstractmethod
    def allocate(self, tokens: List[int]) -> CacheInfo:
        """
        This method should allocate space in the cache for the given tokens and return their indices.
        Parameters:
        - tokens: List of token IDs to allocate space for.

        Returns:
        - CacheInfo: An object containing metadata about the allocated cache space.
        """
        pass
