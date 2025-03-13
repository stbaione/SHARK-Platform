# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base class for kv caches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
from typing import List, Iterable

from .page_pool import PageInfo, PagePool


logger = logging.getLogger(__name__)


# exception for when cache allocation failed
class CacheAllocationFailure(Exception):
    pass


@dataclass
class RefCount:
    """
    A reference counter to replace simple int.
    """

    count: int = 0

    def increment(self) -> int:
        self.count += 1
        return self.count

    def decrement(self) -> int:
        self.count -= 1
        return self.count

    def is_empty(self) -> bool:
        return self.count <= 0


class PageAllocation(ABC):
    """Abstract base class for page allocations in the cache."""

    @property
    @abstractmethod
    def pages(self) -> List[PageInfo]:
        """Returns the list of pages that were allocated."""
        pass

    @abstractmethod
    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        """
        Makes pages available to other requests. For details, reference the derived class in trie_attention_cache.py.
        """
        pass

    @abstractmethod
    def release_pages(self) -> None:
        """Releases the allocation's reference to pages."""
        pass

    @abstractmethod
    def extend_allocation(self, tokens, *, extra_token_slots=0) -> None:
        """
        Extends the allocation to include additional tokens. For details, reference the derived class in trie_attention_cache.py.
        """
        pass


class BasePagedAttentionCacheAllocation(PageAllocation):
    """Represents a page allocation in the cache."""

    def __init__(self, pages: Iterable[PageInfo], cache: "BasePagedAttentionCache"):
        self._pages = tuple(pages)
        self._cache = cache
        self._is_released = False

    @property
    def pages(self) -> List[PageInfo]:
        return list(self._pages)

    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        pass

    def release_pages(self) -> None:
        if self._is_released:
            logger.warning("Releasing already-released allocation")
            return
        self._cache.free_pages(self._pages)
        self._is_released = True

    def extend_allocation(self, tokens, *, extra_token_slots=0) -> None:
        # assert old tokens are a prefix of incoming tokens
        # if we don't have enough pages to hold the tokens, we need to allocate more pages
        token_count = len(tokens) + extra_token_slots
        pages_needed = math.ceil(token_count / self._cache.tokens_per_page)
        if pages_needed > len(self._pages):
            new_pages = self._cache.page_pool.acquire_free_pages(
                pages_needed - len(self._pages)
            )
            if new_pages is None:
                raise CacheAllocationFailure()
            if self._cache.ref_counts is not None:
                self._cache.increment_pages(new_pages)

            self._pages += tuple(new_pages)

    def __rerp__(self) -> str:
        return f"BasePagedAttentionCacheAllocation(pages={self._pages}, cache={self._cache})"


class BasePagedAttentionCache:
    """
    Manages lifecycle of pages (using PageInfo as handles).


    Page States:
        Caching - Page can be read by multiple threads
            - Also maintains a reference count
        Writing - Page is being modified by a single owner thread

    Transitions:
        Caching -> Writing: When acquiring an unreferenced LRU leaf page for writing
        Writing -> Caching: When writing is complete and page is released

    Thread Safety:
        - Multiple readers allowed in ReadableCaching state
        - Single writer exclusive access in Writing state
        - Reference counting prevents eviction of in-use pages
    """

    def __init__(self, page_pool: PagePool, tokens_per_page: int, ref_count=False):
        self.page_pool = page_pool
        self.tokens_per_page = tokens_per_page
        self.ref_counts = (
            None
            if not ref_count
            else [RefCount() for _ in range(len(self.page_pool.attn_page_entries))]
        )

    def acquire_pages_for_tokens(
        self, tokens: List[int], extra_token_slots: int = 1
    ) -> PageAllocation:
        """
        Given a list of tokens, return a list of pages and a start position to continue generation from.

        Parameters:
        - tokens: all the known tokens for this generation request
        - extra_token_slots: number of kvcache slots needed in addition to the ones needed to hold the given tokens.

        In the base implementation, this will just allocate all new pages, but in shared-kv implementations, we will fetch cached pages if applicable.

        The pages are returned in order.

        No token at idx < n_cached_token should be written to. TODO: consider enforcing this.
        """
        token_count = len(tokens)
        pages_needed = math.ceil(token_count / self.tokens_per_page)
        pages = self.page_pool.acquire_free_pages(pages_needed)

        if pages is None:
            raise CacheAllocationFailure()

        if self.ref_counts is not None:
            for page in pages:
                self.ref_counts[page.index].increment()

        return BasePagedAttentionCacheAllocation(pages, cache=self)

    def increment_pages(self, pages: List[PageInfo]):
        for page in pages:
            self.ref_counts[page.index].increment()

    def decrement_pages(
        self, pages: List[PageInfo], return_empty_pages: bool = False
    ) -> None | List[PageInfo]:
        if return_empty_pages:
            empty_pages = []
        for page in pages:
            self.ref_counts[page.index].decrement()
            if return_empty_pages and self.ref_counts[page.index].is_empty():
                empty_pages.append(page)

        return empty_pages if return_empty_pages else None

    def free_pages(self, pages: List[PageInfo]):
        if self.ref_counts is None:
            self.page_pool.free_pages(pages)
            return

        pages_to_free = self.decrement_pages(
            pages,
            return_empty_pages=True,
        )
        self.page_pool.free_pages(pages_to_free)

    def fork_pages(self, pages: List[PageInfo]) -> List[PageInfo]:
        new_pages = pages.copy()
        last_page = new_pages.pop(-1)
        new_page = self.page_pool.copy_page(last_page)
        if new_page is None:
            raise CacheAllocationFailure()

        new_pages.append(new_page)
        self.increment_pages(new_pages)
        return new_pages
