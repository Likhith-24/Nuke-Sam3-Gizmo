# cache.py - Mask caching system for H2 SamViT
# Manages frame-by-frame mask storage with memory limits

import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import threading
import psutil


class MaskCache:
    """LRU cache for computed masks with memory management."""
    
    def __init__(self, memory_percent: float = 25.0):
        self._cache: OrderedDict[Tuple[str, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._memory_percent = memory_percent
        self._max_bytes = self._calculate_max_bytes()
        self._current_bytes = 0
    
    def _calculate_max_bytes(self) -> int:
        """Calculate max cache size based on available memory."""
        available = psutil.virtual_memory().available
        return int(available * (self._memory_percent / 100.0))
    
    def set_memory_percent(self, percent: float) -> None:
        """Update memory percentage limit."""
        self._memory_percent = max(10.0, min(75.0, percent))
        self._max_bytes = self._calculate_max_bytes()
        self._evict_if_needed()
    
    def _mask_bytes(self, mask: np.ndarray) -> int:
        """Get memory size of a mask."""
        return mask.nbytes
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over memory limit."""
        while self._current_bytes > self._max_bytes and self._cache:
            key, mask = self._cache.popitem(last=False)
            self._current_bytes -= self._mask_bytes(mask)
    
    def store(self, node_name: str, frame: int, mask: np.ndarray) -> None:
        """Store a mask in the cache."""
        key = (node_name, frame)
        mask_copy = mask.astype(np.float32).copy()
        mask_size = self._mask_bytes(mask_copy)
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_mask = self._cache.pop(key)
                self._current_bytes -= self._mask_bytes(old_mask)
            
            # Check if single mask exceeds limit
            if mask_size > self._max_bytes:
                print(f"[Cache] Warning: Mask size ({mask_size / 1024**2:.1f}MB) exceeds cache limit")
                return
            
            # Evict old entries to make room
            while self._current_bytes + mask_size > self._max_bytes and self._cache:
                _, old_mask = self._cache.popitem(last=False)
                self._current_bytes -= self._mask_bytes(old_mask)
            
            # Store new mask
            self._cache[key] = mask_copy
            self._current_bytes += mask_size
    
    def get(self, node_name: str, frame: int) -> Optional[np.ndarray]:
        """Retrieve a mask from the cache."""
        key = (node_name, frame)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                mask = self._cache.pop(key)
                self._cache[key] = mask
                return mask.copy()
        
        return None
    
    def has(self, node_name: str, frame: int) -> bool:
        """Check if a mask is cached."""
        return (node_name, frame) in self._cache
    
    def clear_node(self, node_name: str) -> int:
        """Clear all cached masks for a node. Returns count cleared."""
        count = 0
        
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == node_name]
            for key in keys_to_remove:
                mask = self._cache.pop(key)
                self._current_bytes -= self._mask_bytes(mask)
                count += 1
        
        return count
    
    def clear_all(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "used_bytes": self._current_bytes,
                "max_bytes": self._max_bytes,
                "used_percent": (self._current_bytes / self._max_bytes * 100) if self._max_bytes > 0 else 0,
                "memory_percent": self._memory_percent
            }
    
    def get_cache_info(self) -> str:
        """Get human-readable cache info."""
        stats = self.get_stats()
        used_mb = stats["used_bytes"] / (1024 ** 2)
        max_mb = stats["max_bytes"] / (1024 ** 2)
        
        return f"{used_mb:.1f} MB / {max_mb:.1f} MB ({stats['entries']} frames)"


# Global cache instance
_cache: Optional[MaskCache] = None


def get_cache() -> MaskCache:
    """Get or create the global mask cache."""
    global _cache
    if _cache is None:
        _cache = MaskCache()
    return _cache


def store_mask(node_name: str, frame: int, mask: np.ndarray) -> None:
    """Store a mask in the global cache."""
    get_cache().store(node_name, frame, mask)


def get_mask(node_name: str, frame: int) -> Optional[np.ndarray]:
    """Get a mask from the global cache."""
    return get_cache().get(node_name, frame)


def has_mask(node_name: str, frame: int) -> bool:
    """Check if mask is cached."""
    return get_cache().has(node_name, frame)


def clear_node_cache(node_name: str) -> int:
    """Clear cache for a specific node."""
    return get_cache().clear_node(node_name)


def clear_all_cache() -> None:
    """Clear all cached masks."""
    get_cache().clear_all()


def set_cache_memory_percent(percent: float) -> None:
    """Set cache memory percentage."""
    get_cache().set_memory_percent(percent)


def get_cache_stats() -> Dict[str, any]:
    """Get cache statistics."""
    return get_cache().get_stats()


def estimate_cache_capacity(width: int, height: int, memory_percent: float = 25.0) -> int:
    """
    Estimate number of frames that can be cached at given resolution.
    
    Args:
        width: Frame width
        height: Frame height
        memory_percent: Memory percentage for cache
    
    Returns:
        Estimated number of frames
    """
    available = psutil.virtual_memory().available
    max_bytes = available * (memory_percent / 100.0)
    
    # Assume float32 alpha channel
    bytes_per_frame = width * height * 4  # 4 bytes per float32
    
    return int(max_bytes / bytes_per_frame)
