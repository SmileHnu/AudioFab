"""
Cache Manager for AudioFab Agent System

Intelligent caching system with TTL, LRU eviction, and memory optimization.
"""

import json
import hashlib
import logging
import pickle
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class CacheManager:
    """Intelligent caching system for AudioFab agent operations."""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_index = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_mb": 0
        }
        
    def initialize(self):
        """Initialize cache system."""
        self.cache_dir.mkdir(exist_ok=True)
        self._load_cache_index()
        self._cleanup_expired()
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
                self._update_stats()
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            index_file = self.cache_dir / "index.json"
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _update_stats(self):
        """Update cache statistics."""
        total_size = 0
        for entry in self.cache_index.values():
            total_size += entry.get("size_bytes", 0)
        
        self.cache_stats["total_size_mb"] = total_size / (1024 * 1024)
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache_index.items():
            if now > datetime.fromisoformat(entry["expires_at"]):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_cache_entry(self, key: str):
        """Remove a specific cache entry."""
        if key in self.cache_index:
            entry = self.cache_index[key]
            cache_file = self.cache_dir / f"{key}.cache"
            
            if cache_file.exists():
                cache_file.unlink()
            
            del self.cache_index[key]
            self.cache_stats["evictions"] += 1
    
    def _evict_lru(self):
        """Evict least recently used entries when cache is full."""
        if self.cache_stats["total_size_mb"] < self.max_cache_size_mb:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1]["last_accessed"]
        )
        
        # Remove oldest entries until under limit
        target_size = self.max_cache_size_mb * 0.8  # Keep 80% of limit
        
        for key, entry in sorted_entries:
            if self.cache_stats["total_size_mb"] <= target_size:
                break
            
            self._remove_cache_entry(key)
    
    def generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache_index:
            self.cache_stats["misses"] += 1
            return None
        
        entry = self.cache_index[key]
        
        # Check if expired
        if datetime.now() > datetime.fromisoformat(entry["expires_at"]):
            self._remove_cache_entry(key)
            self.cache_stats["misses"] += 1
            return None
        
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if not cache_file.exists():
                self._remove_cache_entry(key)
                self.cache_stats["misses"] += 1
                return None
            
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            
            # Update access time
            entry["last_accessed"] = datetime.now().isoformat()
            self.cache_stats["hits"] += 1
            
            return value
            
        except Exception as e:
            logger.error(f"Cache retrieval failed for {key}: {e}")
            self._remove_cache_entry(key)
            return None
    
    async def set(self, key: str, value: Any, ttl_hours: int = 24) -> bool:
        """Set value in cache."""
        try:
            # Check cache size and evict if necessary
            self._evict_lru()
            
            # Prepare cache entry
            cache_file = self.cache_dir / f"{key}.cache"
            
            # Serialize value
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            file_size = cache_file.stat().st_size
            
            self.cache_index[key] = {
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
                "size_bytes": file_size,
                "ttl_hours": ttl_hours
            }
            
            # Update stats
            self._update_stats()
            self._save_cache_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache storage failed for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete specific cache entry."""
        try:
            self._remove_cache_entry(key)
            self._save_cache_index()
            return True
        except Exception as e:
            logger.error(f"Cache deletion failed for {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            self.cache_index.clear()
            self._update_stats()
            self._save_cache_index()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.cache_stats,
            "cache_entries": len(self.cache_index),
            "hit_rate": (
                self.cache_stats["hits"] /
                max(self.cache_stats["hits"] + self.cache_stats["misses"], 1)
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific cache entry."""
        return self.cache_index.get(key)
    
    def cleanup(self):
        """Perform cache maintenance."""
        self._cleanup_expired()
        self._evict_lru()
        self._save_cache_index()
        logger.info("Cache maintenance completed")
    
    async def batch_set(self, items: List[Dict[str, Any]]) -> int:
        """Set multiple cache entries efficiently."""
        success_count = 0
        
        for item in items:
            key = item.get("key")
            value = item.get("value")
            ttl_hours = item.get("ttl_hours", 24)
            
            if key and value is not None:
                if await self.set(key, value, ttl_hours):
                    success_count += 1
        
        return success_count
    
    async def batch_delete(self, keys: List[str]) -> int:
        """Delete multiple cache entries efficiently."""
        success_count = 0
        
        for key in keys:
            if await self.delete(key):
                success_count += 1
        
        return success_count
    
    def get_size_info(self) -> Dict[str, Any]:
        """Get detailed cache size information."""
        total_entries = len(self.cache_index)
        total_size_mb = self.cache_stats["total_size_mb"]
        
        # Calculate average entry size
        avg_entry_size = total_size_mb / max(total_entries, 1)
        
        # Get size distribution
        size_distribution = {
            "small": 0,  # < 1MB
            "medium": 0,  # 1-10MB
            "large": 0    # > 10MB
        }
        
        for entry in self.cache_index.values():
            size_mb = entry.get("size_bytes", 0) / (1024 * 1024)
            if size_mb < 1:
                size_distribution["small"] += 1
            elif size_mb < 10:
                size_distribution["medium"] += 1
            else:
                size_distribution["large"] += 1
        
        return {
            "total_entries": total_entries,
            "total_size_mb": total_size_mb,
            "average_entry_size_mb": avg_entry_size,
            "size_distribution": size_distribution,
            "utilization_percent": (total_size_mb / self.max_cache_size_mb) * 100
        }