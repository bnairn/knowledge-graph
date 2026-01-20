"""Incremental sync and change detection."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import xxhash

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SourceSyncState:
    """Sync state for a single source."""

    last_sync: str | None = None
    items_synced: int = 0
    page_token: str | None = None


@dataclass
class SyncState:
    """Overall sync state."""

    drive: SourceSyncState = field(default_factory=SourceSyncState)
    gmail: SourceSyncState = field(default_factory=SourceSyncState)
    processed_hashes: dict[str, str] = field(default_factory=dict)


class SyncManager:
    """Manages incremental sync state and change detection."""

    def __init__(self, state_file: Path):
        """Initialize the sync manager.

        Args:
            state_file: Path to persist sync state
        """
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> SyncState:
        """Load sync state from file.

        Returns:
            SyncState, either loaded or new
        """
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                state = SyncState(
                    drive=SourceSyncState(**data.get("drive", {})),
                    gmail=SourceSyncState(**data.get("gmail", {})),
                    processed_hashes=data.get("processed_hashes", {}),
                )
                logger.info("sync_state_loaded", file=str(self.state_file))
                return state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("sync_state_load_failed", error=str(e))

        return SyncState()

    def _save_state(self) -> None:
        """Persist sync state to file."""
        data = {
            "drive": asdict(self.state.drive),
            "gmail": asdict(self.state.gmail),
            "processed_hashes": self.state.processed_hashes,
        }
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug("sync_state_saved")

    def get_last_sync(self, source: str) -> datetime | None:
        """Get last sync time for a source.

        Args:
            source: Source name ('drive' or 'gmail')

        Returns:
            Last sync datetime or None
        """
        source_state = getattr(self.state, source, None)
        if source_state and source_state.last_sync:
            return datetime.fromisoformat(source_state.last_sync)
        return None

    def update_sync_state(
        self,
        source: str,
        sync_time: datetime,
        items_synced: int,
        page_token: str | None = None,
    ) -> None:
        """Update sync state after successful sync.

        Args:
            source: Source name
            sync_time: Time of sync
            items_synced: Number of items synced
            page_token: Optional page token for resumption
        """
        source_state = getattr(self.state, source)
        source_state.last_sync = sync_time.isoformat()
        source_state.items_synced = items_synced
        source_state.page_token = page_token
        self._save_state()

        logger.info(
            "sync_state_updated",
            source=source,
            sync_time=sync_time.isoformat(),
            items_synced=items_synced,
        )

    def compute_hash(self, content: str | bytes) -> str:
        """Compute hash of content for change detection.

        Args:
            content: Content to hash

        Returns:
            Hex digest of hash
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return xxhash.xxh64(content).hexdigest()

    def should_process(self, item_id: str, content_hash: str) -> bool:
        """Check if item needs processing based on hash.

        Args:
            item_id: Unique item identifier
            content_hash: Hash of current content

        Returns:
            True if item should be processed (new or changed)
        """
        stored_hash = self.state.processed_hashes.get(item_id)

        if stored_hash is None:
            logger.debug("item_is_new", item_id=item_id)
            return True

        if stored_hash != content_hash:
            logger.debug("item_changed", item_id=item_id)
            return True

        logger.debug("item_unchanged", item_id=item_id)
        return False

    def record_processed(self, item_id: str, content_hash: str) -> None:
        """Record that an item was processed.

        Args:
            item_id: Unique item identifier
            content_hash: Hash of processed content
        """
        self.state.processed_hashes[item_id] = content_hash
        self._save_state()

    def clear_state(self) -> None:
        """Clear all sync state."""
        self.state = SyncState()
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("sync_state_cleared")

    def get_stats(self) -> dict:
        """Get sync statistics.

        Returns:
            Dictionary with sync stats
        """
        return {
            "drive_last_sync": self.state.drive.last_sync,
            "drive_items_synced": self.state.drive.items_synced,
            "gmail_last_sync": self.state.gmail.last_sync,
            "gmail_items_synced": self.state.gmail.items_synced,
            "total_processed_items": len(self.state.processed_hashes),
        }
