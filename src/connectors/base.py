"""Abstract base class for data source connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


@dataclass
class SyncState:
    """Tracks synchronization state for a connector."""

    last_sync: datetime | None = None
    page_token: str | None = None
    items_synced: int = 0
    errors: list[str] = field(default_factory=list)


class BaseConnector(ABC):
    """Abstract base class for all data source connectors."""

    @abstractmethod
    def list_items(
        self,
        since: datetime | None = None,
        page_size: int = 100,
    ) -> Iterator[Any]:
        """List all items, optionally filtered by modification date.

        Args:
            since: Only return items modified after this time
            page_size: Number of items per page

        Yields:
            Items from the data source
        """
        pass

    @abstractmethod
    def get_item(self, item_id: str) -> Any:
        """Fetch a single item by ID.

        Args:
            item_id: Unique identifier of the item

        Returns:
            The requested item
        """
        pass

    @abstractmethod
    def get_content(self, item_id: str) -> str:
        """Extract text content from an item.

        Args:
            item_id: Unique identifier of the item

        Returns:
            Plain text content of the item
        """
        pass

    @abstractmethod
    def get_sync_state(self) -> SyncState:
        """Get current sync state for incremental sync.

        Returns:
            Current synchronization state
        """
        pass
