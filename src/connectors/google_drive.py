"""Google Drive connector for fetching documents."""

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Iterator

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config.logging_config import get_logger

from .base import BaseConnector, SyncState

logger = get_logger(__name__)


@dataclass
class DriveFile:
    """Represents a Google Drive file."""

    id: str
    name: str
    mime_type: str
    created_time: datetime | None
    modified_time: datetime | None
    size: int | None
    parents: list[str]
    web_view_link: str | None


# MIME types that can be exported to text
EXPORTABLE_MIME_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# MIME types we can extract text from
SUPPORTED_MIME_TYPES = [
    # Google Workspace
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    # Microsoft Office
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    # PDF and text
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
]


class GoogleDriveConnector(BaseConnector):
    """Connector for Google Drive files."""

    def __init__(self, credentials: Credentials):
        """Initialize the connector.

        Args:
            credentials: Google OAuth credentials
        """
        self.service = build("drive", "v3", credentials=credentials)
        self._sync_state = SyncState()

    def list_items(
        self,
        since: datetime | None = None,
        page_size: int = 100,
        folder_id: str | None = None,
    ) -> Iterator[DriveFile]:
        """List files from Google Drive.

        Args:
            since: Only return files modified after this time
            page_size: Number of files per page
            folder_id: Optional folder ID to filter by

        Yields:
            DriveFile objects
        """
        # Build query
        query_parts = [f"mimeType != 'application/vnd.google-apps.folder'"]

        if since:
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S")
            query_parts.append(f"modifiedTime > '{since_str}'")

        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        query = " and ".join(query_parts)

        page_token = None
        total_items = 0

        while True:
            try:
                response = self.service.files().list(
                    q=query,
                    pageSize=page_size,
                    pageToken=page_token,
                    fields="nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, size, parents, webViewLink)",
                    orderBy="modifiedTime desc",
                ).execute()

                files = response.get("files", [])
                logger.debug("drive_list_page", count=len(files))

                for file_data in files:
                    # Skip unsupported file types
                    if file_data.get("mimeType") not in SUPPORTED_MIME_TYPES:
                        continue

                    yield self._parse_file(file_data)
                    total_items += 1

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            except Exception as e:
                logger.error("drive_list_error", error=str(e))
                self._sync_state.errors.append(str(e))
                raise

        self._sync_state.items_synced = total_items
        self._sync_state.last_sync = datetime.now()
        logger.info("drive_list_complete", total=total_items)

    def get_item(self, item_id: str) -> DriveFile:
        """Get a single file by ID.

        Args:
            item_id: Google Drive file ID

        Returns:
            DriveFile object
        """
        try:
            file_data = self.service.files().get(
                fileId=item_id,
                fields="id, name, mimeType, createdTime, modifiedTime, size, parents, webViewLink",
            ).execute()
            return self._parse_file(file_data)
        except Exception as e:
            logger.error("drive_get_error", file_id=item_id, error=str(e))
            raise

    def get_content(self, item_id: str) -> str:
        """Get text content of a file.

        Args:
            item_id: Google Drive file ID

        Returns:
            Text content of the file
        """
        file = self.get_item(item_id)
        content_bytes = self.download_file(item_id, file.mime_type)

        # For plain text, decode directly
        if file.mime_type in ["text/plain", "text/markdown", "text/csv"]:
            return content_bytes.decode("utf-8")

        # For Google Docs exported as text
        if file.mime_type in EXPORTABLE_MIME_TYPES:
            return content_bytes.decode("utf-8")

        # For other formats, return empty (will be handled by extractors)
        return ""

    def download_file(self, file_id: str, mime_type: str) -> bytes:
        """Download file content.

        Args:
            file_id: Google Drive file ID
            mime_type: MIME type of the file

        Returns:
            File content as bytes
        """
        try:
            # Export Google Workspace files
            if mime_type in EXPORTABLE_MIME_TYPES:
                export_mime = EXPORTABLE_MIME_TYPES[mime_type]
                request = self.service.files().export_media(
                    fileId=file_id, mimeType=export_mime
                )
            else:
                # Download binary files
                request = self.service.files().get_media(fileId=file_id)

            buffer = BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            buffer.seek(0)
            content = buffer.read()
            logger.debug("drive_download_complete", file_id=file_id, size=len(content))
            return content

        except Exception as e:
            logger.error("drive_download_error", file_id=file_id, error=str(e))
            raise

    def get_folder_contents(self, folder_id: str) -> Iterator[DriveFile]:
        """Get all files in a folder recursively.

        Args:
            folder_id: Google Drive folder ID

        Yields:
            DriveFile objects
        """
        yield from self.list_items(folder_id=folder_id)

    def get_sync_state(self) -> SyncState:
        """Get current sync state."""
        return self._sync_state

    def _parse_file(self, file_data: dict) -> DriveFile:
        """Parse API response into DriveFile.

        Args:
            file_data: Raw API response

        Returns:
            DriveFile object
        """
        created_time = None
        modified_time = None

        if file_data.get("createdTime"):
            created_time = datetime.fromisoformat(
                file_data["createdTime"].replace("Z", "+00:00")
            )
        if file_data.get("modifiedTime"):
            modified_time = datetime.fromisoformat(
                file_data["modifiedTime"].replace("Z", "+00:00")
            )

        return DriveFile(
            id=file_data["id"],
            name=file_data.get("name", ""),
            mime_type=file_data.get("mimeType", ""),
            created_time=created_time,
            modified_time=modified_time,
            size=int(file_data["size"]) if file_data.get("size") else None,
            parents=file_data.get("parents", []),
            web_view_link=file_data.get("webViewLink"),
        )
