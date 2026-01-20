"""Gmail connector for fetching emails and attachments."""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Iterator

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from config.logging_config import get_logger

from .base import BaseConnector, SyncState

logger = get_logger(__name__)


@dataclass
class EmailAttachment:
    """Represents an email attachment."""

    id: str
    filename: str
    mime_type: str
    size: int


@dataclass
class EmailMessage:
    """Represents a Gmail message."""

    id: str
    thread_id: str
    subject: str
    snippet: str
    from_address: str
    to_addresses: list[str]
    cc_addresses: list[str]
    date: datetime | None
    labels: list[str]
    body_plain: str
    body_html: str
    attachments: list[EmailAttachment] = field(default_factory=list)


class GmailConnector(BaseConnector):
    """Connector for Gmail messages."""

    def __init__(self, credentials: Credentials):
        """Initialize the connector.

        Args:
            credentials: Google OAuth credentials
        """
        self.service = build("gmail", "v1", credentials=credentials)
        self._sync_state = SyncState()

    def list_items(
        self,
        since: datetime | None = None,
        page_size: int = 100,
        query: str | None = None,
        label_ids: list[str] | None = None,
    ) -> Iterator[EmailMessage]:
        """List messages from Gmail.

        Args:
            since: Only return messages after this time
            page_size: Number of messages per page
            query: Gmail search query
            label_ids: Filter by label IDs

        Yields:
            EmailMessage objects
        """
        # Build query
        query_parts = []
        if query:
            query_parts.append(query)
        if since:
            # Gmail uses epoch seconds for date filtering
            epoch = int(since.timestamp())
            query_parts.append(f"after:{epoch}")

        full_query = " ".join(query_parts) if query_parts else None

        page_token = None
        total_items = 0

        while True:
            try:
                request_params = {
                    "userId": "me",
                    "maxResults": page_size,
                }
                if full_query:
                    request_params["q"] = full_query
                if label_ids:
                    request_params["labelIds"] = label_ids
                if page_token:
                    request_params["pageToken"] = page_token

                response = self.service.users().messages().list(**request_params).execute()

                messages = response.get("messages", [])
                logger.debug("gmail_list_page", count=len(messages))

                for msg_ref in messages:
                    # Fetch full message
                    message = self.get_item(msg_ref["id"])
                    yield message
                    total_items += 1

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            except Exception as e:
                logger.error("gmail_list_error", error=str(e))
                self._sync_state.errors.append(str(e))
                raise

        self._sync_state.items_synced = total_items
        self._sync_state.last_sync = datetime.now()
        logger.info("gmail_list_complete", total=total_items)

    def get_item(self, item_id: str) -> EmailMessage:
        """Get a single message by ID.

        Args:
            item_id: Gmail message ID

        Returns:
            EmailMessage object
        """
        try:
            msg_data = self.service.users().messages().get(
                userId="me",
                id=item_id,
                format="full",
            ).execute()
            return self._parse_message(msg_data)
        except Exception as e:
            logger.error("gmail_get_error", message_id=item_id, error=str(e))
            raise

    def get_content(self, item_id: str) -> str:
        """Get text content of a message.

        Args:
            item_id: Gmail message ID

        Returns:
            Plain text body of the message
        """
        message = self.get_item(item_id)
        return message.body_plain or message.snippet

    def get_attachment(self, message_id: str, attachment_id: str) -> bytes:
        """Download an attachment.

        Args:
            message_id: Gmail message ID
            attachment_id: Attachment ID

        Returns:
            Attachment content as bytes
        """
        try:
            attachment = self.service.users().messages().attachments().get(
                userId="me",
                messageId=message_id,
                id=attachment_id,
            ).execute()

            data = attachment.get("data", "")
            content = base64.urlsafe_b64decode(data)
            logger.debug(
                "gmail_attachment_downloaded",
                message_id=message_id,
                size=len(content),
            )
            return content
        except Exception as e:
            logger.error(
                "gmail_attachment_error",
                message_id=message_id,
                attachment_id=attachment_id,
                error=str(e),
            )
            raise

    def get_thread(self, thread_id: str) -> list[EmailMessage]:
        """Get all messages in a thread.

        Args:
            thread_id: Gmail thread ID

        Returns:
            List of EmailMessage objects in the thread
        """
        try:
            thread = self.service.users().threads().get(
                userId="me",
                id=thread_id,
                format="full",
            ).execute()

            messages = []
            for msg_data in thread.get("messages", []):
                messages.append(self._parse_message(msg_data))

            logger.debug("gmail_thread_fetched", thread_id=thread_id, count=len(messages))
            return messages
        except Exception as e:
            logger.error("gmail_thread_error", thread_id=thread_id, error=str(e))
            raise

    def get_sync_state(self) -> SyncState:
        """Get current sync state."""
        return self._sync_state

    def _parse_message(self, msg_data: dict) -> EmailMessage:
        """Parse API response into EmailMessage.

        Args:
            msg_data: Raw API response

        Returns:
            EmailMessage object
        """
        headers = {
            h["name"].lower(): h["value"]
            for h in msg_data.get("payload", {}).get("headers", [])
        }

        # Parse date
        date = None
        if headers.get("date"):
            try:
                date = parsedate_to_datetime(headers["date"])
            except (ValueError, TypeError):
                pass

        # Parse addresses
        to_addresses = self._parse_addresses(headers.get("to", ""))
        cc_addresses = self._parse_addresses(headers.get("cc", ""))

        # Extract body
        body_plain, body_html = self._extract_body(msg_data.get("payload", {}))

        # Extract attachments
        attachments = self._extract_attachments(msg_data.get("payload", {}))

        return EmailMessage(
            id=msg_data["id"],
            thread_id=msg_data.get("threadId", ""),
            subject=headers.get("subject", "(No Subject)"),
            snippet=msg_data.get("snippet", ""),
            from_address=headers.get("from", ""),
            to_addresses=to_addresses,
            cc_addresses=cc_addresses,
            date=date,
            labels=msg_data.get("labelIds", []),
            body_plain=body_plain,
            body_html=body_html,
            attachments=attachments,
        )

    def _parse_addresses(self, address_str: str) -> list[str]:
        """Parse comma-separated email addresses.

        Args:
            address_str: Comma-separated addresses

        Returns:
            List of email addresses
        """
        if not address_str:
            return []
        return [addr.strip() for addr in address_str.split(",") if addr.strip()]

    def _extract_body(self, payload: dict) -> tuple[str, str]:
        """Extract plain text and HTML body from message payload.

        Args:
            payload: Message payload from API

        Returns:
            Tuple of (plain_text, html)
        """
        plain_text = ""
        html_text = ""

        def process_part(part: dict) -> None:
            nonlocal plain_text, html_text

            mime_type = part.get("mimeType", "")
            body = part.get("body", {})
            data = body.get("data", "")

            if data:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                if mime_type == "text/plain":
                    plain_text += decoded
                elif mime_type == "text/html":
                    html_text += decoded

            # Process nested parts
            for sub_part in part.get("parts", []):
                process_part(sub_part)

        process_part(payload)
        return plain_text, html_text

    def _extract_attachments(self, payload: dict) -> list[EmailAttachment]:
        """Extract attachment metadata from message payload.

        Args:
            payload: Message payload from API

        Returns:
            List of EmailAttachment objects
        """
        attachments = []

        def process_part(part: dict) -> None:
            filename = part.get("filename", "")
            body = part.get("body", {})
            attachment_id = body.get("attachmentId")

            if filename and attachment_id:
                attachments.append(
                    EmailAttachment(
                        id=attachment_id,
                        filename=filename,
                        mime_type=part.get("mimeType", "application/octet-stream"),
                        size=body.get("size", 0),
                    )
                )

            for sub_part in part.get("parts", []):
                process_part(sub_part)

        process_part(payload)
        return attachments
