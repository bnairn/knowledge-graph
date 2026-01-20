"""Google OAuth2 authentication for Drive and Gmail APIs."""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from config.logging_config import get_logger

from .token_manager import TokenManager

logger = get_logger(__name__)

# Scopes required for Google Drive and Gmail read access
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]


class GoogleOAuthManager:
    """Manages Google OAuth2 authentication flow."""

    def __init__(self, credentials_path: Path, token_path: Path):
        """Initialize OAuth manager.

        Args:
            credentials_path: Path to OAuth client credentials JSON file
            token_path: Path to store/load the access token
        """
        self.credentials_path = credentials_path
        self.token_manager = TokenManager(token_path)
        self._credentials: Credentials | None = None

    def authenticate(self) -> Credentials:
        """Authenticate and return valid credentials.

        This will:
        1. Try to load existing token from storage
        2. Refresh the token if expired
        3. Run OAuth flow if no valid token exists

        Returns:
            Valid Google OAuth credentials

        Raises:
            FileNotFoundError: If credentials file doesn't exist
            ValueError: If authentication fails
        """
        # Try to load existing token
        self._credentials = self.token_manager.load_token()

        if self._credentials and self._credentials.valid:
            logger.info("using_existing_token")
            return self._credentials

        # Try to refresh expired token
        if self._credentials and self._credentials.expired and self._credentials.refresh_token:
            logger.info("refreshing_token")
            try:
                self._credentials.refresh(Request())
                self.token_manager.save_token(self._credentials)
                return self._credentials
            except Exception as e:
                logger.warning("token_refresh_failed", error=str(e))
                # Fall through to new auth flow

        # Run new OAuth flow
        logger.info("starting_oauth_flow")
        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"OAuth credentials file not found: {self.credentials_path}\n"
                "Download it from Google Cloud Console > APIs & Services > Credentials"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.credentials_path), SCOPES
        )
        self._credentials = flow.run_local_server(port=0)

        # Save for future use
        self.token_manager.save_token(self._credentials)
        logger.info("oauth_flow_completed")

        return self._credentials

    def get_credentials(self) -> Credentials:
        """Get current valid credentials.

        Returns:
            Valid credentials

        Raises:
            ValueError: If not authenticated
        """
        if self._credentials is None:
            self._credentials = self.token_manager.load_token()

        if self._credentials is None:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # Refresh if needed
        if self._credentials.expired and self._credentials.refresh_token:
            self._credentials.refresh(Request())
            self.token_manager.save_token(self._credentials)

        return self._credentials

    def is_authenticated(self) -> bool:
        """Check if we have valid credentials.

        Returns:
            True if authenticated with valid or refreshable credentials
        """
        if self._credentials is None:
            self._credentials = self.token_manager.load_token()
        return self.token_manager.is_token_valid(self._credentials)

    def logout(self) -> None:
        """Clear stored credentials."""
        self.token_manager.delete_token()
        self._credentials = None
        logger.info("logged_out")
