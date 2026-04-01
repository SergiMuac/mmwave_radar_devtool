"""Custom exceptions for the package."""


class RadarError(Exception):
    """Base error for the radar package."""


class DCA1000Error(RadarError):
    """Raised when DCA1000 communication fails."""


class DCA1000ResponseError(DCA1000Error):
    """Raised when DCA1000 reports a failure or malformed response."""


class RadarSerialError(RadarError):
    """Raised when radar UART communication fails."""


class ConfigurationError(RadarError):
    """Raised when user-provided configuration is invalid."""
