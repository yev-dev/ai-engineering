from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class ServiceEndpoint:
    name: str
    base_url: str
    timeout_sec: float = 2.0


class ExternalConnectivityService:
    """Manages service endpoints and provides lightweight health checks."""

    def __init__(self) -> None:
        self._endpoints: dict[str, ServiceEndpoint] = {}

    def register_endpoint(self, endpoint: ServiceEndpoint) -> None:
        self._endpoints[endpoint.name] = endpoint

    def list_endpoints(self) -> list[ServiceEndpoint]:
        return list(self._endpoints.values())

    def health_check(self, endpoint_name: str) -> dict[str, Any]:
        endpoint = self._endpoints[endpoint_name]
        req = Request(endpoint.base_url, method="GET")
        try:
            with urlopen(req, timeout=endpoint.timeout_sec) as resp:
                return {
                    "service": endpoint.name,
                    "url": endpoint.base_url,
                    "status": "up",
                    "http_status": int(resp.status),
                }
        except URLError as exc:
            return {
                "service": endpoint.name,
                "url": endpoint.base_url,
                "status": "down",
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "service": endpoint.name,
                "url": endpoint.base_url,
                "status": "down",
                "error": str(exc),
            }

    def health_check_all(self) -> list[dict[str, Any]]:
        return [self.health_check(endpoint.name) for endpoint in self._endpoints.values()]
