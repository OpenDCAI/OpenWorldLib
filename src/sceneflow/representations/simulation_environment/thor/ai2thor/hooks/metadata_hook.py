from typing import Dict, Any, TYPE_CHECKING

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

if TYPE_CHECKING:
    from ..controller import Controller


class MetadataHook(Protocol):
    def __call__(self, metadata: Dict[str, Any], controller: "Controller"):
        ...
