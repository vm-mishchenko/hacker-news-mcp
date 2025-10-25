from dataclasses import dataclass


@dataclass
class Resource:
    def get(self):
        return "Resource response"


@dataclass
class AppContext:
    """Application context with typed dependencies."""

    resource: Resource
