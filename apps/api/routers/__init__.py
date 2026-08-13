"""API routers package.

Route handlers currently live in ``apps.api.main`` for a single FastAPI app
entry point. Shared auth/config lives in ``apps.api.deps``; streaming capability
checks use ``adapters.protocols.StreamingOpsRepository``.
"""
