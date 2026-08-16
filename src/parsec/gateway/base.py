"""Model adapter seam.

Adapters implement one method. The gateway wraps whichever adapter is
active with event logging, blob capture, cost computation, and ledger
debits — those concerns never live in an adapter.

Adapter kinds:
  - anthropic: real generator (Anthropic Messages API)
  - fake:      scripted deterministic responses for tests
  - replay:    recorded responses from a prior session's event log
  - openai_judge (RESERVED, M5): heterogeneous judge per §6 stage 5 —
    a different model family than the generator. Slot only; not built at M1.
"""

from __future__ import annotations

from typing import Protocol

from parsec.models.gateway import ModelRequest, ModelResponse


class ModelAdapter(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
