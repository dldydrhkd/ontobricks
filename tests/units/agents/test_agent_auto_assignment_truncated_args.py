"""Truncated tool-call arguments in ``agents.agent_auto_assignment.engine``.

Regression: with ``max_tokens=2048`` a wide ``submit_entity_mapping`` call was
cut off mid-JSON; the arguments decoded to ``{}`` and the tool answered
"class_uri and sql_query are required", so the model retried the same
oversized payload again and again.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agents.agent_auto_assignment import engine


def _tool_call_response(raw_args: str) -> dict:
    return {
        "choices": [{
            "finish_reason": "length",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "submit_entity_mapping", "arguments": raw_args},
                }],
            },
        }],
        "usage": {},
    }


_DONE = {"choices": [{"finish_reason": "stop", "message": {"content": "Done."}}], "usage": {}}


@pytest.mark.unit
class TestTruncatedToolArguments:
    def _run(self, responses):
        sent = []

        def fake_call(*_a, **kwargs):
            sent.append([dict(m) for m in _a[3]])
            return responses.pop(0)

        with patch.object(engine, "call_serving_endpoint", side_effect=fake_call), \
             patch.object(engine, "dispatch_tool") as dispatch, \
             patch.object(engine.time, "sleep"):
            engine.run_agent(
                "https://h", "tok", "ep", None, {"tables": []},
                {"entities": [{"uri": "u", "name": "Contract"}], "relationships": []},
                max_iterations=2,
            )
        return sent, dispatch

    def test_truncated_arguments_are_not_dispatched(self):
        _sent, dispatch = self._run([_tool_call_response('{"class_uri": "u", "sql_q'), _DONE])
        dispatch.assert_not_called()

    def test_model_is_told_the_call_was_cut_off(self):
        sent, _ = self._run([_tool_call_response('{"class_uri": "u", "sql_q'), _DONE])
        tool_msg = next(m for m in sent[1] if m.get("role") == "tool")
        assert "cut off" in json.loads(tool_msg["content"])["error"]

    def test_budget_fits_wide_entity_payloads(self):
        assert engine._MAX_TOKENS >= 8192
