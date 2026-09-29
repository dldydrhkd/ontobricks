"""Preamble stripping for ``agents.agent_owl_generator.engine``.

Regression: the model answered "No documents available; …" followed by the
Turtle. The ``@prefix`` gate then skipped the class-cap, pitfall and
evaluator stages, and the unchecked text was accepted as the final OWL.
"""

from __future__ import annotations

import pytest

from agents.agent_owl_generator.engine import _strip_turtle_preamble

_TTL = "@prefix : <http://x/> .\n:A a owl:Class ."


@pytest.mark.unit
class TestStripTurtlePreamble:
    def test_leading_prose_is_removed(self):
        text = "No documents available; schemas are complete.\n\n" + _TTL
        assert _strip_turtle_preamble(text) == _TTL

    def test_code_fence_is_removed(self):
        text = "Here it is:\n```turtle\n" + _TTL + "\n```"
        assert _strip_turtle_preamble(text) == _TTL

    def test_clean_turtle_is_unchanged(self):
        assert _strip_turtle_preamble(_TTL) == _TTL

    def test_prose_without_turtle_is_kept(self):
        assert _strip_turtle_preamble("thinking out loud") == "thinking out loud"

    def test_prefix_inside_a_sentence_is_not_a_start(self):
        text = "I will emit @prefix lines next."
        assert _strip_turtle_preamble(text) == text
