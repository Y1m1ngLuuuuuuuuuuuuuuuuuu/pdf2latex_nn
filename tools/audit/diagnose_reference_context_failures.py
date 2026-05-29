#!/usr/bin/env python3
"""Compatibility entrypoint for ReferenceContext Phase1 diagnostics.

The implementation lives in validate_reference_context_phase1_mineru_evidence.py
so selected200 validation and ad-hoc diagnosis use the same audit-only logic.
"""

from __future__ import annotations

from tools.audit.validate_reference_context_phase1_mineru_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
