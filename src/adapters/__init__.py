"""Versioned adapters from frontend-private formats into stable IR."""

from src.adapters.mineru_v7_document_ir import (
    MinerUV7DocumentIRAdapter,
    MinerUV7DocumentIRAdapterConfig,
    convert_v7_payload_to_document_ir,
    load_v7_document_ir,
    write_v7_document_ir,
)
from src.adapters.mineru_v8_document_ir import (
    convert_v8_payload_to_document_ir,
    load_v8_document_ir,
    write_v8_document_ir,
)

__all__ = [
    "MinerUV7DocumentIRAdapter",
    "MinerUV7DocumentIRAdapterConfig",
    "convert_v7_payload_to_document_ir",
    "load_v7_document_ir",
    "write_v7_document_ir",
    "convert_v8_payload_to_document_ir",
    "load_v8_document_ir",
    "write_v8_document_ir",
]
