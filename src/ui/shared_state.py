"""
Process-level shared state for cross-session data sharing.

The control session (laptop) writes transcription data here.
TV viewer sessions read from here. Protected by a threading lock.
"""

import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any

import streamlit as st


@dataclass
class TranscriptionState:
    """Holds the latest transcription data shared across all sessions."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    version: int = 0
    is_streaming: bool = False
    session_uid: str = ""

    chunks: List[Dict[str, Any]] = field(default_factory=list)
    refined_blocks_ar: List[str] = field(default_factory=list)
    refined_blocks_en: List[str] = field(default_factory=list)
    refined_blocks_de: List[str] = field(default_factory=list)
    cloud_blocks: List[str] = field(default_factory=list)

    def push_chunk(self, chunk: Dict[str, Any]):
        with self.lock:
            self.chunks.append(chunk)
            self.version += 1

    def push_refined(self, ar: str, en: str, de: str):
        with self.lock:
            self.refined_blocks_ar.append(ar)
            self.refined_blocks_en.append(en)
            self.refined_blocks_de.append(de)
            self.version += 1

    def push_cloud(self, block: str):
        with self.lock:
            self.cloud_blocks.append(block)
            self.version += 1

    def set_streaming(self, value: bool, uid: str = ""):
        with self.lock:
            self.is_streaming = value
            if uid:
                self.session_uid = uid
            self.version += 1

    def get_snapshot(self, last_n: int = 5) -> dict:
        """Return a snapshot of the latest data for TV rendering."""
        with self.lock:
            return {
                "version": self.version,
                "is_streaming": self.is_streaming,
                "session_uid": self.session_uid,
                "raw_ar": [c["ar"] for c in self.chunks[-last_n:]],
                "refined_ar": self.refined_blocks_ar[-last_n:],
                "refined_en": self.refined_blocks_en[-last_n:],
                "refined_de": self.refined_blocks_de[-last_n:],
            }

    def reset(self):
        with self.lock:
            self.chunks.clear()
            self.refined_blocks_ar.clear()
            self.refined_blocks_en.clear()
            self.refined_blocks_de.clear()
            self.cloud_blocks.clear()
            self.is_streaming = False
            self.version += 1


@st.cache_resource
def get_shared_state() -> TranscriptionState:
    """Return the process-wide singleton TranscriptionState."""
    return TranscriptionState()
