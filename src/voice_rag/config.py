from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", alias="VAANI_HOST")
    port: int = Field(default=8080, alias="VAANI_PORT")

    raw_dir: Path = Field(default=ROOT / "data" / "raw", alias="VAANI_RAW_DIR")
    index_dir: Path = Field(default=ROOT / "vaani_index", alias="VAANI_INDEX_DIR")
    reports_dir: Path = Field(default=ROOT / "data" / "reports", alias="VAANI_REPORTS_DIR")

    # Ingest
    parquet_name: str = "hinval.parquet"
    ingest_examples: int = 400
    holdout_queries: int = 120
    languages: tuple[str, ...] = ("en", "hi")

    lsa_components: int = 192
    tfidf_features: int = 12000

    # Retrieval
    dense_top_k: int = 8
    sparse_top_k: int = 10
    fused_top_k: int = 6
    rerank_pool: int = 8
    faiss_nprobe: int = 6
    faiss_threads: int = 1
    parent_expand: int = 6
    min_retrieval_score: float = 0.012

    # SLA
    sla_ms: float = 170.0
    bench_n: int = Field(default=120, alias="VAANI_BENCH_N")

    # Tiny local generator — always runs after extract
    slm_enabled: bool = Field(default=True, alias="VAANI_SLM")
    slm_model: str = Field(default="bigscience/mt0-small", alias="VAANI_SLM_MODEL")
    slm_max_new_tokens: int = 14
    slm_timeout_ms: float = 80.0
    slm_threads: int = Field(default=2, alias="VAANI_SLM_THREADS")
    slm_cache_dir: Path = Field(default=ROOT / "models", alias="VAANI_SLM_CACHE")

    # STT — one key, or rotate through several
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    sarvam_api_key_2: str = Field(default="", alias="SARVAM_API_KEY_2")
    sarvam_api_key_3: str = Field(default="", alias="SARVAM_API_KEY_3")
    sarvam_api_key_4: str = Field(default="", alias="SARVAM_API_KEY_4")
    sarvam_api_key_5: str = Field(default="", alias="SARVAM_API_KEY_5")
    sarvam_api_keys: str = Field(default="", alias="SARVAM_API_KEYS")
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "transcribe"

    # Guardrails
    grounding_min_coverage: float = 0.55
    max_query_chars: int = 800

    @property
    def parquet_path(self) -> Path:
        return self.raw_dir / self.parquet_name

    def sarvam_key_list(self) -> list[str]:
        """Deduped keys from SARVAM_API_KEY, _2.._5, and SARVAM_API_KEYS."""
        raw = [
            self.sarvam_api_key,
            self.sarvam_api_key_2,
            self.sarvam_api_key_3,
            self.sarvam_api_key_4,
            self.sarvam_api_key_5,
            self.sarvam_api_keys,
        ]
        out: list[str] = []
        seen: set[str] = set()
        for blob in raw:
            for part in str(blob or "").replace(";", ",").replace("\n", ",").split(","):
                key = part.strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(key)
        return out


settings = Settings()
