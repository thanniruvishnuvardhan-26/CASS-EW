"""
TSRD Downloader & Cache Manager
CASS-EW SIH Problem Statement 26055

This module manages downloading and caching TSRD dataset files from
the Hugging Face repository `alan-turing-institute/turing-synthetic-radar-dataset`.

CRITICAL SAFETY RULES:
1. The scheduler MUST NEVER call this module during execution.
2. Downloading is strictly separate from simulation/scheduling logic.
3. Hugging Face dataset is Gated: Requires valid HF_TOKEN for private/gated access.
4. If authentication fails, raises an explicit informative error without fabricating data.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, List

HF_REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_REPO_ID}"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

DEFAULT_CACHE_DIR = Path("data/tsrd_cache")


class TSRDDownloadError(Exception):
    """Raised when TSRD file download fails or access is unauthorized."""
    pass


class TSRDDownloader:
    def __init__(self, cache_dir: Optional[Path] = None, hf_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def get_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": "CASS-EW-SIH26055-ResearchClient/1.0"
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def check_access(self) -> Dict[str, Any]:
        """
        Queries HF API to check repository visibility and access status.
        """
        try:
            req = urllib.request.Request(HF_API_URL, headers=self.get_headers())
            with urllib.request.urlopen(req, timeout=10) as res:
                data = json.loads(res.read().decode('utf-8'))
                is_gated = data.get('gated', False)
                return {
                    "accessible": True,
                    "gated": is_gated,
                    "repo_id": HF_REPO_ID,
                    "total_files": len(data.get('siblings', [])),
                    "authenticated": bool(self.hf_token),
                    "message": "Repository metadata accessed successfully."
                }
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return {
                    "accessible": False,
                    "gated": True,
                    "repo_id": HF_REPO_ID,
                    "authenticated": bool(self.hf_token),
                    "error_code": 401,
                    "message": "HTTP 401 Unauthorized: TSRD is gated on Hugging Face. Set HF_TOKEN environment variable."
                }
            return {
                "accessible": False,
                "error_code": e.code,
                "message": f"HTTP Error {e.code}: {e.reason}"
            }
        except Exception as e:
            return {
                "accessible": False,
                "error_code": -1,
                "message": f"Network error checking TSRD access: {str(e)}"
            }

    def download_file(self, relative_path: str, force_download: bool = False) -> Path:
        """
        Downloads a specific file (e.g. 'stare/train_stare/config_0.h5') to local cache.
        """
        local_path = self.cache_dir / relative_path
        if local_path.exists() and not force_download:
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_RESOLVE_BASE}/{relative_path}"
        req = urllib.request.Request(url, headers=self.get_headers())

        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(local_path, 'wb') as out_file:
                # Stream chunk by chunk
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
            return local_path
        except urllib.error.HTTPError as e:
            if local_path.exists():
                local_path.unlink()  # Clean up partial download
            if e.code == 401:
                raise TSRDDownloadError(
                    f"HTTP 401 Unauthorized downloading '{relative_path}'. "
                    f"TSRD is gated. Provide a valid HF_TOKEN."
                ) from e
            raise TSRDDownloadError(f"HTTP Error {e.code} downloading '{relative_path}': {e.reason}") from e
        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            raise TSRDDownloadError(f"Failed to download '{relative_path}': {str(e)}") from e


if __name__ == "__main__":
    downloader = TSRDDownloader()
    status = downloader.check_access()
    print("=== TSRD ACCESS STATUS ===")
    print(json.dumps(status, indent=2))
