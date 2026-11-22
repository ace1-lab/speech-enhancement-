"""Download ONLY ~20 utterances from the 154 MB VoiceBank clean test-set zip using HTTP range
requests (the zip's central directory is read remotely; nothing else is transferred)."""
import io, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://huggingface.co/datasets/mrfakename/noisy-speech-files/resolve/main/clean_testset_wav.zip"
WANT = {"p232": 8, "p257": 14}          # the two VoiceBank test speakers -> speaker-disjoint dev/test


class RangeFile(io.RawIOBase):
    """Seekable file-like object backed by HTTP Range requests (or any fetch(start, end) callable)."""
    def __init__(self, size, fetch):
        self.size, self.fetch, self.pos = size, fetch, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if self.pos >= end:
            return b""
        data = self.fetch(self.pos, end - 1); self.pos = end
        return data


def http_fetch(url):
    def f(a, b):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    return f


def http_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        return int(cr.split("/")[1]) if cr else int(r.headers["Content-Length"])


def select_and_extract(zf, out_dir, want=WANT):
    out_dir.mkdir(parents=True, exist_ok=True); n = 0
    names = sorted(i for i in zf.namelist() if i.lower().endswith(".wav") and "__MACOSX" not in i)
    for spk, k in want.items():
        for name in [x for x in names if Path(x).name.startswith(spk + "_")][:k]:
            (out_dir / Path(name).name).write_bytes(zf.read(name)); n += 1
    return n


def main():
    out = ROOT / "data" / "raw_clean"
    print("Reading remote zip index…")
    zf = zipfile.ZipFile(RangeFile(http_size(URL), http_fetch(URL)))
    print(f"Extracted {select_and_extract(zf, out)} utterances to {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"Automatic download failed: {e}\nManual fallback: download clean_testset_wav.zip from "
                 "https://datashare.ed.ac.uk/handle/10283/2791, unzip, and run:\n"
                 "  python scripts/prepare_dataset.py --clean_dir <unzipped folder>")
