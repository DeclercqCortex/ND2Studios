"""One-command CellSAM weights install — run this once per machine after `pip install`.

    python scripts/setup_cellsam.py                  # install / verify the weights
    python scripts/setup_cellsam.py --check          # report only, download nothing
    python scripts/setup_cellsam.py --model cellsam_extra
    python scripts/setup_cellsam.py --version 1.2 --force

WHY THIS EXISTS
    The CellSAM weights cannot ship with this repo: they are ~1.7 GB and licensed for
    **non-commercial academic use** through DeepCell's authenticated endpoint, so every
    user fetches their own copy with their own token. This script does exactly that and
    nothing else, so `analysis.segment` with ``method=cellsam`` behaves identically on a
    fresh clone as on the machine it was developed on.

    It deliberately calls cellSAM's OWN ``get_model()`` rather than re-implementing the
    presigned-URL download: the asset key, the md5 and the extraction layout then always
    come from the installed package and cannot drift from it.

WHAT IT ADDS OVER PLAIN `get_model()`
    A fallback for the single most common failure on managed machines: **TLS interception**
    by corporate proxies or antivirus HTTPS scanning. Those re-sign every connection with a
    private root that is trusted by the OS but absent from certifi, so Python's downloader
    dies with ``CERTIFICATE_VERIFY_FAILED`` while the browser and ``pip`` work fine. On
    Python 3.13 you cannot fix that by adding the root to a CA bundle either — verification
    is strict and rejects most AV-generated roots as RFC-non-compliant (e.g. ``Basic
    Constraints of CA cert not marked critical``).

    The fix is ``truststore``, which routes verification through the OS trust store — the
    same decision the browser makes. This script tries the normal path first and only
    injects ``truststore`` if it sees a TLS error, so nothing is silently loosened on a
    machine that does not need it. Note that the LIBRARY never does this: a global change
    to certificate verification belongs in an explicit, user-run step, not in a node.

AFTERWARDS
    Loading is fully OFFLINE — ``get_model()`` returns early once the version directory
    exists, so no token is consulted again, including from the GUI. Verify with:
        python scripts/_cellsam_smoke.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKEN_ENV = "DEEPCELL_ACCESS_TOKEN"
_TOKEN_URL = "https://users.deepcell.org"


def _token_instructions() -> str:
    """Platform-correct steps for setting the token, without the placeholder trap that
    once cost a debugging session (a token pasted WITH its surrounding <>)."""
    if os.name == "nt":
        setit = ('  $env:%s = "PASTE_TOKEN_HERE"      # this shell, right now\n'
                 '  setx %s "PASTE_TOKEN_HERE"        # persist it for future shells\n'
                 "  NOTE: `setx` writes the registry and does NOT affect the shell you\n"
                 "        type it in, so run BOTH (or reopen the terminal after setx).\n"
                 % (TOKEN_ENV, TOKEN_ENV))
    else:
        setit = ('  export %s="PASTE_TOKEN_HERE"      # this shell\n'
                 "  ...and add that line to ~/.bashrc or ~/.zshrc to persist it\n"
                 % TOKEN_ENV)
    return (f"  1. sign in at {_TOKEN_URL}/login/ and create an access token\n"
            "     (the models are licensed for NON-COMMERCIAL ACADEMIC use)\n"
            "  2. set it — paste the token BARE, with no <>, quotes or spaces:\n"
            + setit
            + "  3. re-run this script\n"
            "\nNO TOKEN AVAILABLE? The weights directory is portable: copy\n"
            "  ~/.deepcell/models/cellsam_v<ver>/  from a machine that already has it,\n"
            "or point the node's `model_path` socket at a single .pt. Either route needs\n"
            "no token and no network (mind the non-commercial academic licence).")


def _weight_files(version: str) -> list:
    d = Path.home() / ".deepcell" / "models" / f"cellsam_v{version}"
    return sorted(d.glob("*.pt")) if d.is_dir() else []


def _verify_checksums(version: str) -> tuple:
    """(ok, total) against the archive's own checksums.md5, if it shipped one."""
    import hashlib
    d = Path.home() / ".deepcell" / "models" / f"cellsam_v{version}"
    man = d / "checksums.md5"
    if not man.is_file():
        return (0, 0)
    ok = total = 0
    for line in man.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        want, name = parts[0].lower(), parts[1].lstrip("*./")
        f = d / name
        if not f.is_file():
            continue
        total += 1
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ok += int(h.hexdigest() == want)
    return (ok, total)


def _is_tls_error(exc: BaseException) -> bool:
    """True for the certificate-verification family, at any nesting depth."""
    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__}: {cur}"
        if any(k in text for k in ("SSLCertVerificationError", "CERTIFICATE_VERIFY_FAILED",
                                   "SSLError", "certificate verify failed",
                                   "unable to get local issuer")):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the CellSAM model weights.")
    ap.add_argument("--model", default="cellsam_general",
                    help="cellsam_general (the published generalist) | cellsam_extra "
                         "(more training data; for domains outside the paper)")
    ap.add_argument("--version", default="", help="model version (default: latest)")
    ap.add_argument("--check", action="store_true", help="report state, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    a = ap.parse_args(argv)

    def die(msg: str) -> int:
        print("\n[FAIL] " + msg)
        return 1

    # ── 1. the package ────────────────────────────────────────────────────────
    print("== 1. cellSAM package ==")
    try:
        import cellSAM
        from cellSAM import _auth
    except ImportError:
        return die("cellSAM is not installed:\n"
                   "    pip install git+https://github.com/vanvalenlab/cellSAM.git\n"
                   "  (it pulls torch, torchvision, segment_anything, kornia, dask-image, "
                   "scikit-learn)")
    version = a.version or max(_auth._model_versions, key=lambda v: tuple(
        int(p) for p in v.split(".")))
    rec = _auth._model_versions.get(version)
    if rec is None:
        return die(f"unknown model version {version!r}; the installed cellSAM offers "
                   f"{sorted(_auth._model_versions)}")
    print(f"   cellSAM {getattr(cellSAM, '__version__', '?')}   model {a.model!r} "
          f"v{version}")
    print(f"   asset  {rec['asset_key']}  (md5 {rec['asset_hash']}, from the package)")

    # ── 2. already installed? ─────────────────────────────────────────────────
    dest = Path.home() / ".deepcell" / "models" / f"cellsam_v{version}"
    print(f"== 2. cache: {dest} ==")
    have = _weight_files(version)
    if have and not a.force:
        for p in have:
            print(f"   {p.name}  {p.stat().st_size / (1 << 20):.0f} MB")
        ok, total = _verify_checksums(version)
        if total:
            print(f"   checksums.md5: {ok}/{total} verified"
                  + ("" if ok == total else "   <-- MISMATCH, re-run with --force"))
            if ok != total:
                return die("cached weights failed their own checksums; re-run with --force")
        want = dest / f"{a.model}.pt"
        if not want.is_file():
            return die(f"{want.name} is not in the cache (present: "
                       f"{[p.name for p in have]}). Re-run with --force, or choose one of "
                       f"those with --model.")
        print("\n[OK] weights already installed — loading is OFFLINE from here on "
              "(no token needed, including in the GUI).")
        print("     verify end to end:  python scripts/_cellsam_smoke.py")
        return 0
    if a.check:
        print("   not installed")
        print("\n[CHECK] weights absent. Run without --check to download "
              f"(~1.7 GB for v{version}).")
        return 1

    # ── 3. the token ──────────────────────────────────────────────────────────
    print("== 3. access token ==")
    tok = os.environ.get(TOKEN_ENV) or ""
    if not tok:
        return die(f"{TOKEN_ENV} is not set in THIS process.\n" + _token_instructions())
    bad = []
    if tok[0] in "<\"'" or tok[-1] in ">\"'":
        bad.append("wrapped in <> or quotes")
    if tok != tok.strip():
        bad.append("has leading/trailing whitespace")
    if bad:
        return die(f"{TOKEN_ENV} looks malformed ({'; '.join(bad)}) — paste it bare.\n"
                   f"    (length {len(tok)}; the value is never printed)\n"
                   + _token_instructions())
    print(f"   set ({len(tok)} chars; never printed)")

    # ── 4. download, with a TLS-interception fallback ──────────────────────────
    print(f"== 4. download (~1.7 GB) -> {dest.parent} ==")

    def fetch() -> None:
        from cellSAM import get_model
        get_model(a.model, version=version)

    t0 = time.time()
    try:
        fetch()
    except Exception as exc:                          # noqa: BLE001 — reported below
        if not _is_tls_error(exc):
            return die(f"{type(exc).__name__}: {exc}")
        print(f"   TLS verification failed: {str(exc)[:130]}")
        print("   -> this machine intercepts HTTPS (corporate proxy or antivirus). "
              "Retrying through the OS trust store.")
        try:
            import truststore
        except ImportError:
            return die(
                "TLS interception detected and `truststore` is not installed. It routes "
                "certificate verification through the OS trust store — the same decision "
                "your browser makes — and is the portable fix on every platform:\n"
                "    pip install truststore\n"
                "  then re-run this script. (Adding the intercepting root to a CA bundle "
                "does NOT work on Python 3.13, which verifies strictly and rejects most "
                "antivirus-generated roots as RFC-non-compliant.)")
        truststore.inject_into_ssl()
        print("   truststore injected; retrying")
        try:
            fetch()
        except Exception as exc2:                     # noqa: BLE001
            return die(f"still failing after truststore: {type(exc2).__name__}: {exc2}\n"
                       "  If your proxy needs a client certificate or the token is "
                       "expired, fix that first. As a last resort, download the archive "
                       "with any tool that works here and unpack it so that\n"
                       f"    {dest / (a.model + '.pt')}\n"
                       "  exists — loading is offline from then on.")
    dt = time.time() - t0

    # ── 5. verify what landed ─────────────────────────────────────────────────
    print("== 5. verify ==")
    got = _weight_files(version)
    if not got:
        return die(f"download reported success but no .pt is present in {dest}")
    for p in got:
        print(f"   {p.name}  {p.stat().st_size / (1 << 20):.0f} MB")
    ok, total = _verify_checksums(version)
    if total:
        print(f"   checksums.md5: {ok}/{total} verified")
        if ok != total:
            return die("the extracted weights failed their own checksums — re-run "
                       "with --force")
    if not (dest / f"{a.model}.pt").is_file():
        return die(f"{a.model}.pt did not appear in {dest}")
    print(f"\n[OK] CellSAM weights installed in {dt:.0f}s. Loading is OFFLINE from here on "
          "(no token needed, including in the GUI).")
    print("     verify end to end:  python scripts/_cellsam_smoke.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
