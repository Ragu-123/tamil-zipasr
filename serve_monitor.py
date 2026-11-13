#!/usr/bin/env python3
"""
Tamil-ZipASR Live Dashboard & Cloudflare Quick Tunnel Server
============================================================
Launches a local-first visualization server (TensorBoard or MLflow) and
instantly exposes it via a secure, zero-config Cloudflare Quick Tunnel (trycloudflare.com).

No accounts, no API keys, no logins required!

Usage on Kaggle / Linux:
    python serve_monitor.py --type tensorboard --logdir /kaggle/working/tensorboard
    python serve_monitor.py --type mlflow --logdir /kaggle/working/mlruns

Output:
    =============================================================================
    >>> LIVE TRAINING DASHBOARD ACTIVE (Cloudflare Tunnel) <<<
    Public URL: https://example-subdomain.trycloudflare.com
    =============================================================================
"""

import os
import sys
import time
import re
import stat
import platform
import subprocess
import argparse
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloudflare_monitor")

CLOUDFLARED_LINUX_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_WINDOWS_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"


def ensure_cloudflared_binary() -> str:
    """Downloads or verifies cloudflared binary."""
    system = platform.system().lower()
    bin_name = "cloudflared.exe" if "windows" in system else "cloudflared"
    bin_path = os.path.join("/tmp" if "linux" in system else os.getcwd(), bin_name)

    if os.path.exists(bin_path) and os.access(bin_path, os.X_OK):
        return bin_path

    logger.info(f"Downloading Cloudflare Tunnel binary ({system})...")
    dl_url = CLOUDFLARED_WINDOWS_URL if "windows" in system else CLOUDFLARED_LINUX_URL

    req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(bin_path, "wb") as f:
        f.write(resp.read())

    # Make executable
    st = os.stat(bin_path)
    os.chmod(bin_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logger.info(f"Cloudflare binary installed at: {bin_path}")
    return bin_path


def start_visualization_server(server_type: str, logdir: str, port: int) -> subprocess.Popen:
    """Starts TensorBoard or MLflow UI process in background."""
    os.makedirs(logdir, exist_ok=True)
    if server_type == "tensorboard":
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", logdir,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--reload_interval", "5",
        ]
        logger.info(f"Starting TensorBoard on port {port} (logdir={logdir})...")
    elif server_type == "mlflow":
        cmd = [
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", logdir,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        logger.info(f"Starting MLflow on port {port} (logdir={logdir})...")
    else:
        raise ValueError(f"Unknown server type: {server_type}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def start_cloudflare_tunnel(cloudflared_bin: str, local_port: int) -> str:
    """Spawns cloudflared quick tunnel and extracts trycloudflare.com URL."""
    cmd = [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{local_port}"]
    logger.info(f"Creating Cloudflare Quick Tunnel to http://127.0.0.1:{local_port}...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url = None
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    timeout = time.time() + 30

    while time.time() < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        match = pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if not tunnel_url:
        logger.warning("Could not extract Cloudflare tunnel URL within timeout. Check network connection.")
        return "Tunnel initialization pending... check log output."

    return tunnel_url


def main():
    parser = argparse.ArgumentParser(description="Live Dashboard & Cloudflare Tunnel for ZipTransducer-Tamil")
    parser.add_argument("--type", choices=["tensorboard", "mlflow"], default="tensorboard", help="Visualization backend")
    parser.add_argument("--logdir", type=str, default="/kaggle/working/tensorboard", help="Directory where logs are stored")
    parser.add_argument("--port", type=int, default=6006, help="Local port for server")
    parser.add_argument("--daemon", action="store_true", help="Keep running indefinitely")
    args = parser.parse_args()

    # 1. Download/verify cloudflared binary
    cloudflared_bin = ensure_cloudflared_binary()

    # 2. Start TensorBoard or MLflow server
    server_proc = start_visualization_server(args.type, args.logdir, args.port)

    # 3. Create Cloudflare Tunnel
    public_url = start_cloudflare_tunnel(cloudflared_bin, args.port)

    banner = f"""
========================================================================================
           ZIPTRANSDUCER-TAMIL LIVE TRAINING DASHBOARD (CLOUDFLARE TUNNEL)
========================================================================================
  Backend:     {args.type.upper()} (Local-First, No Accounts/Keys Required)
  Log Dir:     {args.logdir}
  Local Port:  http://127.0.0.1:{args.port}

  >>> PUBLIC MONITORING LINK:
  >>> {public_url} <<<

  Open this URL on your browser/phone to monitor live loss, CER, WER, and throughput!
========================================================================================
"""
    print(banner, flush=True)

    if args.daemon:
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Stopping dashboard server...")
            server_proc.terminate()


if __name__ == "__main__":
    main()
