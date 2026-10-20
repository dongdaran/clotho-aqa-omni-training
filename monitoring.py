"""TensorBoard and Optuna Dashboard launchers, plus a read-only legacy-log bridge."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from experiment_utils import write_json


def scalar_record(row: dict) -> tuple[int, dict] | None:
    raw_step = row.get('step', row.get('global_step/max_steps'))
    if raw_step is None:
        return None
    try:
        step = int(str(raw_step).split('/')[0])
    except ValueError:
        return None
    values = {key: float(value) for key, value in row.items()
              if isinstance(value, (int, float)) and math.isfinite(value)}
    if not values:
        return None
    return step, values


def bridge_logs(root: Path, interval: float = 5):
    """Import ms-swift's existing logging.jsonl; never touch a training process."""
    from torch.utils.tensorboard import SummaryWriter
    offsets, writers, histories = {}, {}, {}
    destination = root / 'monitor_imports'
    destination.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            for source in root.glob('trial_*/batch_*/logging.jsonl'):
                relative = source.parent.relative_to(root)
                if source not in writers:
                    writers[source] = SummaryWriter(str(destination / relative))
                writer = writers[source]
                position = offsets.get(source, 0)
                if source.stat().st_size < position:
                    position = 0
                with source.open(encoding='utf-8') as stream:
                    stream.seek(position)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if not line.endswith('\n'):
                            stream.seek(start)
                            break
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        parsed = scalar_record(row)
                        if parsed:
                            step, values = parsed
                            for key, value in values.items():
                                writer.add_scalar('train/' + key, value, step)
                    offsets[source] = stream.tell()
                history = source.parent / 'history.json'
                if history.exists() and histories.get(history) != history.stat().st_mtime_ns:
                    try:
                        records = json.loads(history.read_text())
                        for row in records:
                            step = row.get('step', row.get('epoch', 0))
                            for key in ('accuracy', 'macro_f1'):
                                if row.get(key) is not None:
                                    writer.add_scalar('val/' + key, row[key], step)
                        histories[history] = history.stat().st_mtime_ns
                    except (json.JSONDecodeError, OSError):
                        pass
                writer.flush()
            time.sleep(interval)
    finally:
        for writer in writers.values():
            writer.close()


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred+20):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free monitoring port')


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def start_monitoring(root: Path, legacy: bool = False, show_colab: bool = True, with_optuna: bool = True) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Process identities live on this runtime, not on shared Drive.
    import hashlib
    registry = Path('/tmp') / ('aqa-monitors-' + hashlib.sha256(str(root).encode()).hexdigest()[:16] + '.json')
    old = json.loads(registry.read_text()) if registry.exists() else {}
    services = {}

    def launch(name, command, port=None):
        previous = old.get(name, {})
        if previous.get('pid') and _alive(previous['pid']):
            # Avoid trusting a reused PID after a runtime restart.
            cmdline = Path(f"/proc/{previous['pid']}/cmdline")
            if cmdline.exists() and str(root) in cmdline.read_bytes().decode(errors='replace'):
                services[name] = previous
                return
        with (root / f'{name}_server.log').open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        services[name] = {'pid': process.pid, 'port': port}

    port_tb = free_port(6006)
    launch('tensorboard', [sys.executable, '-m', 'tensorboard.main', '--logdir', str(root),
                          '--host', '127.0.0.1', '--port', str(port_tb), '--reload_interval', '5',
                          '--load_fast=false'], port_tb)
    if legacy:
        launch('legacy_bridge', [sys.executable, str(Path(__file__).resolve()), '--bridge-only', '--root', str(root)])
    elif with_optuna:
        from tune import open_storage
        open_storage(root / 'study.journal')
        port_optuna = free_port(8080)
        launch('optuna', [sys.executable, str(Path(__file__).resolve()), '--serve-optuna', '--root', str(root),
                          '--port', str(port_optuna)], port_optuna)
    write_json(registry, services)
    # Wait briefly for startup, so a dead service is reported instead of a false link.
    for name, service in services.items():
        port = service.get('port')
        if not port:
            continue
        ready = False
        until = time.monotonic()+20
        while time.monotonic() < until:
            if not _alive(service['pid']):
                break
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=0.3):
                    ready = True
                    break
            except OSError:
                time.sleep(0.25)
        service['ready'] = ready
        print(f"{name}: http://127.0.0.1:{port}/" if ready else
              f"{name} is starting or failed; inspect {root / (name+'_server.log')}", flush=True)
    if show_colab:
        try:
            from google.colab import output
            from IPython.display import Markdown, display
            for name, service in services.items():
                if not service.get('port') or not service.get('ready'):
                    continue
                display(Markdown(f'**{name}** — 브라우저 Colab에서 아래 화면을 사용하세요.'))
                # Colab recommends an iframe: standalone proxy links can fail
                # under browser storage partitioning, and VS Code lacks this JS API.
                output.serve_kernel_port_as_iframe(service['port'], height=700)
        except Exception as error:
            print(f'Colab proxy link unavailable ({error}); use VS Code port forwarding for the ports above.', flush=True)
    return services


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--bridge-only', action='store_true')
    parser.add_argument('--serve-optuna', action='store_true')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if args.bridge_only:
        bridge_logs(args.root)
    elif args.serve_optuna:
        from optuna_dashboard import run_server
        from tune import open_storage
        run_server(open_storage(args.root / 'study.journal'), host='127.0.0.1', port=args.port)
    else:
        start_monitoring(args.root, legacy=args.legacy)


if __name__ == '__main__':
    main()
