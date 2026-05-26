import os
import io
import glob
import json
import time
import html
import webbrowser
import urllib.parse
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import re
from typing import Optional, Union, List, Dict, Tuple, Any

# Optional dependencies commonly available in this project
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # graceful fallback

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    # Optional, for PR/ROC curves
    from sklearn.metrics import precision_recall_curve as _sk_prc  # type: ignore
    from sklearn.metrics import roc_curve as _sk_roc  # type: ignore
except Exception:
    _sk_prc = None
    _sk_roc = None


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _git_short_commit() -> Optional[str]:
    try:
        import subprocess
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=_project_root(),
            )
            .decode()
            .strip()
        )
        return commit or None
    except Exception:
        return None


def _fmt_bytes(num: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < step:
            return f"{num:.1f} {unit}" if unit != "B" else f"{num} {unit}"
        num /= step
    return f"{num:.1f} PB"


def _file_url(path: str) -> str:
    # Build a file:// URL for local opening; ensure proper URL-encoding
    abspath = os.path.abspath(path)
    return "file://" + urllib.parse.quote(abspath)


# -----------------------------
# Resultados: agregación
# -----------------------------

def _resultados_dir() -> str:
    root = _project_root()
    try:
        from src.config import RESULTADOS_DIR as _R  # type: ignore
    except Exception:
        _R = "Resultados"
    return os.path.join(root, _R)


def _detect_family_and_variant(path: str) -> tuple[str, str]:
    """Infer model family and variant from filename/path.
    Returns (family, variant_label).
    """
    name = os.path.basename(path).lower()
    # Family
    if "gat_report" in name or "gnn_anomaly" in name or name.startswith("gat_model_"):
        family = "GAT"
    elif name.startswith("results_xgb") or "xgb_" in name:
        family = "XGBoost"
    elif name.startswith("results_mlp") or "mlp_" in name:
        family = "MLP"
    elif name.startswith("transformer_results") or "transformer_ts" in name:
        family = "Transformer"
    elif name.startswith("results_anomaly") or "anomaly_" in name:
        family = "Anomalias"
    else:
        family = "Desconocido"
    # Variant from filename hints
    variant = "Base"
    nm = os.path.basename(path)
    if "GraphSMOTE".lower() in nm.lower():
        variant = "GraphSMOTE"
    if "ImGAGN".lower() in nm.lower():
        variant = (variant + "+ImGAGN") if variant != "Base" else "ImGAGN"
    return family, variant


def _parse_results_csv(path: str) -> list[dict]:
    """Return normalized rows of metrics from a results CSV path.
    Normalized keys: family, variant, label, split, time, file, metrics{...}.
    """
    out: list[dict] = []
    if not pd:
        return out
    try:
        df = pd.read_csv(path)
    except Exception:
        return out
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    family, variant = _detect_family_and_variant(path)

    # Transformer summary has columns like val_* and test_*
    cols = set(c.lower() for c in df.columns)
    if {"val_f1_acc_1", "test_f1_acc_1"}.issubset(cols) or any(c.startswith("val_") for c in cols):
        # Build two rows (val/test) if present
        row = df.iloc[0].to_dict()
        for split in ("val", "test"):
            key = f"{split}_f1_acc_1"
            if key in df.columns:
                out.append(
                    {
                        "family": "Transformer" if family == "Transformer" else family,
                        "variant": variant,
                        "label": "Transformer",
                        "split": split,
                        "time": mtime,
                        "file": path,
                        "metrics": {
                            "f1_1": float(row.get(f"{split}_f1_acc_1", float("nan"))),
                            "precision_1": float(row.get(f"{split}_precision_1", float("nan"))),
                            "recall_1": float(row.get(f"{split}_recall_1", float("nan"))),
                            "accuracy": float(row.get(f"{split}_accuracy", float("nan"))),
                        },
                    }
                )
        return out

    # Generic unified results: one row per split
    for _, r in df.iterrows():
        split = str(r.get("split", "")).lower() or "val"
        # Family-specific label
        label = family
        if family == "Anomalias":
            algo = str(r.get("model", "")).strip()
            if algo:
                label = f"Anomalias ({algo})"
        elif family in ("MLP", "XGBoost"):
            label = family
        elif family == "GAT":
            # Compose label with variant
            fam, var = _detect_family_and_variant(path)
            label = f"{fam} {var}" if var and var != "Base" else fam
        metrics = {
            "f1_1": r.get("f1_1", r.get("f1", float("nan"))),
            "precision_1": r.get("precision_1", float("nan")),
            "recall_1": r.get("recall_1", float("nan")),
            "accuracy": r.get("accuracy", float("nan")),
            # auc vs auroc naming
            "auc": r.get("auc", r.get("auroc", float("nan"))),
            "auprc": r.get("auprc", float("nan")),
            "threshold": r.get("threshold", float("nan")),
        }
        # Cast to floats safely
        mclean = {}
        for k, v in metrics.items():
            try:
                mclean[k] = float(v)
            except Exception:
                mclean[k] = float("nan")
        out.append(
            {
                "family": family,
                "variant": variant,
                "label": label,
                "split": split,
                "time": mtime,
                "file": path,
                "metrics": mclean,
            }
        )
    return out


def _collect_all_results() -> list[dict]:
    base = _resultados_dir()
    patterns = [
        "results_gat_report_*.csv",
        "results_gnn_anomaly_*.csv",
        "results_xgb_*.csv",
        "results_mlp_*.csv",
        "results_anomaly_*.csv",
        "transformer_results_*.csv",
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(base, pat)))
    files = sorted(files, key=lambda p: os.path.getmtime(p))
    rows: list[dict] = []
    for p in files:
        rows.extend(_parse_results_csv(p))
    return rows


def _results_summary_payload() -> dict:
    """Return latest-per-family summary, preferring test over val split."""
    rows = _collect_all_results()
    # group by label (family+variant label)
    latest: dict[str, dict] = {}
    for r in rows:
        key = r.get("label") or (r.get("family") or "?")
        # prefer test over val; among same split, pick most recent by file mtime order
        split = r.get("split")
        prev = latest.get(key)
        if prev is None:
            latest[key] = r
            continue
        if prev.get("split") != "test" and split == "test":
            latest[key] = r
            continue
        # If both same split, keep the newer (we iterated ascending by mtime, so replace)
        if prev.get("split") == split:
            latest[key] = r
    # Build payload
    items = []
    import math as _math
    def _clean_metrics(m: dict) -> dict:
        out = {}
        for k, v in (m or {}).items():
            try:
                fv = float(v)
                if _math.isfinite(fv):
                    out[k] = fv
                else:
                    out[k] = None
            except Exception:
                out[k] = None
        return out
    for label, r in latest.items():
        items.append({
            "label": label,
            "family": r.get("family"),
            "variant": r.get("variant"),
            "split": r.get("split"),
            "time": r.get("time"),
            "file": os.path.relpath(r.get("file"), _project_root()) if r.get("file") else None,
            "metrics": _clean_metrics(r.get("metrics", {})),
        })
    # Sort by label for deterministic order
    items = sorted(items, key=lambda d: d.get("label", ""))
    return {"items": items, "metric_keys": ["f1_1", "precision_1", "recall_1", "accuracy", "auc", "auprc"]}


def _results_timeseries_payload(metric: str = "f1_1") -> dict:
    rows = _collect_all_results()
    import math as _math
    series: dict[str, list[dict]] = {}
    for r in rows:
        label = r.get("label") or r.get("family")
        m = r.get("metrics", {})
        val = m.get(metric)
        try:
            val = float(val)
            if not _math.isfinite(val):
                continue
        except Exception:
            continue
        series.setdefault(label, []).append({"time": r.get("time"), "value": float(val), "split": r.get("split")})
    # Ensure chronological order
    for k in series:
        # time is formatted string; sort by file mtime actually better, but we used ascending earlier
        # so keep as-is
        pass
    return {"metric": metric, "series": series}


# -----------------------------
# Predicciones: PR/ROC y matriz de confusión
# -----------------------------

def _label_from_preds_path(path: str) -> str:
    name = os.path.basename(path)
    low = name.lower()
    if low.startswith("preds_gat_report_") or low.startswith("preds_gnn_anomaly_"):
        fam, var = _detect_family_and_variant(path)
        return f"{fam} {var}" if var and var != "Base" else fam
    if low.startswith("preds_mlp_"):
        return "MLP"
    if low.startswith("preds_xgb_"):
        return "XGBoost"
    if low.startswith("preds_anomaly_"):
        return "Anomalias"
    return "Otros"


def _list_preds_files() -> list[dict]:
    base = _resultados_dir()
    patterns = [
        "preds_gat_report_*.csv",
        "preds_gnn_anomaly_*.csv",
        "preds_mlp_*.csv",
        "preds_xgb_*.csv",
        "preds_anomaly_*.csv",
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(base, pat)))
    out: list[dict] = []
    for p in files:
        try:
            mtime = os.path.getmtime(p)
            label = _label_from_preds_path(p)
            # split suele estar en el contenido, pero también en el nombre (para algunos)
            split = None
            bn = os.path.basename(p).lower()
            for s in ("train", "val", "test"):
                if f"_{s}_" in bn or bn.endswith(f"_{s}.csv"):
                    split = s
                    break
            out.append({
                "path": p,
                "label": label,
                "split": split,  # puede ser None, se resolverá al leer
                "mtime": mtime,
            })
        except Exception:
            continue
    # sort newest first
    out.sort(key=lambda d: d["mtime"])  # ascending
    return out


def _load_preds_df(path: str):
    if not pd:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=';')
        except Exception:
            return None
    # normalize required columns
    cols = set(c.lower() for c in df.columns)
    # y_true
    y_col = None
    for c in df.columns:
        if c.lower() == 'y_true':
            y_col = c
            break
    if y_col is None:
        return None
    # score: prob1 or anomaly_score
    score_col = None
    for c in df.columns:
        if c.lower() in ('prob1', 'anomaly_score'):
            score_col = c
            break
    # y_pred
    ypred_col = None
    for c in df.columns:
        if c.lower() == 'y_pred':
            ypred_col = c
            break
    # split
    split = None
    for c in df.columns:
        if c.lower() == 'split':
            try:
                split = str(df[c].iloc[0]).lower()
            except Exception:
                split = None
            break
    return {
        'df': df,
        'y_col': y_col,
        'score_col': score_col,
        'ypred_col': ypred_col,
        'split': split,
    }


def _latest_preds_by_label_split() -> dict:
    items = _list_preds_files()
    best: dict[tuple[str, str], dict] = {}
    for it in items:
        label = it['label']
        split = it['split']
        if split is None:
            meta = _load_preds_df(it['path'])
            if meta:
                split = meta.get('split') or split
        if not split:
            split = 'test'
        key = (label, split)
        prev = best.get(key)
        if not prev or it['mtime'] >= prev['mtime']:
            best[key] = it
    return best


def _preds_labels_payload() -> dict:
    best = _latest_preds_by_label_split()
    labels: dict[str, set] = {}
    for (label, split), it in best.items():
        labels.setdefault(label, set()).add(split)
    items = [
        {"label": label, "splits": sorted(list(splits))}
        for label, splits in sorted(labels.items())
    ]
    return {"items": items}


def _pr_roc_from_df(df, y_col: str, score_col: Optional[str]):
    import numpy as _np
    y = df[y_col].astype(int).to_numpy()
    if score_col is None or score_col not in df.columns:
        return {"pr": {"precision": [], "recall": []}, "roc": {"fpr": [], "tpr": []}, "n": int(y.size)}
    s = df[score_col].astype(float).to_numpy()
    # PR
    if _sk_prc is not None:
        prec, rec, _ = _sk_prc(y, s)
    else:
        # naive fallback: sort by score desc and accumulate
        order = _np.argsort(-s)
        y_sorted = y[order]
        tp = _np.cumsum(y_sorted == 1)
        fp = _np.cumsum(y_sorted == 0)
        total_pos = max(1, int((y == 1).sum()))
        prec = tp / _np.maximum(1, (tp + fp))
        rec = tp / total_pos
        # prepend (0,1) style? Keep as computed
    # ROC
    if _sk_roc is not None:
        fpr, tpr, _ = _sk_roc(y, s)
    else:
        # fallback similar
        thresholds = _np.unique(s)[::-1]
        tpr_list = []
        fpr_list = []
        P = max(1, int((y == 1).sum()))
        N = max(1, int((y == 0).sum()))
        for t in thresholds:
            yp = (s >= t).astype(int)
            tp = int(((yp == 1) & (y == 1)).sum())
            fp = int(((yp == 1) & (y == 0)).sum())
            tpr_list.append(tp / P)
            fpr_list.append(fp / N)
        fpr = _np.array(fpr_list)
        tpr = _np.array(tpr_list)
    return {
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n": int(y.size),
    }


def _confusion_from_df(df, y_col: str, ypred_col: Optional[str]):
    import numpy as _np
    y = df[y_col].astype(int).to_numpy()
    if ypred_col is not None and ypred_col in df.columns:
        yp = df[ypred_col].astype(int).to_numpy()
    else:
        # fallback: use threshold 0.5 on score if present
        score_col = None
        for c in df.columns:
            if c.lower() in ('prob1', 'anomaly_score'):
                score_col = c
                break
        if score_col is None:
            return {"cm": [[0, 0], [0, 0]], "metrics": {}}
        yp = (df[score_col].astype(float).to_numpy() >= 0.5).astype(int)
    tn = int(((yp == 0) & (y == 0)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    tp = int(((yp == 1) & (y == 1)).sum())
    total = max(1, tn + fp + fn + tp)
    acc = (tn + tp) / total
    prec1 = tp / max(1, tp + fp)
    rec1 = tp / max(1, tp + fn)
    f1 = (2 * prec1 * rec1) / max(1e-12, (prec1 + rec1))
    return {
        "cm": [[tn, fp], [fn, tp]],
        "metrics": {
            "accuracy": float(acc),
            "precision_1": float(prec1),
            "recall_1": float(rec1),
            "f1_1": float(f1),
            "support_pos": int((y == 1).sum()),
            "support_neg": int((y == 0).sum()),
        },
    }


def _list_files(base: str, patterns: Optional[list[str]] = None, recursive: bool = True, limit: int = 2000) -> list[dict]:
    files: list[str] = []
    if not os.path.isdir(base):
        return []
    if patterns:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(base, pat), recursive=recursive))
    else:
        if recursive:
            for root, _, fnames in os.walk(base):
                for fn in fnames:
                    files.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(base):
                p = os.path.join(base, fn)
                if os.path.isfile(p):
                    files.append(p)
    files = sorted(set(files), key=lambda p: os.path.getmtime(p))
    files = files[-limit:]
    out = []
    for p in files:
        try:
            rel = os.path.relpath(p, _project_root())
            st = os.stat(p)
            if st.st_size == 0:
                continue

            file_date = None
            match = re.search(r"training_log_(\d{8})_\d{6}\.log", os.path.basename(p))
            if match:
                date_str = match.group(1)
                try:
                    file_date = datetime.strptime(date_str, "%d%m%Y").strftime("%Y-%m-%d")
                except ValueError:
                    pass

            out.append(
                {
                    "path": p,
                    "rel": rel,
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "ext": os.path.splitext(p)[1].lower(),
                    "file_date": file_date,
                }
            )
        except Exception:
            pass
    return out



def _read_text_tail(path: str, max_bytes: int = 32_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
        # Try decode as utf-8 with fallback
        try:
            txt = data.decode("utf-8", errors="replace")
        except Exception:
            txt = data.decode("latin-1", errors="replace")
        return txt
    except Exception as e:
        return f"<no se pudo leer: {html.escape(str(e))}>"

def _read_text_all(path: str, max_bytes: Optional[int] = 2_000_000) -> str:
    """Lee el archivo completo (hasta max_bytes) con fallback de codificación."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes if max_bytes else None)
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin-1", errors="replace")
    except Exception as e:
        return f"<no se pudo leer: {html.escape(str(e))}>"


def _preview_csv(path: str, nrows: int = 30) -> str:
    if not pd:
        # Fallback to simple text tail
        return f"<pre>{html.escape(_read_text_tail(path, 8192))}</pre>"
    try:
        df = pd.read_csv(path, nrows=nrows)
        # Convert to HTML with basic styling
        return df.to_html(border=0, classes=["tbl", "tbl-compact"], index=False)
    except Exception:
        # Retry with ; separator (common in ES locales)
        try:
            df = pd.read_csv(path, nrows=nrows, sep=";")
            return df.to_html(border=0, classes=["tbl", "tbl-compact"], index=False)
        except Exception:
            return f"<pre>{html.escape(_read_text_tail(path, 8192))}</pre>"


def _dump_tensor_info(path: str) -> str:
    if not torch:
        return "<em>PyTorch no disponible para inspección.</em>"
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        summary = {
            "type": type(obj).__name__,
        }
        # If it's a dict-like state or payload, summarize keys
        if isinstance(obj, dict):
            summary["keys"] = list(obj.keys())[:20]
        elif hasattr(obj, "state_dict"):
            sd = obj.state_dict()
            summary["state_keys"] = list(sd.keys())[:20]
        return f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>"
    except Exception as e:
        return f"<em>No se pudo leer tensor: {html.escape(str(e))}</em>"


def _collect_config() -> dict:
    # Import config dynamically to capture current values
    try:
        from src import config as cfg  # type: ignore
    except Exception:
        return {}

    def to_jsonish(obj):
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None
        # primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        # numpy scalars / arrays
        if _np is not None:
            try:
                if isinstance(obj, _np.generic):
                    return obj.item()
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
            except Exception:
                pass
        # torch dtype/device
        try:
            import torch  # type: ignore
            if isinstance(obj, (torch.device,)):
                return str(obj)
        except Exception:
            pass
        # dict: coerce keys to str and values recursively
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    sk = k if isinstance(k, str) else str(k)
                except Exception:
                    sk = repr(k)
                out[sk] = to_jsonish(v)
            return out
        # list/tuple/set
        if isinstance(obj, (list, tuple, set)):
            return [to_jsonish(x) for x in obj]
        # fallback string
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    cfg_vars = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        if k.isupper():
            try:
                v = getattr(cfg, k)
                cfg_vars[k] = to_jsonish(v)
            except Exception:
                pass
    return cfg_vars


def _gen_section_header(title: str) -> str:
    return f"<h2>{html.escape(title)}</h2>"


def _gen_file_list_section(title: str, base: str, patterns: Optional[list[str]] = None, preview_limit: int = 8) -> str:
    items = _list_files(base, patterns=patterns, recursive=True)
    if not items:
        return _gen_section_header(title) + f"<p><em>Sin archivos en {html.escape(base)}</em></p>"
    buf = io.StringIO()
    buf.write(_gen_section_header(title))
    buf.write('<table class="tbl">')
    buf.write("<thead><tr><th>Archivo</th><th>Tamaño</th><th>Modificado</th><th>Preview</th></tr></thead><tbody>")
    count = 0
    for it in reversed(items):  # most recent first
        rel_txt = html.escape(it["rel"])
        size = _fmt_bytes(it["size"]) if isinstance(it["size"], int) else str(it["size"])
        mtime = html.escape(it["mtime"])
        path = it["path"]
        ext = it["ext"]
        # Links: open file and folder
        file_link = _file_url(path)
        folder_link = _file_url(os.path.dirname(path))
        link_html = f"<a href='{file_link}'>Abrir</a> · <a href='{folder_link}'>Carpeta</a>"
        preview_html = ""
        if count < preview_limit:
            if ext in (".csv", ".tsv"):
                preview_html = _preview_csv(path)
            elif ext in (".json", ".txt", ".log", ".md", ".py"):
                preview_html = f"<pre>{html.escape(_read_text_tail(path, 12000))}</pre>"
            elif ext in (".pt", ".pth"):
                preview_html = _dump_tensor_info(path)
            else:
                preview_html = "<em>—</em>"
        else:
            preview_html = "<em>(sin vista previa)</em>"
        buf.write(
            "<tr>"
            f"<td><code>{rel_txt}</code><br><small>{link_html}</small></td>"
            f"<td>{size}</td>"
            f"<td>{mtime}</td>"
            f"<td class=preview>{preview_html}</td>"
            "</tr>"
        )
        count += 1
    buf.write("</tbody></table>")
    return buf.getvalue()


def _gen_logs_section() -> str:
    root = _project_root()
    # Prefer explicit .log files, also surface TensorBoard runs directory
    logs = _list_files(root, patterns=["**/*.log"], recursive=True)
    runs_dir = os.path.join(root, "Resultados", "gnn", "training", "tensorboard")
    runs_note = ""
    if os.path.isdir(runs_dir):
        runs_note = f"<p>TensorBoard: <code>{html.escape(os.path.relpath(runs_dir, root))}</code> (sugerido: <code>tensorboard --logdir={html.escape(runs_dir)}</code>)</p>"
    if not logs and not runs_note:
        return _gen_section_header("Logs") + "<p><em>No se encontraron archivos de log.</em></p>"

    buf = io.StringIO()
    buf.write(_gen_section_header("Logs"))
    if runs_note:
        buf.write(runs_note)
    if logs:
        buf.write('<table class="tbl">')
        buf.write("<thead><tr><th>Archivo</th><th>Tamaño</th><th>Modificado</th><th>Tail</th></tr></thead><tbody>")
        for it in reversed(logs[-20:]):  # limit to 20 most recent logs
            buf.write(
                "<tr>"
                f"<td><code>{html.escape(it['rel'])}</code></td>"
                f"<td>{_fmt_bytes(it['size'])}</td>"
                f"<td>{html.escape(it['mtime'])}</td>"
                f"<td class=preview><pre>{html.escape(_read_text_tail(it['path'], 12000))}</pre></td>"
                "</tr>"
            )
        buf.write("</tbody></table>")
    return buf.getvalue()


def _gen_config_section() -> str:
    cfg = _collect_config()
    buf = io.StringIO()
    buf.write(_gen_section_header("Config (src/config.py)"))
    if not cfg:
        buf.write("<p><em>No se pudo cargar configuración.</em></p>")
        return buf.getvalue()
    buf.write('<table class="tbl">')
    buf.write("<thead><tr><th>Clave</th><th>Valor</th></tr></thead><tbody>")
    for k in sorted(cfg.keys()):
        v = cfg[k]
        try:
            txt = json.dumps(v, ensure_ascii=False)
        except Exception:
            txt = str(v)
        buf.write(f"<tr><td><b>{html.escape(k)}</b></td><td><code>{html.escape(txt)}</code></td></tr>")
    buf.write("</tbody></table>")
    return buf.getvalue()


def _gen_project_overview() -> str:
    root = _project_root()
    commit = _git_short_commit() or "N/A"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = []
    for d in ("Datos", "Resultados", "src", "tests"):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            nfiles = sum(len(files) for _, _, files in os.walk(p))
            stats.append((d, nfiles))
    items = "".join(
        f"<li><b>{html.escape(name)}</b>: {count} archivos</li>" for name, count in stats
    )
    return (
        f"<p><b>Proyecto:</b> <code>{html.escape(root)}</code><br>"
        f"<b>Commit:</b> <code>{html.escape(str(commit))}</code><br>"
        f"<b>Generado:</b> {html.escape(now)}</p>"
        f"<ul>{items}</ul>"
    )


def _build_html() -> str:
    root = _project_root()
    try:
        from src.config import RESULTADOS_DIR  # type: ignore
    except Exception:
        RESULTADOS_DIR = os.path.join(root, "Resultados")

    resultados_dir = os.path.join(root, RESULTADOS_DIR)
    datos_dir = os.path.join(root, "Datos")
    src_dir = os.path.join(root, "src")
    tests_dir = os.path.join(root, "tests")

    parts = []
    parts.append("<h1>Reporte Visual</h1>")
    parts.append(_gen_project_overview())
    parts.append(_gen_config_section())
    parts.append(_gen_logs_section())

    style = """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 20px; color: #222; }
    h1 { margin-bottom: 0.25rem; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #eee; padding-bottom: 0.25rem; }
    code { background: #fafafa; padding: 2px 4px; border-radius: 4px; }
    table.tbl { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 14px; }
    table.tbl th, table.tbl td { border: 1px solid #eee; padding: 6px 8px; vertical-align: top; }
    table.tbl thead { background: #f7f7f7; position: sticky; top: 0; }
    .tbl-compact { font-size: 12px; }
    td.preview { max-width: 720px; }
    pre { white-space: pre; word-wrap: break-word; background: #fbfbfb; padding: 8px; border-radius: 4px; border: 1px solid #f0f0f0; }
    .notice { padding: 8px 12px; background: #fffbe6; border: 1px solid #ffe58f; border-radius: 4px; }
    .footer { margin-top: 2rem; color: #666; font-size: 12px; }
    </style>
    """

    html_doc = (
        "<!DOCTYPE html><html lang=es><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Reporte Visual</title>{style}</head><body>"
        + "".join(parts)
        + "<div class='footer'>src/visual.py</div>"
        + "</body></html>"
    )
    return html_doc


def run_visual_report(open_in_browser: bool = True) -> str:
    """
    Genera un reporte HTML con vista de logs, resultados, datasets y variables
    de configuración. Devuelve la ruta del archivo HTML generado.
    """
    root = _project_root()
    try:
        from src.config import RESULTADOS_DIR  # type: ignore
    except Exception:
        RESULTADOS_DIR = "Resultados"
    out_dir = os.path.join(root, RESULTADOS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reporte_visual.html")
    html_doc = _build_html()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    if open_in_browser:
        try:
            webbrowser.open(f"file://{out_path}")
        except Exception:
            pass
    print(f"✅ Reporte visual generado → {out_path}")
    return out_path


if __name__ == "__main__":
    run_visual_report(open_in_browser=True)

# ---------------------------------------------------------------------------
# Servidor dinámico (HTTP) para visualización en tiempo real
# ---------------------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, payload: Union[dict, list], status: int = 200):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _error(handler: BaseHTTPRequestHandler, status: int, message: str):
    _json_response(handler, {"error": message, "status": status}, status=status)


def _safe_path_from_rel(rel: str) -> Optional[str]:
    root = _project_root()
    full = os.path.abspath(os.path.join(root, rel))
    if os.path.commonpath([root, full]) != root:
        return None
    return full


def _overview_payload() -> dict:
    root = _project_root()
    commit = _git_short_commit() or "N/A"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = {}
    for d in ("Datos", "Resultados", "src", "tests"):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            nfiles = sum(len(files) for _, _, files in os.walk(p))
            stats[d] = nfiles
    return {"root": root, "commit": commit, "generated": now, "stats": stats}


def _list_section_payload(section: str, limit: int = 2000) -> list[dict]:
    root = _project_root()
    # Respetar RESULTADOS_DIR desde config si está definido
    try:
        from src.config import RESULTADOS_DIR as _R  # type: ignore
    except Exception:
        _R = "Resultados"
    base_map = {
        "resultados": os.path.join(root, _R),
        "datos": os.path.join(root, "Datos"),
        "src": os.path.join(root, "src"),
        "tests": os.path.join(root, "tests"),
    }
    base = base_map.get(section.lower())
    if not base:
        return []
    return _list_files(base, patterns=None, recursive=True, limit=limit)


def _logs_list_payload(limit: int = 40) -> list[dict]:
    root = _project_root()
    items = _list_files(root, patterns=["**/*.log"], recursive=True, limit=4000)
    items = sorted(items, key=lambda it: it["mtime"])  # already str; sort by str okay (YYYY-mm-dd HH:MM:SS)
    return list(reversed(items))[:limit]


def _tail_payload(rel: str, max_bytes: int = 12000) -> dict:
    safe = _safe_path_from_rel(rel)
    if not safe or not os.path.isfile(safe):
        return {"rel": rel, "content": "", "error": "Archivo no encontrado"}
    return {"rel": rel, "content": _read_text_tail(safe, max_bytes=max_bytes)}


def _read_text_payload(rel: str, max_bytes: Optional[int] = None) -> dict:
    safe = _safe_path_from_rel(rel)
    if not safe or not os.path.isfile(safe):
        return {"rel": rel, "content": "", "error": "Archivo no encontrado"}
    return {"rel": rel, "content": _read_text_all(safe, max_bytes=max_bytes)}


def _preview_payload(rel: str, nrows: int = 30, max_bytes: int = 12000) -> dict:
    safe = _safe_path_from_rel(rel)
    if not safe or not os.path.isfile(safe):
        return {"rel": rel, "type": "text", "content": "", "error": "Archivo no encontrado"}
    ext = os.path.splitext(safe)[1].lower()
    if ext in (".csv", ".tsv"):
        if pd is not None:
            try:
                df = pd.read_csv(safe, nrows=nrows)
                return {"rel": rel, "type": "html", "content": df.to_html(border=0, index=False)}
            except Exception:
                try:
                    df = pd.read_csv(safe, nrows=nrows, sep=";")
                    return {"rel": rel, "type": "html", "content": df.to_html(border=0, index=False)}
                except Exception:
                    pass
        # fallback
        return {"rel": rel, "type": "text", "content": _read_text_tail(safe, max_bytes)}
    elif ext in (".json", ".txt", ".log", ".md", ".py", ".tex"):
        return {"rel": rel, "type": "text", "content": _read_text_tail(safe, max_bytes)}
    elif ext in (".pt", ".pth"):
        return {"rel": rel, "type": "html", "content": _dump_tensor_info(safe)}
    else:
        return {"rel": rel, "type": "text", "content": "(sin vista previa)"}


INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"es">
<head>
  <meta charset=\"utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reporte Visual</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #222; }
    h1 { margin-bottom: 0.25rem; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #eee; padding-bottom: 0.25rem; }
    code { background: #fafafa; padding: 2px 4px; border-radius: 4px; }
    table.tbl { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 14px; }
    table.tbl th, table.tbl td { border: 1px solid #eee; padding: 6px 8px; vertical-align: top; }
    table.tbl thead { background: #f7f7f7; position: sticky; top: 0; }
    .tbl-compact { font-size: 12px; }
    td.preview { max-width: 720px; }
    /* Asegurar barras de desplazamiento y evitar desbordes en logs */
    pre { white-space: pre; overflow: auto; background: #fbfbfb; padding: 8px; border-radius: 4px; border: 1px solid #f0f0f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
    #log-tail { max-height: 50vh; max-width: 100%; overflow: auto; }
    .row { display: flex; gap: 24px; flex-wrap: wrap; }
    .col { flex: 1 1 480px; }
    .toolbar { display: flex; align-items: center; gap: 8px; }
    .footer { margin-top: 2rem; color: #666; font-size: 12px; }
    .muted { color: #777; }
    button { padding: 6px 10px; }
    input[type=checkbox] { transform: scale(1.2); }
    .chart-box { background:#fff; border:1px solid #eee; border-radius:8px; padding:10px; }
    .log-date { color: #888; }
    .log-level { font-weight: bold; }
    .log-level-INFO { color: #3498db; }
    .log-level-WARNING { color: #f39c12; }
    .log-level-ERROR { color: #e74c3c; }
    .log-level-DEBUG { color: #9b59b6; }
    .log-level-CRITICAL { color: #c0392b; font-weight: bold; }
    .log-message { color: #333; }
    /* Botón Volver siempre visible y rojo */
    #btn-shutdown {
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 1000;
      background: #d32f2f;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 8px 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
      cursor: pointer;
    }
    #btn-shutdown:hover { background: #b71c1c; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <button id="btn-shutdown">Volver a la Terminal</button>
  <h1>Reporte Visual</h1>
  <div id="overview" class="muted">Cargando…</div>
  <div class="toolbar" style="margin:10px 0 20px 0">
    <button id="btn-tex">Ver documentación (LaTeX)</button>
  </div>

  <div class="row">
    <div class="col">
      <h2>Logs</h2>
      <div class="toolbar">
        <button id="btn-refresh-logs">Actualizar</button>
        <label><input type="checkbox" id="auto-logs" checked> Auto</label>
        <label>Fecha: <select id="sel-log-date"><option value="">Todas</option></select></label>
        <label>Archivo: <select id="sel-log"></select></label>
        <span id="log-meta" class="muted"></span>
      </div>
      <h3>Contenido (tail)</h3>
      <pre id="log-tail" style="max-height: 360px; overflow:auto">Seleccione un log…</pre>
    </div>
  </div>

  <h2>Curvas (PR / ROC) y Matriz de Confusión</h2>
  <div class="toolbar">
    <label>Modelo: <select id="sel-label"></select></label>
    <label>Split: <select id="sel-split"><option>test</option></select></label>
    <button id="btn-refresh-curves">Actualizar curvas</button>
  </div>
  <div class="row">
    <div class="col chart-box">
      <h3>Precision-Recall</h3>
      <canvas id="chart-pr-curve" height="160"></canvas>
    </div>
    <div class="col chart-box">
      <h3>ROC</h3>
      <canvas id="chart-roc-curve" height="160"></canvas>
    </div>
  </div>

  <div class="row">
    <div class="col">
      <h3>Matriz de Confusión</h3>
      <div id="cm-metrics" class="muted" style="margin-bottom:6px"></div>
      <table class="tbl" id="tbl-cm"><tbody></tbody></table>
    </div>
    <div class="col chart-box">
      <h3>Serie temporal de métricas</h3>
      <div class="toolbar">
        <label>Métrica: 
          <select id="sel-metric">
            <option value="f1_1" selected>F1 (1)</option>
            <option value="precision_1">Precisión (1)</option>
            <option value="recall_1">Recall (1)</option>
            <option value="accuracy">Accuracy</option>
            <option value="auc">AUC</option>
            <option value="auprc">AUPRC</option>
          </select>
        </label>
        <button id="btn-refresh-series">Actualizar serie</button>
      </div>
      <canvas id="chart-series" height="160"></canvas>
    </div>
  </div>

  <h2>Resultados (Gráficos)</h2>
  <div class="toolbar">
    <button id="btn-refresh-results">Actualizar métricas</button>
  </div>

  
  <div class="row">
    <div class="col chart-box">
      <h3>F1 (Accidente=1) — último por modelo</h3>
      <canvas id="chart-f1" height="160"></canvas>
    </div>
    <div class="col chart-box">
      <h3>Precisión y Recall — último por modelo</h3>
      <canvas id="chart-pr" height="160"></canvas>
    </div>
  </div>
  <div class="row">
    <div class="col chart-box">
      <h3>AUC y AUPRC — último por modelo</h3>
      <canvas id="chart-auc" height="160"></canvas>
    </div>
    <div class="col">
      <h3>Resumen numérico (último)</h3>
      <table class="tbl" id="tbl-summary"><thead>
        <tr><th>Modelo</th><th>Split</th><th>F1</th><th>Precisión</th><th>Recall</th><th>Acc</th><th>AUC</th><th>AUPRC</th><th>Archivo</th><th>Fecha</th></tr>
      </thead><tbody></tbody></table>
    </div>
  </div>

  <!-- Flags de configuración (movido al final, tras resultados) -->
  <h2>Flags de Config</h2>
  <div class="toolbar"><button id="btn-refresh-cfg">Actualizar</button></div>
  <table class="tbl" id="tbl-cfg"><thead>
    <tr><th>Variable</th><th>Valor</th><th>Descripción</th><th>Uso</th></tr>
  </thead><tbody></tbody></table>

  <div class='footer'>Hecho con ❤️ — src/visual.py (servidor dinámico)</div>

  <script>
  const fmtBytes = (n) => {
    const units = ['B','KB','MB','GB','TB'];
    let x = Number(n), i = 0; if (!isFinite(x)) return String(n);
    while (x >= 1024 && i < units.length-1) { x/=1024; i++; }
    return (i===0? x.toFixed(0): x.toFixed(1)) + ' ' + units[i];
  };

  const qs = (sel) => document.querySelector(sel);
  const qsa = (sel) => Array.from(document.querySelectorAll(sel));

  async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP '+r.status);
    return await r.json();
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  async function loadOverview() {
    try {
      const o = await fetchJSON('/api/overview');
      const el = qs('#overview');
      if (el) el.innerHTML = `<b>Proyecto:</b> <code>${o.root}</code> · <b>Commit:</b> <code>${o.commit}</code> · <b>Generado:</b> ${o.generated}`
        + (o.stats ? `<br><span class='muted'>Archivos — Resultados: ${o.stats.Resultados||0}, Datos: ${o.stats.Datos||0}, src: ${o.stats.src||0}, tests: ${o.stats.tests||0}</span>` : '');
    } catch (e) { const el = qs('#overview'); if (el) el.textContent = 'No disponible'; try { console.error('overview error', e); }catch(_) {} }
  }

  function renderFileTable(tbody, items, clickCb) {
    tbody.innerHTML = '';
    items.forEach(it => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><code>${it.rel}</code></td><td>${fmtBytes(it.size)}</td><td>${it.mtime}</td>`;
      if (clickCb) tr.addEventListener('click', () => clickCb(it));
      tbody.appendChild(tr);
    });
  }

  function updateLogMeta() {
    const meta = qs('#log-meta');
    const sel = qs('#sel-log');
    const items = window.__logItems || [];
    if (!meta || !sel) return;
    const it = items.find(x => x.rel === sel.value);
    meta.textContent = it ? `${it.mtime} · ${fmtBytes(it.size)}` : '';
  }

  async function loadTailForSelected() {
    const sel = qs('#sel-log');
    if (!sel || !sel.value) return;
    const logEl = qs('#log-tail');
    if (!logEl) return;
    try {
      const tail = await fetchJSON('/api/tail?rel=' + encodeURIComponent(sel.value));
      const logContent = tail.content || '';
      
      let htmlContent = escapeHtml(logContent);

      const regex = /^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) - (INFO|WARNING|ERROR|DEBUG|CRITICAL) - (.*)$/gm;
      
      htmlContent = htmlContent.replace(regex, (match, date, level, message) => {
        return `<span class="log-date">${date}</span> - <span class="log-level log-level-${level}">${level}</span> - <span class="log-message">${message}</span>`;
      });

      logEl.innerHTML = htmlContent;
    } catch (e) {
      if (logEl) logEl.textContent = 'Error al cargar el log: ' + (e.message || 'desconocido');
      console.error('Error loading log tail:', e);
    }
  }

  async function loadLogs() {
    try {
      const items = await fetchJSON('/api/logs');
      window.__logItems = items;
      
      const selDate = qs('#sel-log-date');
      const selFile = qs('#sel-log');

      if (!selDate || !selFile) return;

      // Populate date filter
      const dates = [...new Set(items.map(it => it.file_date).filter(d => d))];
      const prevDate = selDate.value;
      selDate.innerHTML = '<option value="">Todas</option>';
      dates.sort().reverse().forEach(d => {
          const opt = document.createElement('option');
          opt.value = d;
          opt.textContent = d;
          selDate.appendChild(opt);
      });
      if (prevDate && dates.includes(prevDate)) {
          selDate.value = prevDate;
      }

      // Filter items based on selected date
      const selectedDate = selDate.value;
      const filteredItems = selectedDate ? items.filter(it => it.file_date === selectedDate) : items;

      // Populate file selector
      const prevFile = selFile.value;
      selFile.innerHTML = '';
      filteredItems.forEach(it => {
        const opt = document.createElement('option');
        opt.value = it.rel;
      const bn = it.rel.split(/[\u005C/]/).pop();
        opt.textContent = bn;
        selFile.appendChild(opt);
      });

      if (filteredItems.length) {
        if (prevFile && filteredItems.some(it => it.rel === prevFile)) {
            selFile.value = prevFile;
        } else if (filteredItems.length > 0) {
            selFile.value = filteredItems[0].rel;
        }
        updateLogMeta();
        await loadTailForSelected();
      } else {
        qs('#log-tail').textContent = '';
        updateLogMeta();
      }
    } catch (e) {}
  }

  // Documentación de flags (descripción y uso en el pipeline)
  const CONFIG_DOCS = {
    RESULTADOS_DIR: {
      desc: 'Carpeta raíz para exportar artefactos y métricas.',
      uso: 'Ubicación de CSV/JSON/plots, checkpoints y reportes.'
    },
    SEED: {
      desc: 'Semilla global para reproducibilidad.',
      uso: 'Inicializa torch y numpy; afecta splits y entrenamiento.'
    },
    DT_MINUTES: {
      desc: 'Granularidad temporal de agregación (minutos).',
      uso: 'Controla features, series de flujo y loaders.'
    },
    ACC_NEIGHBORHOODS: {
      desc: 'Vecindarios a marcar alrededor de accidentes.',
      uso: 'Expande etiqueta positiva a hops temporales/espaciales (graph.py).'
    },
    SPEED_BINS: {
      desc: 'Bines fijos de velocidad para entropía.',
      uso: 'Define discretización en features de velocidad (features.py).'
    },
    HARMONIC_V_MIN: {
      desc: 'Velocidad mínima para media armónica.',
      uso: 'Evita divisiones por cero en métricas de velocidad.'
    },
    DT_REF: {
      desc: 'DT base para escalar parámetros de flujo.',
      uso: 'Referencia cuando AUTOSCALE_FLOW_PARAMS=1.'
    },
    XAI: {
      desc: 'Activa la recolección/guardado de pesos de atención para XAI.',
      uso: 'En GAT (gat_model.py) guarda atenciones; train_pretrain puede regularizar/analizar estas alphas.'
    },
    EXPORT_LEGACY_GAT_CSV: {
      desc: 'Exporta CSV de resultados con prefijo antiguo para compatibilidad.',
      uso: 'Duplica archivos tipo gnn_anomaly_* además de gat_report_* en Resultados.'
    },
    DEBUG: {
      desc: 'Mensajes de depuración detallados en procesamiento y entrenamiento.',
      uso: 'Imprime trazas en utils, graph, gat_model, train_pretrain, etc.'
    },
    AUTOSCALE_FLOW_PARAMS: {
      desc: 'Reescala parámetros de “Flujo” según DT_MINUTES.',
      uso: 'Ajusta FLOW_ROLL_W, FLOW_EWMA_ALPHAS, FLOW_EXCEED_THRESHOLDS y SMOOTH_V_W.'
    },
    FLOW_SUBBINS_PER_MIN: {
      desc: 'Sub-bins por minuto para agregación subminuto.',
      uso: 'Controla subdivisión de ventanas temporales (features.py).'
    },
    FLOW_SUBBINS_M: {
      desc: 'Sub-bins efectivos por ventana DT.',
      uso: 'Derivado de FLOW_SUBBINS_PER_MIN×DT_MINUTES.'
    },
    FLOW_TRIM_P: {
      desc: 'Proporción para media recortada en flujo.',
      uso: 'Robustez a outliers en agregaciones (features.py).'
    },
    FLOW_WINSOR_P: {
      desc: 'Proporción para winsorización en flujo.',
      uso: 'Limita extremos antes de promediar (features.py).'
    },
    FLOW_ROLL_W: {
      desc: 'Tamaño de ventana rolling (nº de ventanas).',
      uso: 'Historia temporal para métricas de flujo.'
    },
    FLOW_EWMA_ALPHAS: {
      desc: 'Alphas para EWMAs de flujo.',
      uso: 'Suavizado exponencial con distintos horizontes.'
    },
    FLOW_EXCEED_THRESHOLDS: {
      desc: 'Umbrales de excedencia de flujo (conteos).',
      uso: 'Cómputo de indicadores binarios o tasas (features.py).'
    },
    SMOOTH_V_W: {
      desc: 'Ventana de suavizado para velocidad.',
      uso: 'Media móvil de velocidad por pórtico.'
    },
    FLOW_RUN_HIGH_Q: {
      desc: 'Cuantil para runs altos de flujo.',
      uso: 'Corte para duración de rachas altas.'
    },
    FLOW_RUN_LOW_Q: {
      desc: 'Cuantil para runs bajos de flujo.',
      uso: 'Corte para duración de rachas bajas.'
    },
    TIME_SLOTS_TO_FILTER: {
      desc: 'Tramos horarios a filtrar del dataset.',
      uso: 'Permite centrar el análisis en ventanas diarias concretas.'
    },
    MAX_EPOCHS: {
      desc: 'Máximo de épocas de entrenamiento.',
      uso: 'Límite superior de ciclos de entrenamiento.'
    },
    EARLY_STOPPING_PATIENCE: {
      desc: 'Paciencia de early stopping (evaluaciones).',
      uso: 'Detiene si no mejora tras N evaluaciones.'
    },
    TIMEOUT: {
      desc: 'Tiempo máximo de búsqueda/entrenamiento (segundos).',
      uso: 'Controla duración de optimizaciones y jobs largos.'
    },
    N_TRIALS: {
      desc: 'Número de trials en Optuna.',
      uso: 'Búsqueda de hiperparámetros (main.py).'
    },
    NUM_EPOCHS_OPTUNA: {
      desc: 'Épocas por trial en Optuna.',
      uso: 'Costo por evaluación de hiperparámetros.'
    },
    BATCH_SIZE: {
      desc: 'Tamaño de batch para loaders de grafos.',
      uso: 'Vecindario y entrenamientos (GraphSMOTE, etc.).'
    },
    NUM_NEIGHBORS: {
      desc: 'Vecinos por capa y tipo de arista.',
      uso: 'Sampling en NeighborLoader (hetero).'
    },
    TRAIN_RATIO: {
      desc: 'Porcentaje de datos para TRAIN.',
      uso: 'Partición temporal/secuencial de datasets.'
    },
    VAL_RATIO: {
      desc: 'Porcentaje de datos para VAL.',
      uso: 'Partición temporal/secuencial de datasets.'
    },
    F_BETA_THRESHOLD: {
      desc: 'Beta para selección de umbral por Fβ.',
      uso: 'Escoge tau en validación (MLP/XGB/Anomalía).'
    },
    GRAPHSMOTE_MODE: {
      desc: 'Modo de GraphSMOTE: offline/online/none.',
      uso: 'Estrategia de síntesis de nodos positivos.'
    },
    TARGET_POS_RATIO: {
      desc: 'Proporción objetivo de positivos en TRAIN.',
      uso: 'Equilibrio de clases con SMOTE.'
    },
    GRAPHSMOTE_K: {
      desc: 'k para k-NN en síntesis SMOTE.',
      uso: 'Selección de vecinos sintetizadores.'
    },
    PRETRAIN_EDGE_EPOCHS: {
      desc: 'Épocas para pre-entrenar generador de aristas.',
      uso: 'Mejora calidad de edges para sintéticos.'
    },
    SMOTE_EVERY_N_EPOCHS: {
      desc: 'Frecuencia de refresco en modo online.',
      uso: 'Re-sintetiza nodos cada N épocas.'
    },
    GS_SEED: {
      desc: 'Semilla específica para GraphSMOTE.',
      uso: 'Reproducibilidad de la síntesis.'
    },
    SAVE_AUG_GRAPH_PATH: {
      desc: 'Ruta para guardar el grafo aumentado.',
      uso: 'Persistencia de augmentations en Resultados.'
    },
    EMB_NUM_NEIGHBORS: {
      desc: 'Vecinos por capa para loaders de embeddings.',
      uso: 'Carga de z para decoders y síntesis.'
    },
    EMB_BATCH_SIZE: {
      desc: 'Batch size para loaders de embeddings.',
      uso: 'Eficiencia en extracción de z.'
    },
    DECODER_EPOCHS: {
      desc: 'Épocas para entrenar decodificadores z→x.',
      uso: 'Reconstrucción de features desde embeddings.'
    },
    GRAPHSMOTE_CONECT: {
      desc: 'Estrategia de conexión para nodos sintéticos en GraphSMOTE.',
      uso: '1: Top-K por edge_gen hacia reales; 0: vecinos 1-hop de reales con clase=1.'
    },
    BIDIRECCTION: {
      desc: 'Direccionalidad de aristas entre nodos reales y sintéticos.',
      uso: '1: bidireccional real↔sintético; 0: solo real→sintético (GraphSMOTE/ImGAGN).'
    },
    SAVE_GAT_ALIASES: {
      desc: 'Guarda alias adicionales de modelos GAT (puede duplicar .pt).',
      uso: 'Crea archivos alias por variante y global además del archivo único.'
    }
  };

  async function loadCfg() {
    try {
      const cfg = await fetchJSON('/api/config');
      const tbody = qs('#tbl-cfg tbody');
      tbody.innerHTML = '';
      // Mostrar TODOS los flags disponibles en config; si no hay docs, usar fallback
      Object.keys(cfg).sort().forEach(k => {
        const tr = document.createElement('tr');
        let val = cfg[k];
        if (typeof val === 'object') val = JSON.stringify(val);
        const doc = CONFIG_DOCS[k] || { desc: '—', uso: '—' };
        tr.innerHTML = `<td><b>${k}</b></td>`+
                       `<td><code>${val}</code></td>`+
                       `<td>${doc.desc}</td>`+
                       `<td>${doc.uso}</td>`;
        tbody.appendChild(tr);
      });
    } catch (e) {}
  }

  function _mkBarChart(id, labels, datasets) {
    const ctx = qs('#'+id);
    if (!ctx) return;
    if (typeof Chart === 'undefined') {
      // Chart.js no disponible: dejar un aviso y salir
      const p = ctx.parentElement; if (p && !p.querySelector('.muted')) { const note = document.createElement('div'); note.className='muted'; note.textContent='Chart.js no disponible (sin conexión)'; p.appendChild(note); }
      return;
    }
    if (window.__charts && window.__charts[id]) { try { window.__charts[id].destroy(); } catch (e) {} }
    window.__charts = window.__charts || {};
    window.__charts[id] = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { position:'top' } },
        scales: { y: { beginAtZero: true, max: 1.0 } }
      }
    });
  }

  function renderSummary(summary) {
    const items = (summary && summary.items) ? summary.items : [];
    const labels = items.map(x => x.label);
    const f1 = items.map(x => (x.metrics && isFinite(x.metrics.f1_1)) ? x.metrics.f1_1 : null);
    try { _mkBarChart('chart-f1', labels, [{ label:'F1', data:f1, backgroundColor:'#4682B4' }]); } catch(e) {}
    const prec = items.map(x => (x.metrics && isFinite(x.metrics.precision_1)) ? x.metrics.precision_1 : null);
    const rec  = items.map(x => (x.metrics && isFinite(x.metrics.recall_1)) ? x.metrics.recall_1 : null);
    try { _mkBarChart('chart-pr', labels, [
      { label:'Precisión', data:prec, backgroundColor:'#2E8B57' },
      { label:'Recall', data:rec, backgroundColor:'#DAA520' },
    ]); } catch(e) {}
    const auc = items.map(x => (x.metrics && isFinite(x.metrics.auc)) ? x.metrics.auc : null);
    const auprc = items.map(x => (x.metrics && isFinite(x.metrics.auprc)) ? x.metrics.auprc : null);
    try { _mkBarChart('chart-auc', labels, [
      { label:'AUC', data:auc, backgroundColor:'#8A2BE2' },
      { label:'AUPRC', data:auprc, backgroundColor:'#FF7F50' },
    ]); } catch(e) {}

    const tb = qs('#tbl-summary tbody');
    if (tb) {
      tb.innerHTML = '';
      items.forEach(x => {
        const tr = document.createElement('tr');
        const m = x.metrics || {};
        const fmt = v => (v==null || Number.isNaN(v)) ? '' : Number(v).toFixed(4);
        tr.innerHTML = `<td>${x.label}</td><td>${x.split || ''}</td><td>${fmt(m.f1_1)}</td><td>${fmt(m.precision_1)}</td><td>${fmt(m.recall_1)}</td><td>${fmt(m.accuracy)}</td><td>${fmt(m.auc)}</td><td>${fmt(m.auprc)}</td><td><code>${x.file||''}</code></td><td>${x.time||''}</td>`;
        tb.appendChild(tr);
      });
    }
  }

  async function loadResults() {
    try {
      const summary = await fetchJSON('/api/results/summary');
      renderSummary(summary);
    } catch (e) {}
  }

  // ---- Curvas (PR/ROC) y Matriz ----
  async function loadPredLabels() {
    try {
      const data = await fetchJSON('/api/preds/labels');
      const sel = qs('#sel-label');
      const selSplit = qs('#sel-split');
      if (!sel || !selSplit) return;
      sel.innerHTML = '';
      data.items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.label; opt.textContent = item.label;
        sel.appendChild(opt);
      });
      if (data.items.length) {
        sel.value = data.items[0].label;
        updateSplits(data.items[0].splits);
        // Render initial curves
        refreshCurves();
      }
      sel.addEventListener('change', () => {
        const found = data.items.find(x => x.label === sel.value);
        updateSplits(found ? found.splits : ['test']);
      });
    } catch (e) {}
  }

  function updateSplits(splits) {
    const selSplit = qs('#sel-split');
    selSplit.innerHTML = '';
    (splits || ['test']).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      selSplit.appendChild(opt);
    });
  }

  function _mkLineChart(id, x, ys, labels, colors) {
    const ctx = qs('#'+id);
    if (!ctx) return;
    if (typeof Chart === 'undefined') {
      const p = ctx.parentElement; if (p && !p.querySelector('.muted')) { const note = document.createElement('div'); note.className='muted'; note.textContent='Chart.js no disponible (sin conexión)'; p.appendChild(note); }
      return;
    }
    if (window.__charts && window.__charts[id]) { try { window.__charts[id].destroy(); } catch(e) {} }
    window.__charts = window.__charts || {};
    const datasets = ys.map((arr, i) => ({ label: labels[i], data: arr, borderColor: colors[i%colors.length], fill:false, tension:0.2 }));
    window.__charts[id] = new Chart(ctx, { type:'line', data: { labels:x, datasets }, options:{ responsive:true, plugins:{ legend:{position:'top'} }, scales:{ y:{ beginAtZero:true, max:1.0 } } } });
  }

  async function refreshCurves() {
    const elLabel = qs('#sel-label');
    const label = elLabel ? elLabel.value : null;
    const elSplit = qs('#sel-split');
    const split = (elSplit && elSplit.value) ? elSplit.value : 'test';
    if (!label) return;
    try {
      const curves = await fetchJSON('/api/curves?label=' + encodeURIComponent(label) + '&split=' + encodeURIComponent(split));
      const pr = curves.pr || {precision:[], recall:[]};
      const roc = curves.roc || {fpr:[], tpr:[]};
      _mkLineChart('chart-pr-curve', pr.recall, [pr.precision], ['Precisión'], ['#2E8B57']);
      _mkLineChart('chart-roc-curve', roc.fpr, [roc.tpr], ['TPR'], ['#4682B4']);
      // CM
      const cm = await fetchJSON('/api/confusion?label=' + encodeURIComponent(label) + '&split=' + encodeURIComponent(split));
      const m = cm.metrics || {}; const C = cm.cm || [[0,0],[0,0]];
      const tb = qs('#tbl-cm tbody');
      if (tb) {
        tb.innerHTML = `<tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>`+
                       `<tr><th>Real 0</th><td>${C[0][0]}</td><td>${C[0][1]}</td></tr>`+
                       `<tr><th>Real 1</th><td>${C[1][0]}</td><td>${C[1][1]}</td></tr>`;
      }
      const meta = qs('#cm-metrics');
      if (meta) {
        const fmt = v => (v==null || Number.isNaN(v))? '' : Number(v).toFixed(4);
        meta.textContent = `Acc=${fmt(m.accuracy)} · P1=${fmt(m.precision_1)} · R1=${fmt(m.recall_1)} · F1=${fmt(m.f1_1)} · +=${m.support_pos||0} · -=${m.support_neg||0}`;
      }
    } catch (e) {}
  }

  async function refreshSeries() {
    const elMetric = qs('#sel-metric');
    const metric = (elMetric && elMetric.value) ? elMetric.value : 'f1_1';
    try {
      const data = await fetchJSON('/api/results/timeseries?metric=' + encodeURIComponent(metric));
      const x = []; const labels = []; const ys = [];
      const colors = ['#4682B4','#2E8B57','#DAA520','#8A2BE2','#FF7F50','#DC143C','#708090'];
      const series = data.series || {};
      // Build aligned x as timestamps union; simplify: use index order per label
      Object.keys(series).forEach((lab) => {
        const arr = series[lab];
        labels.push(lab);
        ys.push(arr.map(p => p.value));
        if (x.length < arr.length) x.length = arr.length;
      });
      // x as ordinal indices with last timestamps from first series
      const xLabels = (series[labels[0]]||[]).map(p => p.time);
      _mkLineChart('chart-series', xLabels, ys, labels, colors);
    } catch (e) {}
  }

  async function loadSection(sec, tableId) {
    try {
      const items = await fetchJSON('/api/files?section=' + encodeURIComponent(sec));
      renderFileTable(qs('#'+tableId+' tbody'), items);
    } catch (e) {}
  }

  document.addEventListener('DOMContentLoaded', () => {
    try { loadOverview(); } catch(e) { try{ console.error('loadOverview()', e); }catch(_){} }
    try { loadLogs(); } catch(e) { try{ console.error('loadLogs()', e); }catch(_){} }
    try { loadCfg(); } catch(e) { try{ console.error('loadCfg()', e); }catch(_){} }
    try { loadResults(); } catch(e) { try{ console.error('loadResults()', e); }catch(_){} }
    try { loadPredLabels(); } catch(e) { try{ console.error('loadPredLabels()', e); }catch(_){} }

    qs('#btn-refresh-logs').addEventListener('click', loadLogs);
    qs('#sel-log-date').addEventListener('change', loadLogs);
    qs('#btn-refresh-cfg').addEventListener('click', loadCfg);
    const btnRes = qs('#btn-refresh-results'); if (btnRes) btnRes.addEventListener('click', loadResults);
    const btnCurv = qs('#btn-refresh-curves'); if (btnCurv) btnCurv.addEventListener('click', refreshCurves);
    const btnSer = qs('#btn-refresh-series'); if (btnSer) btnSer.addEventListener('click', refreshSeries);

    const selLog = qs('#sel-log'); if (selLog) selLog.addEventListener('change', ()=>{ updateLogMeta(); loadTailForSelected(); });
    const btnTex = qs('#btn-tex'); if (btnTex) btnTex.addEventListener('click', ()=>{ try { window.open('/tex?rel=' + encodeURIComponent('pipeline_documentacion.tex'), '_blank'); } catch(e){} });

    let timer = setInterval(()=>{ const a = qs('#auto-logs'); if (a && a.checked) { try { loadTailForSelected(); } catch(e){} } }, 3000);

    // Mostrar listas de archivos por defecto (antes se ocultaban)
    // Si prefieres ocultarlas, descomenta el bloque siguiente.
    // ['tbl-resultados','tbl-datos','tbl-src','tbl-tests'].forEach(id => {
    //   const el = qs('#'+id);
    //   if (el) {
    //     const col = el.closest('.col');
    //     if (col) col.style.display = 'none';
    //   }
    // });

    qs('#btn-shutdown').addEventListener('click', async () => {
      if (confirm('¿Seguro que quieres detener el servidor y volver a la terminal?')) {
        try {
          await fetchJSON('/api/shutdown');
          document.body.innerHTML = '<h1>Servidor detenido. Puedes cerrar esta ventana.</h1>';
        } catch (e) {
          alert('No se pudo detener el servidor.');
        }
      }
    });
  });
  // Diagnóstico básico: mostrar errores JS en Overview
  window.addEventListener('error', (ev) => { try { const el = qs('#overview'); if (el) el.textContent = 'Error JS: ' + (ev && ev.message ? ev.message : 'desconocido'); } catch(_) {} });
  window.addEventListener('unhandledrejection', (ev) => { try { const el = qs('#overview'); if (el) el.textContent = 'Error: ' + (ev && ev.reason ? (ev.reason.message || String(ev.reason)) : 'rechazo de promesa'); } catch(_) {} });
  </script>
</body>
</html>
""";


LATEX_HTML = """
<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Documento LaTeX</title>
  <link rel=\"stylesheet\" href=\"/static/latexjs/latex.min.css\" />
  <style>
    body { margin: 0; background: #f6f6f6; }
    .topbar { position: sticky; top: 0; background:#fff; border-bottom:1px solid #eee; padding:10px; display:flex; gap:8px; align-items:center; z-index: 5; }
    .container { max-width: 900px; margin: 16px auto; background:#fff; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .muted { color: #777; }
    button { padding:6px 10px; }
    pre.err { background:#fff3f3; color:#b71c1c; padding:10px; border:1px solid #f5c2c7; border-radius:6px; white-space: pre-wrap; }
  </style>
  <script src=\"/static/latexjs/latex.min.js\"></script>
  <script>
    async function fetchJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error('HTTP '+r.status); return await r.json(); }
    function getParam(name){ const u=new URL(window.location.href); return u.searchParams.get(name); }
    async function renderLatex(){
      const rel = getParam('rel') || 'pipeline_documentacion.tex';
      document.getElementById('meta').textContent = rel;
      try {
        const data = await fetchJSON('/api/read_text?rel=' + encodeURIComponent(rel));
        const tex = data && data.content ? data.content : '';
        const out = document.getElementById('out'); out.innerHTML = '';
        try {
          const gen = new latexjs.HtmlGenerator({ hyphenate: true });
          latexjs.parse(tex, { generator: gen });
          if (gen.domFragment) { out.appendChild(gen.domFragment()); }
          else if (gen.documentFragment) { out.appendChild(gen.documentFragment()); }
          else if (gen.container) { out.appendChild(gen.container); }
          else { out.innerHTML = '<pre class="err">No se pudo renderizar con latex.js.\n\n' + tex.replace(/[&<>]/g, m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m])) + '</pre>'; }
        } catch (e) {
          out.innerHTML = '<pre class="err">Error latex.js: ' + (e && e.message ? e.message : e) + '\n\n' + tex.replace(/[&<>]/g, m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m])) + '</pre>';
        }
      } catch (e) {
        document.getElementById('out').innerHTML = '<pre class="err">No se pudo leer el archivo: ' + (e && e.message ? e.message : e) + '</pre>';
      }
    }
    document.addEventListener('DOMContentLoaded', () => {
      document.getElementById('btn-refresh').addEventListener('click', renderLatex);
      document.getElementById('btn-home').addEventListener('click', () => { window.location.href = '/'; });
      renderLatex();
    });
  </script>
</head>
<body>
  <div class=\"topbar\">
    <button id=\"btn-home\">Inicio</button>
    <button id=\"btn-refresh\">Recargar</button>
    <span class=\"muted\">Archivo: <code id=\"meta\"></code></span>
  </div>
  <div class=\"container\">
    <div id=\"out\" class=\"latex\">Cargando…</div>
  </div>
</body>
</html>
""";


class VisualHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # silenciar logs de acceso del servidor
        return

    def do_GET(self):  # noqa: N802 (for style linters)
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            # Servir estáticos locales bajo /static/* desde src/static/*
            if path.startswith("/static/"):
                root = _project_root()
                static_base = os.path.join(root, "src", "static")
                # Construir ruta absoluta segura
                rel_sub = path[len("/static/"):]
                full = os.path.abspath(os.path.join(static_base, rel_sub))
                # Validar que permanece dentro de static_base
                if os.path.commonpath([static_base, full]) != static_base or not os.path.isfile(full):
                    _error(self, 404, "Recurso estático no encontrado")
                    return
                try:
                    with open(full, "rb") as fh:
                        body = fh.read()
                    # Content-Type mínimo por extensión
                    ctype = "application/octet-stream"
                    ext = os.path.splitext(full)[1].lower()
                    if ext == ".js":
                        ctype = "application/javascript"
                    elif ext == ".css":
                        ctype = "text/css; charset=utf-8"
                    elif ext in (".svg",):
                        ctype = "image/svg+xml"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:
                    _error(self, 500, f"Error sirviendo estático: {e}")
                return

            if path == "/" or path == "/index.html":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/overview":
                _json_response(self, _overview_payload())
                return

            if path == "/api/config":
                _json_response(self, _collect_config())
                return

            if path == "/api/logs":
                try:
                    limit = int(qs.get("limit", ["40"])[0])
                except Exception:
                    limit = 40
                _json_response(self, _logs_list_payload(limit=limit))
                return

            if path == "/api/tail":
                rel = qs.get("rel", [None])[0]
                if not rel:
                    _error(self, 400, "Falta parámetro 'rel'")
                    return
                _json_response(self, _tail_payload(rel))
                return

            if path == "/api/preview":
                rel = qs.get("rel", [None])[0]
                if not rel:
                    _error(self, 400, "Falta parámetro 'rel'")
                    return
                _json_response(self, _preview_payload(rel))
                return

            if path == "/api/read_text":
                rel = qs.get("rel", [None])[0]
                if not rel:
                    _error(self, 400, "Falta parámetro 'rel'")
                    return
                _json_response(self, _read_text_payload(rel))
                return

            if path == "/api/files":
                sec = qs.get("section", [None])[0]
                if not sec:
                    _error(self, 400, "Falta parámetro 'section'")
                    return
                try:
                    limit = int(qs.get("limit", ["2000"])[0])
                except Exception:
                    limit = 2000
                _json_response(self, _list_section_payload(sec, limit=limit))
                return

            if path == "/api/results/summary":
                _json_response(self, _results_summary_payload())
                return

            if path == "/api/results/timeseries":
                metric = qs.get("metric", ["f1_1"])[0]
                _json_response(self, _results_timeseries_payload(metric=metric))
                return

            if path == "/api/preds/labels":
                _json_response(self, _preds_labels_payload())
                return

            if path == "/api/shutdown":
                _json_response(self, {"message": "Server is shutting down."})
                import threading
                threading.Thread(target=self.server.shutdown).start()
                return

            if path == "/api/curves":
                label = qs.get("label", [None])[0]
                split = qs.get("split", ["test"])[0]
                if not label:
                    _error(self, 400, "Falta parámetro 'label'")
                    return
                best = _latest_preds_by_label_split()
                key = (label, split)
                it = best.get(key)
                if not it:
                    _json_response(self, {"label": label, "split": split, "pr": {"precision":[], "recall":[]}, "roc": {"fpr":[], "tpr":[]}, "n": 0})
                    return
                meta = _load_preds_df(it['path'])
                if not meta:
                    _json_response(self, {"label": label, "split": split, "pr": {"precision":[], "recall":[]}, "roc": {"fpr":[], "tpr":[]}, "n": 0})
                    return
                curves = _pr_roc_from_df(meta['df'], meta['y_col'], meta['score_col'])
                _json_response(self, curves)
                return

            if path == "/api/confusion":
                label = qs.get("label", [None])[0]
                split = qs.get("split", ["test"])[0]
                if not label:
                    _error(self, 400, "Falta parámetro 'label'")
                    return
                best = _latest_preds_by_label_split()
                it = best.get((label, split))
                if not it:
                    _json_response(self, {"cm": [[0,0],[0,0]], "metrics": {}})
                    return
                meta = _load_preds_df(it['path'])
                if not meta:
                    _json_response(self, {"cm": [[0,0],[0,0]], "metrics": {}})
                    return
                payload = _confusion_from_df(meta['df'], meta['y_col'], meta['ypred_col'])
                _json_response(self, payload)
                return

            if path == "/tex":
                # Página que renderiza LaTeX en el navegador con latex.js
                body = LATEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            _error(self, 404, "Ruta no encontrada")
        except Exception as e:
            _error(self, 500, f"Error interno: {e}")


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """HTTPServer que silencia errores esperables como ConnectionReset/BrokenPipe.
    Útil cuando el navegador cierra conexiones al recargar/pestaña cerrada.
    """
    daemon_threads = True

    def handle_error(self, request, client_address):  # noqa: D401
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            # Silenciar errores de red esperables
            return
        # Fallback: comportamiento mínimo (sin traceback ruidoso)
        try:
            print(f"[HTTP-Error] {client_address}: {exc}")
        except Exception:
            pass


def _pick_free_port(host: str = "127.0.0.1", prefer: Optional[int] = None) -> int:
    if prefer:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, prefer))
                return prefer
        except Exception:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def run_visual_server(host: str = "127.0.0.1", port: Optional[int] = None, open_browser: bool = True):
    """
    Levanta un servidor HTTP local que muestra el reporte visual dinámico.
    Presiona Ctrl+C para detenerlo.
    """
    p = _pick_free_port(host, prefer=port)
    server = QuietThreadingHTTPServer((host, p), VisualHandler)
    url = f"http://{host}:{p}/"
    print(f"🌐 Servidor de Reporte Visual en {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo servidor…")
    finally:
        server.server_close()
