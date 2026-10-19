#!/usr/bin/env python3
"""
Sync builtin_providers.toml with the latest provider & model catalog from models.dev.

Source of truth: https://models.dev/api.json (the open-source AI model database
maintained by OpenCode / SST and synced hourly against upstream provider APIs).

Policy:
- ZERO hardcoded model generations or family caps for first-party providers:
  all active (`status != "deprecated"`), text-in/text-out (`"text"` in modalities),
  tool-capable (`tool_call == true`) chat models are included automatically.
- Models within each provider are sorted by `release_date` descending (newest
  first), matching OpenCode's model picker ordering.
- Dated snapshots (`-YYYYMMDD`, `-YYYY-MM-DD`) collapse into their undated alias
  when both exist in the same provider.
- Non-coding specialized endpoints (image/audio/video/tts/whisper/realtime/
  embedding/moderation/robotics/lyria/veo/sora/guard/deep-research/computer-use)
  are excluded.
- Reasoning effort (`off | low | medium | high | max`), `effort_control`, and
  `effort_always_on` are derived directly from `reasoning` and `reasoning_options`
  in models.dev, with optional manual overrides in `provider_overlay.toml`.
- Freshness uses a monotonic `serial`: when catalog content changes, `serial`
  increments by 1. If nothing changed except the timestamp/serial stamps, the
  file is left untouched so git produces no empty commit.
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.request

try:
    import tomllib
except ImportError:
    tomllib = None

MODELS_DEV_URL = "https://models.dev/api.json"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CATALOG_PATH = os.path.join(ROOT_DIR, "builtin_providers.toml")
OVERLAY_PATH = os.path.join(os.path.dirname(__file__), "provider_overlay.toml")

SERIAL_RE = re.compile(r"^serial = (\d+)$", re.M)
STAMP_RE = re.compile(r'^(serial = \d+|updated_at = ".*")$', re.M)
DATE_SUFFIX = re.compile(r"-20\d{6}$|-20\d\d-\d\d-\d\d$")

EXCLUDE_SUBSTRINGS = (
    "audio",
    "image",
    "transcribe",
    "tts",
    "whisper",
    "realtime",
    "embedding",
    "rerank",
    "diarize",
    "moderation",
    "robotics",
    "lyria",
    "veo",
    "sora",
    "imagen",
    "flux",
    "character",
    "mt-turbo",
    "mt-plus",
    "safeguard",
    "llama-guard",
    "prompt-guard",
    "computer-use",
    "live-preview",
    "live-translate",
    "deep-research",
    "voxtral",
    "ocr",
)

EFFORT_VALUES = ("off", "low", "medium", "high", "max")

EFFORT_HIGH_HINTS = (
    "opus",
    "fable",
    "mythos",
    "argon",
    "pro",
    "codex",
    "reasoner",
    "r1",
    "sol",
    "terra",
    "astra",
    "build",
    "k3",
    "k2.7-code",
    "o1",
    "o3",
    "max",
    "thinking",
    "qwq",
)
EFFORT_LOW_HINTS = ("luna", "mini", "nano", "lite")


def has_token_hint(model_id, hints):
    low = model_id.lower()
    tokens = set(re.split(r"[-._/:]+", low))
    for h in hints:
        if "-" in h or "." in h:
            if h in low:
                return True
        elif h in tokens:
            return True
    return False

# Pinned built-in providers in display priority order.
# First-party labs come first so their canonical model IDs always claim unprefixed keys.
PROVIDERS_CONFIG = [
    {
        "name": "gemini",
        "title": "Gemini",
        "models_dev_id": "google",
        "format": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "continuation": True,
    },
    {
        "name": "anthropic",
        "title": "Anthropic",
        "models_dev_id": "anthropic",
        "format": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "continuation": True,
    },
    {
        "name": "openai",
        "title": "OpenAI",
        "models_dev_id": "openai",
        "format": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "continuation": True,
    },
    {
        "name": "deepseek",
        "title": "DeepSeek",
        "models_dev_id": "deepseek",
        "format": "openai",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "continuation": True,
    },
    {
        "name": "grok",
        "title": "Grok",
        "models_dev_id": "xai",
        "format": "openai",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "continuation": True,
    },
    {
        "name": "kimi",
        "title": "Kimi",
        "models_dev_id": "moonshotai",
        "format": "openai",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "continuation": True,
    },
    {
        "name": "alibaba",
        "title": "Alibaba (Qwen)",
        "models_dev_id": "alibaba",
        "format": "openai",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "continuation": True,
    },
    {
        "name": "zai",
        "title": "Z.AI (GLM)",
        "models_dev_id": "zai",
        "format": "openai",
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "continuation": True,
    },
    {
        "name": "minimax",
        "title": "MiniMax",
        "models_dev_id": "minimax",
        "format": "anthropic",
        "base_url": "https://api.minimax.io/anthropic/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "continuation": True,
    },
    {
        "name": "mistral",
        "title": "Mistral",
        "models_dev_id": "mistral",
        "format": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "continuation": True,
    },
    {
        "name": "cohere",
        "title": "Cohere",
        "models_dev_id": "cohere",
        "format": "openai",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "api_key_env": "COHERE_API_KEY",
        "continuation": True,
    },
    {
        "name": "groq",
        "title": "Groq",
        "models_dev_id": "groq",
        "format": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "continuation": True,
    },
    {
        "name": "cerebras",
        "title": "Cerebras",
        "models_dev_id": "cerebras",
        "format": "openai",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "continuation": True,
    },
    {
        "name": "togetherai",
        "title": "Together AI",
        "models_dev_id": "togetherai",
        "format": "openai",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "continuation": True,
    },
    {
        "name": "fireworks",
        "title": "Fireworks AI",
        "models_dev_id": "fireworks-ai",
        "format": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "continuation": True,
    },
    {
        "name": "deepinfra",
        "title": "Deep Infra",
        "models_dev_id": "deepinfra",
        "format": "openai",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
        "continuation": True,
    },
    {
        "name": "nvidia",
        "title": "NVIDIA",
        "models_dev_id": "nvidia",
        "format": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "continuation": True,
    },
    {
        "name": "openrouter",
        "title": "OpenRouter",
        "models_dev_id": "openrouter",
        "format": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "continuation": True,
        # OpenRouter mirrors 300+ routes; keep the freshest 80 tool-capable models
        # (plus any free routes) so the catalog stays fast and compact.
        "max_models": 80,
    },
]


def fetch_models_dev():
    req = urllib.request.Request(
        MODELS_DEV_URL,
        headers={"User-Agent": "sqwai-providers-sync/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"ERROR: could not fetch models.dev catalog: {e}", file=sys.stderr)
        sys.exit(1)


def load_overlay():
    if tomllib is None or not os.path.exists(OVERLAY_PATH):
        return {}
    try:
        with open(OVERLAY_PATH, "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        print(f"  ! overlay unreadable ({e}), ignored", file=sys.stderr)
        return {}
    out = {}
    for provider, table in raw.items():
        if not isinstance(table, dict):
            continue
        for mid, spec in table.items():
            if not isinstance(spec, dict):
                continue
            try:
                ctx = int(spec["context"])
                effort = str(spec["effort"])
            except (KeyError, ValueError, TypeError):
                continue
            if ctx <= 0 or effort not in EFFORT_VALUES:
                continue
            out.setdefault(provider, {})[mid] = {
                "context": ctx,
                "effort": effort,
                "effort_control": spec.get("effort_control"),
                "effort_always_on": bool(spec.get("effort_always_on", False)),
                "release_date": str(spec.get("release_date", "2026-01-01")),
            }
    return out


def usable_model(m):
    if not isinstance(m, dict):
        return False
    mid = m.get("id")
    if not isinstance(mid, str) or not mid:
        return False
    if m.get("status") == "deprecated":
        return False
    if m.get("type", "chat") != "chat":
        return False
    if not m.get("tool_call", False):
        return False
    modalities = m.get("modalities") or {}
    inputs = modalities.get("input") or ["text"]
    outputs = modalities.get("output") or ["text"]
    if not any(str(x).startswith("text") for x in inputs):
        return False
    if not any(str(x).startswith("text") for x in outputs):
        return False
    low = mid.lower()
    if any(sub in low for sub in EXCLUDE_SUBSTRINGS):
        return False
    limit = m.get("limit") or {}
    ctx = limit.get("context") or limit.get("input") or 0
    if not isinstance(ctx, (int, float)) or int(ctx) <= 0:
        return False
    return True


def collapse_dated_snapshots(models_dict):
    """When both an undated alias and a dated snapshot (-YYYYMMDD / -YYYY-MM-DD)
    exist in the same provider, keep the undated alias."""
    by_base = {}
    for mid, m in models_dict.items():
        base = DATE_SUFFIX.sub("", mid)
        by_base.setdefault(base, []).append((mid, m))
    kept = []
    for base, items in by_base.items():
        if len(items) == 1:
            kept.append(items[0][1])
            continue
        # If the exact undated base id exists, prefer it; otherwise keep the newest release_date
        exact = [m for mid, m in items if mid == base]
        if exact:
            # Propagate the newest release_date from snapshots to the alias if the alias has an older one
            newest_date = max((m.get("release_date") or "") for _, m in items)
            chosen = dict(exact[0])
            if newest_date and newest_date > (chosen.get("release_date") or ""):
                chosen["release_date"] = newest_date
            kept.append(chosen)
        else:
            items.sort(
                key=lambda kv: (kv[1].get("release_date") or "", kv[0]),
                reverse=True,
            )
            kept.append(items[0][1])
    return kept


def resolve_effort(m):
    mid = m["id"]
    low = mid.lower()
    reasoning = bool(m.get("reasoning", False))
    if not reasoning:
        return "off", None, False

    ropts = m.get("reasoning_options")
    # Pick default effort level based on model tier and supported effort values
    allowed_efforts = set()
    has_effort_opt = False
    has_budget_opt = False
    has_toggle_opt = False
    if isinstance(ropts, list):
        for opt in ropts:
            if not isinstance(opt, dict):
                continue
            otype = opt.get("type")
            if otype == "effort":
                has_effort_opt = True
                for v in opt.get("values") or []:
                    if isinstance(v, str):
                        allowed_efforts.add(v)
            elif otype == "budget_tokens":
                has_budget_opt = True
            elif otype == "toggle":
                has_toggle_opt = True

    if has_token_hint(mid, EFFORT_HIGH_HINTS):
        default_effort = "high"
    elif has_token_hint(mid, EFFORT_LOW_HINTS):
        default_effort = "low"
    else:
        default_effort = "medium"

    if allowed_efforts:
        if default_effort not in allowed_efforts:
            for fallback in ("high", "medium", "low", "max"):
                if fallback in allowed_efforts:
                    default_effort = fallback
                    break

    effort_control = None
    effort_always_on = False
    if isinstance(ropts, list):
        if len(ropts) == 0:
            # Empty reasoning_options in models.dev means always-on reasoning with no caller control
            effort_control = "none"
            effort_always_on = True
        elif has_toggle_opt and not has_effort_opt and not has_budget_opt:
            effort_control = "toggle"

    return default_effort, effort_control, effort_always_on


def build_catalog(data, today, serial, overlay):
    lines = [
        "# Monotonic freshness stamp, bumped by scripts/update_providers.py on",
        "# every content change (never by hand). The client compares serials, not",
        "# dates: a replayed old file never wins, and missing serial reads as 0.",
        f"serial = {serial}",
        f'updated_at = "{today}"',
        "",
    ]

    used_keys = set()
    total_models = 0
    provider_count = 0

    for p in PROVIDERS_CONFIG:
        pname = p["name"]
        mdev_id = p["models_dev_id"]
        src_prov = data.get(mdev_id)
        if not isinstance(src_prov, dict):
            print(f"WARNING: provider {pname} ({mdev_id}) missing in models.dev", file=sys.stderr)
            continue

        raw_models = src_prov.get("models") or {}
        usable = {
            mid: m
            for mid, m in raw_models.items()
            if usable_model(m)
        }
        collapsed = collapse_dated_snapshots(usable)

        # Merge any overlay entries for this provider
        prov_overlay = overlay.get(pname, {})
        have_ids = {m["id"] for m in collapsed}
        for omid, ospec in prov_overlay.items():
            if omid not in have_ids:
                collapsed.append(
                    {
                        "id": omid,
                        "release_date": ospec["release_date"],
                        "limit": {"context": ospec["context"]},
                        "_overlay": ospec,
                    }
                )

        # Sort newest first by release_date, then id descending (so 5-5 beats 5 on same date)
        collapsed.sort(
            key=lambda m: (str(m.get("release_date") or ""), m["id"]),
            reverse=True,
        )

        max_models = p.get("max_models")
        if max_models and len(collapsed) > max_models:
            collapsed = collapsed[:max_models]

        if not collapsed:
            print(f"WARNING: provider {pname} resolved 0 models, skipping", file=sys.stderr)
            continue

        provider_count += 1
        lines += [
            f"# {p['title']}",
            f"[providers.{pname}]",
            f'format = "{p["format"]}"',
            f'base_url = "{p["base_url"]}"',
            f'api_key_env = "{p["api_key_env"]}"',
            f"continuation = {'true' if p.get('continuation', True) else 'false'}",
            "",
        ]

        print(f"[{pname}] {len(collapsed)} models")
        for m in collapsed:
            mid = m["id"]
            if "_overlay" in m:
                ospec = m["_overlay"]
                ctx = ospec["context"]
                effort = ospec["effort"]
                effort_control = ospec.get("effort_control")
                effort_always_on = ospec.get("effort_always_on", False)
            elif mid in prov_overlay:
                ospec = prov_overlay[mid]
                ctx = ospec["context"]
                effort = ospec["effort"]
                effort_control = ospec.get("effort_control")
                effort_always_on = ospec.get("effort_always_on", False)
            else:
                limit = m.get("limit") or {}
                ctx = int(limit.get("context") or limit.get("input") or 128000)
                effort, effort_control, effort_always_on = resolve_effort(m)

            # First-party / unique models keep their bare ID as key; if a relay
            # provider serves the same model ID, namespace the TOML key with
            # `<provider>/<id>` while keeping `id = "<id>"` for wire requests.
            key = mid if mid not in used_keys else f"{pname}/{mid}"
            used_keys.add(key)
            total_models += 1

            lines += [
                f'[models."{key}"]',
                f'provider = "{pname}"',
                f'id = "{mid}"',
                f"context = {ctx}",
                f'effort = "{effort}"',
            ]
            if effort_control:
                lines.append(f'effort_control = "{effort_control}"')
            if effort_always_on:
                lines.append("effort_always_on = true")
            lines.append("")

    if provider_count < 6 or total_models < 50:
        print(
            f"ERROR: sanity check failed ({provider_count} providers, {total_models} models)",
            file=sys.stderr,
        )
        sys.exit(1)

    return "\n".join(lines) + "\n", provider_count, total_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=DEFAULT_CATALOG_PATH,
        help="Path to output builtin_providers.toml",
    )
    args = ap.parse_args()

    data = fetch_models_dev()
    overlay = load_overlay()
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    target = os.path.abspath(args.output)
    existing = None
    if os.path.exists(target):
        with open(target, encoding="utf-8") as f:
            existing = f.read()

    m = SERIAL_RE.search(existing) if existing else None
    serial = (int(m.group(1)) + 1) if m else 5

    content, prov_count, model_count = build_catalog(data, today, serial, overlay)

    if existing is not None and STAMP_RE.sub("", existing) == STAMP_RE.sub("", content):
        print(
            f"No catalog changes ({prov_count} providers, {model_count} models); leaving {target} untouched."
        )
        return

    # Validate generated TOML before atomic replace
    if tomllib is not None:
        parsed = tomllib.loads(content)
        assert len(parsed.get("providers", {})) == prov_count
        assert len(parsed.get("models", {})) == model_count

    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, target)
    print(
        f"Updated {target} with {prov_count} providers and {model_count} models (serial {serial})."
    )


if __name__ == "__main__":
    main()
