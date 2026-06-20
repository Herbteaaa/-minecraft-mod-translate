import os
import zipfile
import json
import shutil
import tempfile
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from huggingface_hub import HfFolder
except Exception:
    HfFolder = None

# ============================
# NLLB-200 翻訳モデルロード
# ============================
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hf_cache")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN and HfFolder is not None:
    HF_TOKEN = HfFolder.get_token()
tokenizer = None
model = None
TARGET_BOS_TOKEN_ID = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_TRANSLATION_SIZE = 16


def load_translator():
    global tokenizer, model, TARGET_BOS_TOKEN_ID

    if tokenizer is not None and model is not None:
        return

    print(f"[LOAD] NLLB モデルを読み込み中: {TRANSLATION_MODEL_NAME}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    load_kwargs = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME, **load_kwargs)
    if DEVICE.type == "cuda":
        model = model.to(device=DEVICE, dtype=torch.float16)
    else:
        model = model.to(device=DEVICE)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "jpn_Jpan"
    TARGET_BOS_TOKEN_ID = tokenizer.convert_tokens_to_ids("jpn_Jpan")
    model.eval()
    print("[LOAD] NLLB モデルの読み込み完了")

# ============================
# 翻訳対象ファイルのパターン
# ============================
TARGET_NAMES = [
    "en_us.json",
    "en_us.lang"
]

GLOSSARY_DIR = os.path.join(os.path.dirname(__file__), "glossaries")

PLACEHOLDER_PATTERN = re.compile(
    r"(%(?:\d+\$)?[sdfox]|\{\d+\}|\{[A-Za-z_][A-Za-z0-9_]*\}|§.|<[^>]+>|\$\{[^}]+\})"
)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。．.!?！？])\s+|\n+")
JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
MIXED_SEGMENT_PATTERN = re.compile(r"__PLACEHOLDER_\d+__|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9_][A-Za-z0-9_\-\./ ]*")

DEFAULT_GLOSSARY = {
    "block": "ブロック",
    "item": "アイテム",
    "entity": "エンティティ",
    "mode": "モード",
    "level": "レベル",
    "difficulty": "難易度",
    "creative": "クリエイティブ",
    "survival": "サバイバル",
}


def load_glossary(mod_name: str | None = None):
    glossary = dict(DEFAULT_GLOSSARY)

    common_path = os.path.join(GLOSSARY_DIR, "common.json")
    if os.path.exists(common_path):
        with open(common_path, "r", encoding="utf-8") as f:
            glossary.update(json.load(f))

    if mod_name:
        mod_path = os.path.join(GLOSSARY_DIR, f"{mod_name}.json")
        if os.path.exists(mod_path):
            with open(mod_path, "r", encoding="utf-8") as f:
                glossary.update(json.load(f))

    return glossary


def build_glossary_pattern(glossary: dict[str, str]):
    if not glossary:
        return None

    return re.compile(
        r"\b(" + "|".join(re.escape(key) for key in sorted(glossary, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )


def apply_custom_dict(text: str, glossary: dict[str, str], pattern=None) -> str:
    if not glossary:
        return text

    if pattern is None:
        pattern = build_glossary_pattern(glossary)

    def repl(match):
        return glossary[match.group(1).lower()]

    return pattern.sub(repl, text)


def protect_placeholders(text: str):
    placeholders = []

    def repl(match):
        placeholders.append(match.group(0))
        return f"__PLACEHOLDER_{len(placeholders) - 1}__"

    return PLACEHOLDER_PATTERN.sub(repl, text), placeholders


def restore_placeholders(text: str, placeholders):
    for index, placeholder in enumerate(placeholders):
        text = text.replace(f"__PLACEHOLDER_{index}__", placeholder)
    return text


def split_translation_chunks(text: str):
    chunks = []
    for part in SENTENCE_SPLIT_PATTERN.split(text):
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks or [text]


def translate_mixed_text(text, glossary):
    parts = []
    last_index = 0

    for match in MIXED_SEGMENT_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_index:
            parts.append(text[last_index:start])

        segment = match.group(0)
        if segment.startswith("__PLACEHOLDER_") and segment.endswith("__"):
            parts.append(segment)
            last_index = end
            continue

        if JAPANESE_PATTERN.search(segment):
            parts.append(segment)
        else:
            parts.append(translate_text(segment, glossary))

        last_index = end

    if last_index < len(text):
        parts.append(text[last_index:])

    return "".join(parts)


def detect_modid(temp_dir, full_path):
    rel_path = os.path.relpath(full_path, temp_dir)
    parts = rel_path.split(os.sep)

    if len(parts) >= 4 and parts[0] == "assets" and parts[2] == "lang":
        return parts[1]

    return None


def translate_value(value, glossary):
    if isinstance(value, str):
        return translate_text(value, glossary)

    if isinstance(value, list):
        return [translate_value(item, glossary) for item in value]

    if isinstance(value, dict):
        return {key: translate_value(inner_value, glossary) for key, inner_value in value.items()}

    return value


def translate_text_single(text, glossary=None):
    if not isinstance(text, str):
        return text

    if not text.strip():
        return text

    if glossary is None:
        glossary = DEFAULT_GLOSSARY

    protected_text, placeholders = protect_placeholders(text)

    if JAPANESE_PATTERN.search(protected_text):
        translated = translate_mixed_text(protected_text, glossary)
        translated = restore_placeholders(translated, placeholders)
        return apply_custom_dict(translated, glossary)

    load_translator()

    protected_text = apply_custom_dict(protected_text, glossary)

    try:
        inputs = tokenizer([protected_text], return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=TARGET_BOS_TOKEN_ID,
                max_new_tokens=256,
            )
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        print(f"[WARN] 翻訳失敗: {text}")
        return text

    translated = restore_placeholders(translated, placeholders)
    return apply_custom_dict(translated, glossary)


def translate_values(values, glossary=None):
    if glossary is None:
        glossary = DEFAULT_GLOSSARY

    if not values:
        return []

    if len(values) == 1:
        return [translate_text_single(values[0], glossary)]

    prepared_items = []
    translated_values = [None] * len(values)

    for index, text in enumerate(values):
        if not isinstance(text, str):
            translated_values[index] = text
            continue

        if not text.strip():
            translated_values[index] = text
            continue

        protected_text, placeholders = protect_placeholders(text)

        if JAPANESE_PATTERN.search(protected_text):
            translated = translate_mixed_text(protected_text, glossary)
            translated = restore_placeholders(translated, placeholders)
            translated_values[index] = apply_custom_dict(translated, glossary)
            continue

        prepared_items.append((index, apply_custom_dict(protected_text, glossary), placeholders))

    if prepared_items:
        load_translator()

        try:
            for start in range(0, len(prepared_items), BATCH_TRANSLATION_SIZE):
                batch = prepared_items[start:start + BATCH_TRANSLATION_SIZE]
                batch_texts = [item[1] for item in batch]
                batch_placeholders = [item[2] for item in batch]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=TARGET_BOS_TOKEN_ID,
                        max_new_tokens=256,
                    )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for (index, _, placeholders), translated in zip(batch, decoded):
                    translated = restore_placeholders(translated, placeholders)
                    translated_values[index] = apply_custom_dict(translated, glossary)
        except Exception:
            print("[WARN] バッチ翻訳失敗。個別翻訳にフォールバックします。")
            for index, text, _placeholders in prepared_items:
                translated_values[index] = translate_text_single(text, glossary)

    return translated_values


def translate_structure(value, glossary):
    texts = []

    def collect(node):
        if isinstance(node, str):
            texts.append(node)
        elif isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            for inner_value in node.values():
                collect(inner_value)

    def rebuild(node, translated_iter):
        if isinstance(node, str):
            return next(translated_iter)

        if isinstance(node, list):
            return [rebuild(item, translated_iter) for item in node]

        if isinstance(node, dict):
            return {key: rebuild(inner_value, translated_iter) for key, inner_value in node.items()}

        return node

    collect(value)
    translated_texts = translate_values(texts, glossary)
    return rebuild(value, iter(translated_texts))


# ============================
# JSON / LANG 翻訳
# ============================
def translate_text(text, glossary=None):
    return translate_text_single(text, glossary)


def translate_json_file(path, glossary):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = translate_structure(data, glossary)

    out_path = path.replace("en_us.json", "ja_jp.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def translate_lang_file(path, glossary):
    out_lines = []
    keys = []
    values = []
    markers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                keys.append(key)
                values.append(val.strip())
                markers.append(len(out_lines))
                out_lines.append(None)
            else:
                out_lines.append(line)

    translated_values = translate_values(values, glossary)
    for key, position, translated in zip(keys, markers, translated_values):
        out_lines[position] = f"{key}={translated}\n"

    out_path = path.replace("en_us.lang", "ja_jp.lang")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


# ============================
# jar を解凍して翻訳
# ============================
def process_jar(jar_path, output_dir):
    print(f"翻訳中: {jar_path}")

    temp_dir = tempfile.mkdtemp()
    translated_count = 0
    failed_count = 0

    # jar を展開
    with zipfile.ZipFile(jar_path, "r") as jar:
        jar.extractall(temp_dir)

    target_files = []

    # 翻訳対象を探索
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file in TARGET_NAMES:
                full_path = os.path.join(root, file)
                target_files.append(full_path)

    total_targets = len(target_files)
    if total_targets == 0:
        print("  → 翻訳対象なし")
    else:
        for index, full_path in enumerate(target_files, start=1):
            file = os.path.basename(full_path)
            print(f"  → [{index}/{total_targets}] 翻訳対象: {full_path}")
            modid = detect_modid(temp_dir, full_path)
            glossary = load_glossary(modid)

            if modid:
                print(f"    [GLOSSARY] {modid}")
            else:
                print("    [GLOSSARY] common")

            if file.endswith(".json"):
                try:
                    translate_json_file(full_path, glossary)
                    translated_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"    [WARN] JSON 翻訳失敗: {e}")
            elif file.endswith(".lang"):
                try:
                    translate_lang_file(full_path, glossary)
                    translated_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"    [WARN] LANG 翻訳失敗: {e}")

    # jar を再パック
    base_name = os.path.basename(jar_path)
    out_path = os.path.join(output_dir, base_name)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as new_jar:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, temp_dir)
                new_jar.write(full_path, arcname)

    shutil.rmtree(temp_dir)
    print(f"翻訳完了: {out_path} (成功 {translated_count}, 失敗 {failed_count})")
    return translated_count, failed_count


# ============================
# メイン処理
# ============================
def main():
    mods_dir = r"C:\Users\herbt\curseforge\minecraft\Instances\MODS(1)\mods"
    out_dir = r"C:\Users\herbt\curseforge\minecraft\Instances\translated_mods"

    os.makedirs(out_dir, exist_ok=True)

    jar_files = [file for file in os.listdir(mods_dir) if file.endswith(".jar")]
    total_jars = len(jar_files)
    total_translated = 0
    total_failed = 0

    for index, file in enumerate(jar_files, start=1):
        full_path = os.path.join(mods_dir, file)
        print(f"[JAR {index}/{total_jars}]")
        translated_count, failed_count = process_jar(full_path, out_dir)
        total_translated += translated_count
        total_failed += failed_count

    print(f"すべての MOD の翻訳が完了しました。成功 {total_translated}, 失敗 {total_failed}")


if __name__ == "__main__":
    main()
