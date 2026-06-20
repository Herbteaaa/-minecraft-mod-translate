import os
import zipfile
import json
import shutil
import tempfile
from argostranslate import translate

# ============================
# Argos 翻訳モデルロード
# ============================
installed_languages = translate.get_installed_languages()
en = next(lang for lang in installed_languages if lang.code == "en")
ja = next(lang for lang in installed_languages if lang.code == "ja")
translator = en.get_translation(ja)

# ============================
# 翻訳対象ファイルのパターン
# ============================
TARGET_NAMES = [
    "en_us.json",
    "en_us.lang"
]

# ============================
# JSON / LANG 翻訳
# ============================
def translate_text(text):
    return translator.translate(text)

def translate_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keys = []
    values = []

    # 値が list の場合は結合して文字列化
    for k, v in data.items():
        if isinstance(v, list):
            v = "\n".join(str(x) for x in v)
        else:
            v = str(v)
            keys.append(k)
            values.append(v)

    # まとめ翻訳
    joined = "\n".join(values)
    translated = translate_text(joined).split("\n")

    # JSON 再構築
    result = {k: v for k, v in zip(keys, translated)}

    out_path = path.replace("en_us.json", "ja_jp.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def translate_lang_file(path):
    out_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                translated = translate_text(val.strip())
                out_lines.append(f"{key}={translated}\n")
            else:
                out_lines.append(line)

    out_path = path.replace("en_us.lang", "ja_jp.lang")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

# ============================
# jar を解凍して翻訳
# ============================
def process_jar(jar_path, output_dir):
    print(f"翻訳中: {jar_path}")

    temp_dir = tempfile.mkdtemp()

    # jar を展開
    with zipfile.ZipFile(jar_path, "r") as jar:
        jar.extractall(temp_dir)

    # 翻訳対象を探索
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file in TARGET_NAMES:
                full_path = os.path.join(root, file)
                print("  → 翻訳対象:", full_path)

                if file.endswith(".json"):
                    translate_json_file(full_path)
                elif file.endswith(".lang"):
                    translate_lang_file(full_path)

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
    print("翻訳完了:", out_path)

# ============================
# メイン処理
# ============================
def main():
    mods_dir = r"C:\Users\herbt\curseforge\minecraft\Instances\MODS(1)\mods"
    out_dir = r"C:\Users\herbt\curseforge\minecraft\Instances\translated_mods"

    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(mods_dir):
        if file.endswith(".jar"):
            full_path = os.path.join(mods_dir, file)
            process_jar(full_path, out_dir)

    print("すべての MOD の翻訳が完了しました。")

if __name__ == "__main__":
    main()
