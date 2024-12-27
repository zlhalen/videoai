import os
import random
import string

import requests
from asr_main import ASRPipeline, logger
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

TMP_DIR = "./cached_data"
os.makedirs(TMP_DIR, exist_ok=True)

asr = ASRPipeline()


def generate_random_string(length=6):
    characters = string.ascii_letters + string.digits  # 包括 a-z, A-Z 和 0-9
    random_string = "".join(
        random.choices(characters, k=length)
    )  # 生成指定长度的随机字符串
    return random_string


def aug_fn(fn: str) -> str:
    base, ext = os.path.splitext(fn)
    new_fn = base + "." + generate_random_string() + ext
    return new_fn


@app.route("/asr_seg", methods=["POST"])
def rec_audio():
    file = request.files.get("file", None)
    furl = request.form.get("url")

    tmp_path = ""
    if file:
        fn = file.filename
        tmp_path = os.path.join(TMP_DIR, aug_fn(fn))
        file.save(tmp_path)
    elif furl:
        logger.info(f"Downloading: {furl}")
        try:
            resp = requests.get(furl, stream=True)
            resp.raise_for_status()
            fn = os.path.basename(furl)
            tmp_path = os.path.join(TMP_DIR, aug_fn(fn))
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            return jsonify({"msg": f"error when downloading from {furl}: {e}"}), 400
    else:
        return jsonify({"msg": "error: No video file provided"}), 400

    try:
        asr_result = asr(tmp_path)
        asr_result["msg"] = "success"
    except Exception as e:
        asr_result = {"msg": f"error: {e}"}

    # delete tmp paths
    os.remove(tmp_path)

    return asr_result


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1])
    app.run(host="0.0.0.0", port=port, processes=1)
