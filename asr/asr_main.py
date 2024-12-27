"""
ASR & Sentence splitting
"""

import logging
import os
import re

import moviepy.editor as mpy
import spacy
import whisper  # must import torch before spacy or mpy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
# logger.setLevel(logging.INFO)


def calculate_chinese_chars(s):
    if len(s) == 0:
        return (0, 0, 0.0)
    chinese_pattern = re.compile(r"[\u4e00-\u9fa5]+")
    chinese_characters = chinese_pattern.findall(s)
    chinese_count = sum(len(chars) for chars in chinese_characters)
    return (chinese_count, len(s), chinese_count / len(s))


def extract_audio(video_path):
    audio_suffixs = [".wav", ".mp3", ".aac", ".m4a", ".flac"]
    video_suffixs = [".mp4", ".avi", ".mkv", ".flv", ".mov", ".webm", ".ts", ".mpeg"]
    ext = os.path.splitext(video_path)[1].lower()
    if ext in audio_suffixs:
        return video_path, 0
    elif ext in video_suffixs:
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        video = mpy.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        return audio_path, 1
    else:
        raise NameError("Unsupported file format")


class ASRPipeline:
    def __init__(self):
        # for deployment in volcengine
        ckpt_whisper = "/ML-A800/team/mm/ccx/ckpt/whisper/large-v3.pt"
        if not os.path.exists(ckpt_whisper):
            ckpt_whisper = "large-v3"
        logger.info(f"loading checkpoints from: {ckpt_whisper}")

        self.asr_model = whisper.load_model(ckpt_whisper)
        self.sent_seg_en = spacy.load("en_core_web_sm")

        self.threshold_zh = 0.2
        self.max_time_diff = 5.0

        logger.info("Finished initializing ASRPipeline.")

    def __call__(self, test_file: str):
        """
        test_file: str, path of the audio or video file
        """
        test_file, rm_tag = extract_audio(test_file)
        logger.info(f"Recognizing the speech in {test_file}...")
        outputs = self.asr_model.transcribe(test_file, word_timestamps=True)
        asr_result = []

        # for debugging
        # import json
        # with open('tmp.txt', 'w') as f:
        #     f.write(json.dumps(outputs))

        asr_text = outputs["text"]
        seg_list = []
        for seg in outputs["segments"]:
            seg_list.append(
                {
                    "text": seg["text"],
                    "start_time": seg["start"],
                    "end_time": seg["end"],
                }
            )
            for word in seg["words"]:
                asr_result.append(
                    {
                        "text": word["word"],
                        "start_time": word["start"],
                        "end_time": word["end"],
                    }
                )
        total_words = len(asr_result)
        if rm_tag == 1:
            os.remove(test_file)
        logger.info(f"speech recognition done: {test_file}")

        zh_ratio = calculate_chinese_chars(asr_text)[2]
        is_zh = zh_ratio > self.threshold_zh
        if is_zh:
            logger.info("Chinese Voice")
        else:
            logger.info("English Voice")
            doc = self.sent_seg_en(asr_text)
            sentences = [sent.text for sent in doc.sents]

        # 结合分割好的句子，聚合词级时间戳，生成句子级时间戳
        sentence_timestamps = []
        sentence_start_idx = 0

        if is_zh:
            # 中文部分直接使用whisper分割好的句子
            sentence_timestamps = seg_list
        else:
            for sentence in sentences:
                # words_in_sentence = re.split(r'[ ,\-\.]+', sentence.strip(' .')) # hard to define the splitting rule
                # num_words = len(words_in_sentence)
                l_sent = len(sentence.replace(" ", ""))
                accu_len = 0
                num_words = total_words - sentence_start_idx
                for ia in range(sentence_start_idx, total_words):
                    accu_len += len(asr_result[ia]["text"].replace(" ", ""))
                    if accu_len >= l_sent:
                        num_words = ia - sentence_start_idx + 1
                        break

                cur_st = 0
                for iw in range(1, num_words + 1):
                    # 拒绝间隔过久的两个词被组合在一个句子中
                    if (
                        iw == num_words
                        or asr_result[sentence_start_idx + iw]["start_time"]
                        - asr_result[sentence_start_idx + iw - 1]["end_time"]
                        > self.max_time_diff
                    ):
                        # truncate
                        sentence_timestamps.append(
                            {
                                "text": "".join(
                                    [
                                        w["text"]
                                        for w in asr_result[
                                            sentence_start_idx
                                            + cur_st : sentence_start_idx
                                            + iw
                                        ]
                                    ]
                                ).strip(),
                                "start_time": asr_result[sentence_start_idx + cur_st][
                                    "start_time"
                                ],
                                "end_time": asr_result[sentence_start_idx + iw - 1][
                                    "end_time"
                                ],
                            }
                        )
                        cur_st = iw

                sentence_start_idx += num_words

        logger.info(sentence_timestamps)
        ret = {
            "paragraph": asr_text,
            "sentences": sentence_timestamps,
            "words": asr_result,
        }
        return ret


if __name__ == "__main__":
    # 中文测试
    # test_file = "video/genius_dev_default_16661ac9-8bf6-4aaa-a4d8-2881844f1f6f.mp4"
    # 英文测试
    test_file = "~/Downloads/飞书20241104-152849.mp4"

    asr = ASRPipeline()
    asr_result = asr(test_file)
    print(asr_result)
