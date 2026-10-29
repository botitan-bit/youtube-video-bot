import os
import re
import json
import hashlib
import logging
import subprocess
import tempfile
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
from groq import Groq
from faster_whisper import WhisperModel

from youtube_uploader import upload_video, build_description, post_comment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============ КОНФИГУРАЦИЯ ============
SEARCH_WINDOW_HOURS = 18
# ОБНОВЛЕННЫЙ СПИСОК КАНАЛОВ (убраны ragulaku и pilachzx)
CHANNELS = ["novosti_efir", "mnogonazi", "botcharov", "slavaded1337"]

BLACKLIST_WORDS = [
    "животн", "кошк", "собак", "природ", "зоопарк", "питомц",
    "спорт", "футбол", "хоккей", "матч", "чемпионат", "олимпиад",
    "реклам", "промокод", "скидк", "купи", "закажи",
    "гороскоп", "совещан", "визит", "юбилей", "поздрав",
    "трамп", "байден", "зеленск",
    "шоу-бизнес", "звезд", "мем", "бонсмэшинг", "зумер", "тикток",
    "айфон", "iphone",
]
GROQ_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "small"
TELEGRAM_LINK = "https://t.me/pervonovosti"
MAX_LINK = "https://max.ru/channel_pervonovosti"

STATE_DIR = Path(".")
POSTED_HASHES_FILE = STATE_DIR / "posted_hashes.txt"
POSTED_NEWS_FILE = STATE_DIR / "posted_news.txt"
POSTED_VIDEOS_FILE = STATE_DIR / "posted_videos.txt"
PUBLISHED_TITLES_FILE = STATE_DIR / "published_titles.txt"

# ============ МОДЕЛЬ ДАННЫХ ============
@dataclass
class Candidate:
    channel: str
    post_url: str
    text: str
    video_url: str
    published_at: datetime
    video_width: int = 0
    video_height: int = 0

# ============ УТИЛИТЫ ============
def load_lines(path: Path, max_lines: int | None = None) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-max_lines:] if max_lines else lines

def append_line(path: Path, line: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def text_fingerprint(text: str) -> str:
    words = re.findall(r"\w{5,}", text.lower())
    stemmed = [re.sub(r"(ами|ями|ов|ев|ый|ая|ое|ые|ить|ать|ует|ают)$", "", w) for w in words]
    return hashlib.sha256(" ".join(sorted(set(stemmed))).encode()).hexdigest()

def jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def is_duplicate(text: str) -> bool:
    new_fp = text_fingerprint(text)
    recent = load_lines(POSTED_HASHES_FILE, max_lines=150)
    for old_fp in recent:
        if jaccard_similarity(new_fp, old_fp) > 0.4:
            return True
    return False

def has_blacklisted_words(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in BLACKLIST_WORDS)

def parse_telegram_time(time_str: str) -> datetime:
    return datetime.fromisoformat(time_str)

# ============ СБОР ИЗ TELEGRAM ============
def fetch_channel_posts(channel: str) -> list[dict]:
    url = f"https://t.me/s/{channel}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Не удалось получить канал {channel}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    posts = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=SEARCH_WINDOW_HOURS)

    for msg in soup.select(".tgme_widget_message"):
        try:
            time_el = msg.select_one("time[datetime]")
            if not time_el:
                continue
            published_at = parse_telegram_time(time_el["datetime"])
            if published_at < cutoff:
                continue

            text_el = msg.select_one(".tgme_widget_message_text")
            text = text_el.get_text(" ", strip=True) if text_el else ""

            video_el = msg.select_one("video")
            if not video_el:
                continue
            video_url = video_el.get("src")
            if not video_url:
                continue

            post_link_el = msg.select_one("a.tgme_widget_message_date")
            post_url = post_link_el["href"] if post_link_el else ""

            posts.append({
                "channel": channel,
                "post_url": post_url,
                "text": text,
                "video_url": video_url,
                "published_at": published_at,
            })
        except Exception as e:
            logger.debug(f"Ошибка парсинга поста в {channel}: {e}")
            continue
    return posts

def collect_candidates() -> list[Candidate]:
    all_posts = []
    for ch in CHANNELS:
        posts = fetch_channel_posts(ch)
        logger.info(f"{ch}: найдено {len(posts)} видео-постов за {SEARCH_WINDOW_HOURS}ч")
        all_posts.extend(posts)

    candidates = []
    for p in all_posts:
        if not p["text"] or has_blacklisted_words(p["text"]):
            continue
        if is_duplicate(p["text"]):
            logger.info(f"Дубликат: {p['post_url']}")
            continue
        if p["post_url"] in load_lines(POSTED_NEWS_FILE):
            continue
        candidates.append(Candidate(**p))

    candidates.sort(key=lambda c: c.published_at, reverse=True)
    return candidates

# ============ ВЫБОР И ФОРМАТИРОВАНИЕ ЧЕРЕЗ GROQ ============
def select_best_candidate(candidates: list[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    top5 = candidates[:5]
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = (
        "Ты — редактор жёстких новостей. Выбери ОДИН самый важный и жёсткий сюжет из списка. "
        "Приоритет: СВО, криминал, ДТП, пожары, ЧП. "
        "ЗАПРЕЩЕНО выбирать: спорт, животных, шоу-бизнес, мемы, рекламу, политические провокации. "
        "Верни ТОЛЬКО номер выбранного кандидата (1-5).\n\n"
    )
    for i, c in enumerate(top5, 1):
        prompt += f"{i}. [{c.channel}] {c.text[:300]}\n\n"
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip()
        match = re.search(r"([1-5])", answer)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(top5):
                logger.info(f"Groq выбрал кандидата #{idx + 1}: {top5[idx].post_url}")
                return top5[idx]
    except Exception as e:
        logger.error(f"Ошибка Groq при выборе: {e}")
    logger.warning("Groq не дал четкого ответа, берём первого кандидата.")
    return top5[0]

def format_with_groq(candidate: Candidate) -> tuple[str, str, str]:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = (
        "Перепиши новость для YouTube. СОХРАНИ все факты. Удали оскорбления и упоминания других каналов/ссылки на Telegram. "
        "Сократи до 600 символов. Без эмодзи и графических символов. "
        "Формат ответа СТРОГО:\n"
        "TITLE: <заголовок до 60 символов>\n"
        "TEXT: <текст>\n"
        "HASHTAGS: <3-5 ключевых тегов, отражающих суть новости, через пробел, начиная с #>\n\n"
        f"Исходный текст:\n{candidate.text}"
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = resp.choices[0].message.content
        title_match = re.search(r"TITLE:\s*(.+)", content)
        text_match = re.search(r"TEXT:\s*(.+?)(?:\nHASHTAGS:|$)", content, re.DOTALL)
        tags_match = re.search(r"HASHTAGS:\s*(.+)", content)
        
        title = title_match.group(1).strip()[:60] if title_match else candidate.text[:60]
        body = text_match.group(1).strip() if text_match else candidate.text[:600]
        hashtags = tags_match.group(1).strip() if tags_match else "#новости #срочно"
        return title, body, hashtags
    except Exception as e:
        logger.error(f"Ошибка Groq при форматировании: {e}")
        return candidate.text[:60], candidate.text[:600], "#новости #срочно"

# ============ СКАЧИВАНИЕ И ДЛИТЕЛЬНОСТЬ ============
def download_video(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Ошибка скачивания видео: {e}")
        return False

def get_duration(file_path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            text=True
        ).strip()
        return float(out)
    except Exception:
        return 0.0

# ============ УМНАЯ ПРОВЕРКА РЕЧИ ============
def has_human_speech(video_path: Path) -> bool:
    """Проверяет, есть ли в видео человеческая речь (а не просто музыка)."""
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(video_path), language="ru")
        words = "".join([seg.text for seg in segments]).strip()
        return len(words.split()) > 5
    except Exception as e:
        logger.error(f"Ошибка проверки речи: {e}")
        return True

# ============ ГЕНЕРАЦИЯ ТИТРОВ (БЕЗ ОЗВУЧКИ) ============
def generate_ass_from_text(text: str, video_duration: float, output_ass: Path):
    """Разбивает текст на титры и распределяет их равномерно по времени видео."""
    
    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Если предложение слишком длинное, режем по запятым
        if len(s) > 60:
            sub_parts = re.split(r'(?<=[,;])\s+', s)
            chunks.extend([part.strip() for part in sub_parts if part.strip()])
        else:
            chunks.append(s)
            
    total_chars = sum(len(c) for c in chunks)
    current_time = 0.0
    
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(output_ass, "w", encoding="utf-8") as f:
        f.write(ass_header)
        if total_chars == 0 or video_duration <= 0:
            return
            
        for chunk in chunks:
            # Вычисляем время на основе количества символов
            chunk_duration = (len(chunk) / total_chars) * video_duration
            end_time = current_time + chunk_duration
            
            start_str = f"{int(current_time//3600)}:{int((current_time%3600)//60):02d}:{int(current_time%60):02d}.{int((current_time%1)*100):02d}"
            end_str = f"{int(end_time//3600)}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}.{int((end_time%1)*100):02d}"
            
            chunk_clean = chunk.replace("\n", " ").strip()
            f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{chunk_clean}\n")
            current_time = end_time

# ============ СБОРКА ВИДЕО ============
def process_video(input_video: Path, ass_file: Path, output_path: Path, video_has_speech: bool) -> bool:
    if video_has_speech:
        logger.info("Видео содержит речь. Оставляем оригинал без изменений.")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video.absolute()),
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path.absolute()),
        ]
    else:
        logger.info("Видео без речи. Накладываем титры, оригинальный звук (если есть) оставляем.")
        ass_path_escaped = str(ass_file.absolute()).replace("\\", "/")
        video_filters = (
            f"[0:v]hqdn3d=1.5:1.5:6:6,unsharp=5:5:1.0,"
            f"ass='{ass_path_escaped}'[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video.absolute()),
            "-filter_complex", video_filters,
            "-map", "[outv]",
            "-map", "0:a?",  # Берем оригинальный звук, если он есть
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path.absolute()),
        ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка FFmpeg: {e.stderr.decode()[-1000:]}")
        return False

# ============ ОСНОВНОЙ ЦИКЛ ============
def main():
    logger.info("=== Запуск новостного бота ===")
    candidates = collect_candidates()
    if not candidates:
        logger.info("Нет свежих видео-кандидатов. Пропускаем запуск.")
        return
    chosen = select_best_candidate(candidates)
    if not chosen:
        logger.info("Не удалось выбрать кандидата.")
        return
    
    title, body, hashtags = format_with_groq(chosen)
    if not body:
        logger.warning("Пустой текст после форматирования, пропускаем.")
        return

    final_title = f"{title} {hashtags}"
    if len(final_title) > 100:
        final_title = final_title[:97] + "..."
    
    logger.info(f"Заголовок: {final_title}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_video = tmpdir / "raw.mp4"
        ass_file = tmpdir / "subs.ass"
        final_video = tmpdir / "final.mp4"

        if not download_video(chosen.video_url, raw_video):
            logger.error("Не удалось скачать видео.")
            return
        logger.info("Видео скачано.")

        video_has_speech = has_human_speech(raw_video)

        if not video_has_speech:
            video_dur = get_duration(raw_video)
            logger.info(f"Генерация титров для видео длительностью {video_dur:.1f} сек...")
            generate_ass_from_text(body, video_dur, ass_file)
            logger.info("Титры готовы.")
        else:
            ass_file.touch()

        logger.info("Сборка финального видео...")
        if not process_video(raw_video, ass_file, final_video, video_has_speech):
            logger.error("Не удалось обработать видео.")
            return

        description = build_description(body, hashtags, TELEGRAM_LINK, MAX_LINK)
        
        video_id = upload_video(str(final_video), final_title, description, tags=[])

        if video_id:
            comment_text = f"📢 Наш Telegram: {TELEGRAM_LINK}\n📢 MAX: {MAX_LINK}"
            post_comment(video_id, comment_text)

            append_line(POSTED_NEWS_FILE, chosen.post_url)
            append_line(POSTED_HASHES_FILE, text_fingerprint(chosen.text))
            append_line(POSTED_VIDEOS_FILE, hashlib.sha256(Path(final_video).read_bytes()).hexdigest())
            append_line(PUBLISHED_TITLES_FILE, final_title)
            logger.info(f"✅ Успешно опубликовано: {final_title} ({video_id})")
        else:
            logger.warning("Загрузка не удалась. Состояние не обновлено.")

if __name__ == "__main__":
    main()
