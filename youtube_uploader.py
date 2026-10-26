import os
import logging
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Обновили права: добавили force-ssl для возможности оставлять комментарии
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl"
]

def get_youtube_service():
    """Создает авторизованный клиент YouTube API из переменных окружения."""
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")

    if not all([refresh_token, client_id, client_secret]):
        raise ValueError("Не заданы YOUTUBE_REFRESH_TOKEN / YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET в Secrets")

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)

def upload_video(video_path: str, title: str, description: str, tags: list[str]) -> str | None:
    """Загружает видео на YouTube."""
    youtube = get_youtube_service()

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": "25",  # News & Politics
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Загрузка видео: {int(status.progress() * 100)}%")
        except HttpError as e:
            error_content = e.content.decode("utf-8") if e.content else ""
            if "quotaExceeded" in error_content or e.resp.status == 403:
                logger.warning("❌ Квота YouTube API исчерпана. Пропускаем запуск.")
                return None
            raise

    video_id = response.get("id")
    logger.info(f"✅ Видео загружено: https://youtu.be/{video_id}")
    return video_id

def post_comment(video_id: str, comment_text: str) -> bool:
    """Публикует комментарий под видео."""
    try:
        youtube = get_youtube_service()
        body = {
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": comment_text
                    }
                }
            }
        }
        response = youtube.commentThreads().insert(
            part="snippet",
            body=body
        ).execute()
        logger.info(f"✅ Комментарий успешно добавлен.")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при добавлении комментария (возможно, спам-фильтр YouTube): {e}")
        return False

def build_description(news_text: str, hashtags: str, tg_link: str, max_link: str) -> str:
    """Формирует описание с хештегами и ссылками на соцсети."""
    social = f"\n\n---\n📢 Наш Telegram: {tg_link}\n📢 MAX: {max_link}"
    return news_text.strip() + "\n\n" + hashtags + social
