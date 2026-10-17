# Запуск и деплой (Чел 2)

Целевая машина: ноутбук с RTX 4050 (6 ГБ), Windows. Всё ниже — для PowerShell.

## 1. Один раз: установка

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg            # после установки ПЕРЕОТКРЫТЬ терминал
winget install Cloudflare.cloudflared
ffmpeg -version                       # проверка, что ffmpeg виден

cd путь\к\vortex_123
python -m venv .venv
.venv\Scripts\Activate.ps1            # если ругается: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

## 2. Ключ для жюри

```powershell
python -c "import secrets; print(secrets.token_urlsafe(24))"
```
Сохранить ключ у себя (не в git). Каждый раз перед запуском:
```powershell
$env:API_KEY = "<ключ>"
```

## 3. Запуск сервиса

```powershell
$env:API_KEY = "<ключ>"
uvicorn service.app:app --host 0.0.0.0 --port 8000
```
- Интерфейс: http://localhost:8000/
- Состояние: http://localhost:8000/health (`analyzer_ready: true` — модели загружены)
- **Не** добавлять `--reload` и `--workers`: воркер с моделями должен быть один.
- Для проверки фронта без моделей: `$env:STUB_ALWAYS_VIOLATION = "1"` перед запуском.

## 4. Публичный адрес

### Быстрая проверка (адрес меняется при каждом перезапуске)
Во втором окне терминала:
```powershell
cloudflared tunnel --url http://localhost:8000
```
В выводе будет `https://<случайные-слова>.trycloudflare.com`. Годится для тестов внутри команды.
**Жюри такой адрес не отдавать**: если туннель перезапустится, адрес сменится.

### Постоянный адрес для жюри
Вариант А — именованный туннель Cloudflare (нужен домен, добавленный в бесплатный аккаунт Cloudflare):
```powershell
cloudflared tunnel login
cloudflared tunnel create vortex
cloudflared tunnel route dns vortex vortex.<ваш-домен>
cloudflared tunnel run --url http://localhost:8000 vortex
```
Вариант Б — нет домена: ngrok, у бесплатного аккаунта есть один постоянный домен
(`ngrok http --url=<ваш-домен>.ngrok-free.app 8000`). API через него работает; в браузере
ngrok один раз показывает страницу-предупреждение.

Проверка снаружи (с телефона без Wi-Fi или с другого компьютера): открыть адрес, загрузить видео.

## 5. Подготовка ноутбука-сервера

- Зарядка подключена все время.
- Параметры → Система → Питание: сон — «Никогда»; закрытие крышки — «Ничего не делать».
- Приостановить обновления Windows (Центр обновления → Приостановить на 1 неделю).
- Закрыть браузеры и тяжёлые программы: оперативки 16 ГБ, она нужна моделям.
- Держать окна `uvicorn` и `cloudflared` открытыми **до объявления результатов**, а не только до дедлайна.

## 6. Проверка перед сдачей

```powershell
# 1) как жюри: все видео одновременно через ПУБЛИЧНЫЙ адрес
python scripts/jury_sim.py --url https://<адрес> --key <ключ> --videos data/videos --labels data/labels_clean.csv --split train --parallel 5
# 2) метрики по этим ответам
python scripts/evaluate.py --labels data/labels_clean.csv --preds preds_api/ --split train
```
Цель: «Не засчитаются у жюри: 0», максимальное время заметно меньше 300 с.

Проверка локальности: после того как модели один раз скачаны — отключить Wi-Fi,
перезапустить сервис, прогнать видео через http://localhost:8000. Должно работать.
(Для моделей с Hugging Face: `$env:HF_HUB_OFFLINE = "1"`.)

## 7. Что отдать жюри

- Интерфейс: `https://<адрес>/`
- API: `https://<адрес>/api/v1/analyze`, `https://<адрес>/api/v1/jobs/{job_id}`
- Ключ: заголовок `X-API-Key: <ключ>`

## Настройки (переменные окружения)

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `API_KEY` | случайный при запуске | ключ для `X-API-Key` |
| `DATA_DIR` | `var` | где лежат `jobs.db` и временные загрузки |
| `SOFT_DEADLINE_SEC` | 230 | к какому моменту после загрузки видео должно быть готово; время делится поровну между видео в очереди |
| `RUN_TIMEOUT_SEC` | 270 | жёсткий лимит одного анализа; дальше процесс анализа убивается и перезапускается |
| `MIN_RUN_SEC` | 20 | минимум времени на анализ, даже если видео долго ждало в очереди |
| `MAX_RUN_SEC` | 75 | максимум времени на одно видео, даже если очередь пуста |
| `SKIP_WARMUP` | — | `1` = не грузить модели при старте |
| `MAX_UPLOAD_MB` | 500 | максимальный размер файла |
| `ANALYSIS_FPS` | 2 | сколько кадров в секунду берёт пайплайн |
| `STUB_ALWAYS_VIOLATION` | — | `1` = фейковое нарушение (отладка без моделей) |

## Docker (для переносимости в контур банка)

```powershell
docker build -t vortex .
docker run --gpus all -p 8000:8000 -e API_KEY=<ключ> -v ${PWD}/weights:/app/weights vortex
```
На Windows нужен Docker Desktop с WSL2; `--gpus all` даёт контейнеру доступ к RTX.
На хакатоне проще запускать без Docker (раздел 3), а Docker показать как способ переноса.

## Как устроено внутри

```
POST /api/v1/analyze ─► файл в var/uploads, запись в var/jobs.db (queued) ─► очередь
                                                                             │
Supervisor (поток) берёт задания по одному ─► отдельный процесс с моделями: analyze(path, deadline)
   • результат проверяется по схеме ТЗ и пишется в jobs.db (done)
   • зависание > RUN_TIMEOUT_SEC или падение процесса ─► failed, процесс перезапускается
   • видео удаляется сразу после обработки
GET /api/v1/jobs/{id} ─► читает jobs.db: processing | done (+результат) | failed
```
После перезапуска сервиса незавершённые задания с сохранившимся файлом снова встают в очередь.
