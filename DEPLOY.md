# Запуск и деплой

Инструкция по запуску (раздел 8 ТЗ). Сервер — ноутбук с NVIDIA RTX 4050 (6 ГБ), Windows, PowerShell.
Публичный адрес — постоянный домен ngrok. Всё работает локально; интернет нужен только туннелю ngrok
и один раз — чтобы скачать веса детектора.

Архитектура и модели — `docs/DOCUMENTATION.md`, раздел 2–3.

## 1. Установка (один раз)

Нужны: Python 3.11–3.12, ffmpeg в PATH (`ffmpeg -version`), драйвер NVIDIA, ngrok.

```powershell
cd путь\к\vortex_123
python -m venv .venv
.venv\Scripts\Activate.ps1            # если ругается: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # должно быть True
```

**Веса детектора** `PekingU/rtdetr_v2_r18vd` скачиваются с Hugging Face при первом запуске сервиса
**без** офлайн-переменных (раздел 2, но без строк `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`).
Дальше они лежат в кэше `%USERPROFILE%\.cache\huggingface` и интернет модели не нужен.

**ngrok:** скачать с ngrok.com, распаковать `ngrok.exe`, один раз:
```powershell
ngrok config add-authtoken <токен из кабинета ngrok>
```
Постоянный домен — в кабинете ngrok → Domains (`<имя>.ngrok-free.app`).

## 2. Запуск (два окна PowerShell)

**Окно 1 — сервис:**
```powershell
cd путь\к\vortex_123
.venv\Scripts\Activate.ps1
$env:API_KEY = "<ключ жюри>"
$env:HF_HUB_OFFLINE = "1"            # не ходить в интернет за моделями
$env:TRANSFORMERS_OFFLINE = "1"
$env:CHECKS_NOVLM = "1"              # финальный режим: детектор + правила, без VLM (это и так по умолчанию)
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
```
Ждать в логе строки `модели прогреты за … с: RT-DETR` и `анализатор готов`.
**Не** добавлять `--reload` и `--workers`: процесс анализа с моделью должен быть один.

**Окно 2 — публичный адрес:**
```powershell
ngrok http --url=<имя>.ngrok-free.app 8000
```

Проверка: http://localhost:8000/health → `"analyzer_ready": true`; затем `https://<имя>.ngrok-free.app/` с телефона
через мобильный интернет.

Ключ генерируется так: `python -c "import secrets; print(secrets.token_urlsafe(24))"`. В git его не кладём.

## 3. Что отдаётся жюри

| | |
|---|---|
| Интерфейс | `https://<имя>.ngrok-free.app/` |
| Загрузка видео | `POST https://<имя>.ngrok-free.app/api/v1/analyze` (multipart, поле `file`) |
| Результат | `GET https://<имя>.ngrok-free.app/api/v1/jobs/{job_id}` |
| Ключ | заголовок `X-API-Key: <ключ>` |

```bash
curl -X POST https://<имя>.ngrok-free.app/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
curl https://<имя>.ngrok-free.app/api/v1/jobs/<job_id> -H "X-API-Key: <ключ>"
```
ngrok в **браузере** один раз показывает страницу-предупреждение («Visit Site»); запросы API из скриптов
и curl она не затрагивает.

## 4. Проверка перед сдачей

```powershell
# как жюри: 5 видео одновременно через ПУБЛИЧНЫЙ адрес
python scripts/jury_sim.py --url https://<имя>.ngrok-free.app --key <ключ> --videos data\videos --limit 5 --parallel 5
```
Цель: `Запрос без ключа -> 401 OK`, `Не засчитаются у жюри: 0`, максимальное время заметно меньше 300 с.

Сценарий раздела 4 ТЗ в интерфейсе — на 1–2 видео с нарушением: статус обработки → вердикт →
у каждого нарушения тип, таймкод, 1–3 кадра, описание и «Источник: …, пункт …».

Офлайн: выключить Wi-Fi (ngrok при этом тоже отключится), прогнать видео через http://localhost:8000 — анализ должен работать.

## 5. Пока идёт проверка жюри

- Зарядка подключена; Параметры → Питание: сон «Никогда», закрытие крышки «Ничего не делать».
- Обновления Windows на паузе, лишние программы закрыты (оперативка и видеопамять нужны модели).
- Оба окна (сервис и ngrok) открыты **до объявления результатов**, не только до дедлайна.
- Если сервис упал — просто запустить окно 1 заново: задания хранятся в `var/jobs.db`,
  незавершённые снова встанут в очередь. Если упал ngrok — запустить окно 2 заново, домен тот же.

## 6. Как устроен сервис

```
POST /api/v1/analyze ─► проверка ключа ─► файл в var/uploads, запись в var/jobs.db (queued) ─► очередь
                                                                                              │
Supervisor (поток) берёт задания по одному ─► отдельный процесс с моделью: analyze(path, deadline)
   • модель грузится при старте (прогрев), а не на первом видео жюри
   • deadline: время до 230 с после загрузки делится поровну между видео в очереди (не больше 75 с на одно)
   • результат проверяется по схеме раздела 10 ТЗ и пишется в jobs.db (done)
   • зависание > 270 с или падение процесса ─► failed, процесс анализа перезапускается, очередь идёт дальше
   • видео удаляется сразу после обработки
GET /api/v1/jobs/{id} ─► processing | done (+ verdict, violations) | failed
Интерфейс (/) ходит в /ui/*, поэтому ключ жюри не попадает в браузер.
```

## 7. Переменные окружения

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `API_KEY` | случайный при запуске | ключ для `X-API-Key` (обязательно задать) |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | — | `1` = не обращаться к Hugging Face (после того как веса скачаны) |
| `CHECKS_NOVLM` | `1` | `1` = без VLM (финальный режим); `0` = прицельные вопросы VLM Qwen3-VL-2B |
| `PIPELINE` | `checks` | `checks` — детектор людей + правила; `windows` — старый режим VLM по окнам |
| `PEOPLE_DET` | `1` | `0` = выключить детектор людей |
| `SOFT_DEADLINE_SEC` | 230 | к какому моменту после загрузки видео должно быть готово |
| `MAX_RUN_SEC` | 75 | максимум на одно видео, даже если очередь пуста |
| `MIN_RUN_SEC` | 20 | минимум на анализ, даже если видео долго ждало |
| `RUN_TIMEOUT_SEC` | 270 | жёсткий лимит одного анализа, дальше перезапуск процесса анализа |
| `DATA_DIR` | `var` | где лежат `jobs.db` и временные загрузки |
| `MAX_UPLOAD_MB` | 500 | максимальный размер файла |
| `SKIP_WARMUP` | — | `1` = не грузить модели при старте |
| `STUB_ALWAYS_VIOLATION` | — | `1` = фейковое нарушение без моделей (отладка интерфейса) |

## 8. Docker (перенос во внутренний контур банка)

```powershell
docker build -t vortex .
docker run --gpus all -p 8000:8000 -e API_KEY=<ключ> -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  -v $env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface vortex
```
Кэш с весами монтируется томом, поэтому контейнеру интернет не нужен. На хакатоне сервис запущен без Docker (раздел 2);
сборка образа на хакатоне не проверялась.
