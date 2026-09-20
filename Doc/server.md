# Server Configuration

## Infrastructure

| Item         | Detail                  |
|--------------|-------------------------|
| Cloud        | AWS EC2                 |
| OS           | Amazon Linux 2023       |
| Package Mgr  | dnf                     |
| SSH User     | ec2-user                |
| Domain       | horizonyhj.com          |
| DNS / CDN    | Cloudflare              |
| SSL          | Let's Encrypt (Certbot) |

---

## Directory Layout

```
/home/ec2-user/
├── venv/                  # Python 虚拟环境（项目外一级）
│   └── bin/gunicorn
└── Horisation/            # 项目根目录
    ├── app.py
    ├── requirements.txt
    ├── Backend/
    ├── Static/
    └── Template/
```

---

## Traffic Architecture

```
Browser
  → Cloudflare (DNS + SSL)
  → Nginx port 443 (HTTPS)
      /static/  →  直接返回 /home/ec2-user/Horisation/Static/
      /         →  proxy_pass http://127.0.0.1:8000
  → Gunicorn 127.0.0.1:8000
  → Flask app:app
```

---

## Nginx

Config file: `/etc/nginx/conf.d/horizonyhj.com.conf`

```nginx
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name horizonyhj.com www.horizonyhj.com;

    ssl_certificate /etc/letsencrypt/live/horizonyhj.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/horizonyhj.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location /static/ {
        alias /home/ec2-user/Horisation/Static/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    client_max_body_size 100M;
}

server {
    listen 80;
    listen [::]:80;
    server_name horizonyhj.com www.horizonyhj.com;
    return 301 https://$host$request_uri;
}
```

Common commands:
```bash
sudo nginx -t                   # 检查配置语法
sudo systemctl reload nginx     # 重载配置（不中断连接）
sudo systemctl restart nginx    # 完全重启
```

---

## Gunicorn (systemd)

Service file: `/etc/systemd/system/horisation.service`

```ini
[Unit]
Description=Horisation Flask App
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/Horisation
ExecStart=/home/ec2-user/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Common commands:
```bash
sudo systemctl start horisation      # 启动
sudo systemctl stop horisation       # 停止
sudo systemctl restart horisation    # 重启
sudo systemctl status horisation     # 查看状态
sudo systemctl enable horisation     # 开机自启
```

---

## Deploy / Update

```bash
cd /home/ec2-user/Horisation
git pull
sudo systemctl restart horisation
```

---

## Certificate renewal — this has failed once

On 2026-09-11 the Let's Encrypt certificate expired at 20:33 UTC and the site returned
**526** from Cloudflare (origin certificate invalid) until it was renewed by hand.
Certbot should have renewed it around 2026-08-12; the timer had been failing silently
for a month. Gunicorn and nginx were both up throughout — a green deploy tells you
nothing about this.

Diagnose from anywhere, no SSH needed:

```bash
echo | openssl s_client -connect 34.201.2.158:443 -servername horizonyhj.com 2>/dev/null \
  | openssl x509 -noout -dates
```

Renew on the server:

```bash
sudo certbot renew --force-renewal && sudo nginx -t && sudo systemctl reload nginx
sudo systemctl list-timers | grep -i certbot     # the timer must be active
```

The nginx config above answers port 80 with a `301` to HTTPS. Certbot's HTTP-01 challenge
follows that redirect, so it can still succeed as long as 443 serves
`/.well-known/acme-challenge/` — but if renewal ever fails, read
`journalctl -u certbot` rather than assuming; the cause of the August 2026 silent failure
was never established.

Renewed by hand 2026-09-12. Current certificate: `notBefore Sep 12 15:27 2026`,
`notAfter Dec 11 15:27 2026`. If the timer is not fixed, **it goes down again on
2026-12-11**.

## R2 bucket CORS — required for "Export as image"

Market's export renders listings to a canvas with html2canvas (`useCORS: true`).
Drawing a cross-origin image onto a canvas needs the image server to send
`Access-Control-Allow-Origin`; a plain `<img>` does not, which is why the Market
page showed photos while the export showed grey boxes (2026-09-20). The R2 public
bucket (`pub-a99b72ec….r2.dev`) ships with **no** CORS policy.

Set in Cloudflare → R2 → bucket → Settings → CORS Policy:

```json
[
  {
    "AllowedOrigins": ["https://horizonyhj.com", "http://localhost:5173"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": [],
    "MaxAgeSeconds": 86400
  }
]
```

Verify from anywhere (any listing image URL will do):

```bash
curl -sI -H "Origin: https://horizonyhj.com" \
  "https://pub-a99b72ec2d5f4c96b1891a7dafc657c1.r2.dev/listings/<id>/<file>.jpg" \
  | grep -i access-control-allow-origin
```

The CORS policy alone was not enough. The Market page plain-loads every photo
first, and Chrome caches that response — fetched without an Origin, so carrying no
Access-Control-Allow-Origin — under the bare URL; a later CORS request for the same
URL is served from that entry and rejected. So the export requests each photo as
`…jpg?export=1` with `crossOrigin="anonymous"` (`exportSrc()` in Market.jsx): a
distinct cache entry, CORS from the first byte. Both halves are required.

If the site's domain changes, or the bucket is replaced, this must be redone —
nothing in the repo can do it. Users who exported before the fix may need one
hard refresh, to drop cached image responses that carry no CORS header.

## AI provider key — the tarot reading needs one

The AI reading (`POST /api/tarot/reading`) calls DeepSeek over HTTPS from the server. The key is
**never in git**; it lives in one of two places, checked in this order:

1. Environment — `DEEPSEEK_API_KEY` (and optionally `AI_PROVIDER`, `AI_MODEL`). In the systemd unit:
   ```ini
   [Service]
   Environment=DEEPSEEK_API_KEY=sk-...
   ```
   then `sudo systemctl daemon-reload && sudo systemctl restart horisation`.
2. File — `/home/ec2-user/Horisation/Key/ai_config.json`, same folder as `r2_config.json`:
   ```bash
   cp ai_config.example.json Key/ai_config.json && nano Key/ai_config.json   # fill api_key
   sudo systemctl restart horisation
   ```

Without either, every `/reading` answers 503 with `error_kind: config` and a Chinese message; the
rest of the tarot page, and the rest of the site, are unaffected.

**Kill switch:** `AI_ENABLED=0` in the unit's `Environment=` turns the feature off (503 `disabled`)
without touching the key — for a leaked key or a runaway bill. The site-wide cap of 100 ok calls
per 24h (`Backend/Service/ai/quota.py`) is the automatic version of the same thing.

**What to look at after the first real reading:** `sqlite3 _data/market.db "select model, ok,
error_kind, latency_ms, prompt_tokens, completion_tokens, cost_usd from ai_usage order by
created_at desc limit 5"`. Latency should be 3–12 s; DeepSeek can be slower at US evening hours
(China daytime), which is what the 30 s read timeout and the single retry are for.

---

## Cloudflare SSL Mode

Set to **Full** or **Full (strict)**
- Cloudflare ↔ Origin: HTTPS
- Only ports 80 and 443 open publicly
- SSH restricted to specific IP
- Gunicorn not exposed to internet
