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

Note the nginx config above listens on **443 only**. Certbot's HTTP-01 challenge needs
port 80 to answer at `/.well-known/acme-challenge/`; if renewal fails with a connection
error on 80, that is why.

Renewed by hand 2026-09-12. Current certificate: `notBefore Sep 12 15:27 2026`,
`notAfter Dec 11 15:27 2026`. If the timer is not fixed, **it goes down again on
2026-12-11**.

## Cloudflare SSL Mode

Set to **Full** or **Full (strict)**
- Cloudflare ↔ Origin: HTTPS
- Only ports 80 and 443 open publicly
- SSH restricted to specific IP
- Gunicorn not exposed to internet
