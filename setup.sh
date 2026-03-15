#!/bin/bash
# =============================================================
# setup.sh — One-click nginx + HTTPS setup for mikastars39.site
# Usage: bash setup.sh
# =============================================================

set -e  # exit on any error

DOMAIN="mikastars39.site"
WWW_DOMAIN="www.mikastars39.site"
REPO="https://github.com/MikaStars39/mikastars39.site"
WEBROOT="/var/www/${DOMAIN}"
NGINX_CONF="/etc/nginx/sites-available/${DOMAIN}"
EMAIL="meloxdmy@qq.com"

# ---------- 颜色输出 ----------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $1"; }
warning() { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ---------- 检查 root ----------
[ "$EUID" -ne 0 ] && error "请用 root 运行：sudo bash setup.sh"

# ---------- 检查邮箱 ----------
[ -z "$EMAIL" ] && error "请先在脚本顶部填写 EMAIL 变量（用于 HTTPS 证书）"

# ============================================================
# 1. 安装依赖
# ============================================================
info "更新软件包列表..."
apt-get update -qq

info "安装 nginx / git / certbot..."
apt-get install -y -qq nginx git certbot python3-certbot-nginx

# ============================================================
# 2. Clone 或更新代码
# ============================================================
if [ -d "${WEBROOT}/.git" ]; then
    info "仓库已存在，强制同步远端..."
    git -C "${WEBROOT}" fetch origin
    git -C "${WEBROOT}" reset --hard origin/main
else
    info "克隆仓库到 ${WEBROOT}..."
    git clone "${REPO}" "${WEBROOT}"
fi

chown -R www-data:www-data "${WEBROOT}"
chmod -R 755 "${WEBROOT}"

# ============================================================
# 3. 写入 nginx 配置（HTTP，先让 certbot 能验证域名）
# ============================================================
info "写入 nginx 配置..."
cat > "${NGINX_CONF}" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN} ${WWW_DOMAIN};

    root ${WEBROOT};
    index index.html;

    # HTML 不缓存
    location ~* \.html$ {
        add_header Cache-Control "no-cache, must-revalidate";
        expires 0;
        try_files \$uri \$uri/ =404;
    }

    # JS / CSS / 图片 / 字体 长期缓存
    location ~* \.(js|css|png|jpg|jpeg|svg|woff|woff2|ico)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        expires 1y;
    }

    # 其余请求
    location / {
        try_files \$uri \$uri/ =404;
    }
}
EOF

# 清理所有已启用的站点配置（避免 server_name 冲突）
rm -f /etc/nginx/sites-enabled/*
# 启用本站点
ln -sf "${NGINX_CONF}" /etc/nginx/sites-enabled/

nginx -t || error "nginx 配置有误，请检查 ${NGINX_CONF}"
# nginx 可能还没启动，用 restart 兼容两种情况
systemctl restart nginx
info "nginx 配置完成（HTTP）"

# ============================================================
# 4. 申请 HTTPS 证书（Let's Encrypt）
# ============================================================
info "申请 HTTPS 证书..."
certbot --nginx \
    -d "${DOMAIN}" \
    -d "${WWW_DOMAIN}" \
    --non-interactive \
    --agree-tos \
    --email "${EMAIL}" \
    --redirect   # 自动把 HTTP 重定向到 HTTPS

info "HTTPS 证书申请完成"

# ============================================================
# 5. 设置证书自动续期
# ============================================================
info "设置证书自动续期..."
# certbot 安装时已在 /etc/cron.d/certbot 或 systemd timer 里注册
# 下面确认 timer 已启用
if systemctl is-active --quiet certbot.timer 2>/dev/null; then
    info "certbot.timer 已在运行"
elif crontab -l 2>/dev/null | grep -q certbot; then
    info "certbot cron 已配置"
else
    # 手动加一条 cron
    (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && systemctl reload nginx") | crontab -
    info "已添加每日凌晨3点自动续期任务"
fi

# ============================================================
# 6. 启动并开机自启
# ============================================================
systemctl enable nginx
systemctl start nginx

# ============================================================
# 完成
# ============================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}  https://${DOMAIN}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "后续更新网站只需在服务器上运行："
echo "  git -C ${WEBROOT} pull && systemctl reload nginx"
