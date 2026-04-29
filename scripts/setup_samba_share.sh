#!/usr/bin/env bash
# Расшарить /home/ros2/DuckSidian/incoming по SMB для перелива файлов с Windows.
# Запуск: bash scripts/setup_samba_share.sh
set -euo pipefail

SHARE_DIR="/home/ros2/DuckSidian/incoming"
SHARE_NAME="ducksidian"
SMB_USER="${SUDO_USER:-$USER}"

echo "[1/5] Создаю папку $SHARE_DIR"
mkdir -p "$SHARE_DIR"
chown "$SMB_USER":"$SMB_USER" "$SHARE_DIR"
chmod 0775 "$SHARE_DIR"

echo "[2/5] Ставлю samba (нужен sudo)"
sudo apt-get update -qq
sudo apt-get install -y samba

echo "[3/5] Прописываю шару [$SHARE_NAME] в /etc/samba/smb.conf"
SMB_CONF="/etc/samba/smb.conf"
if ! sudo grep -q "^\[$SHARE_NAME\]" "$SMB_CONF"; then
  sudo tee -a "$SMB_CONF" >/dev/null <<EOF

[$SHARE_NAME]
   comment = DuckSidian incoming dropzone
   path = $SHARE_DIR
   browseable = yes
   read only = no
   writable = yes
   guest ok = no
   create mask = 0664
   directory mask = 0775
   valid users = $SMB_USER
   force user = $SMB_USER
EOF
  echo "  -> добавлено"
else
  echo "  -> уже есть, пропускаю"
fi

echo "[4/5] Устанавливаю Samba-пароль для пользователя $SMB_USER"
echo "     (нажми Enter, потом вводи пароль 2 раза — это пароль ИМЕННО для SMB,"
echo "      может совпадать с системным или быть любым другим)"
sudo smbpasswd -a "$SMB_USER"
sudo smbpasswd -e "$SMB_USER" >/dev/null

echo "[5/5] Перезапускаю smbd и открываю порты в ufw (если активен)"
sudo systemctl restart smbd nmbd
sudo systemctl enable smbd nmbd >/dev/null 2>&1 || true
if sudo ufw status 2>/dev/null | grep -q "Status: active"; then
  sudo ufw allow from 192.168.0.0/16 to any app Samba || sudo ufw allow Samba || true
fi

IP=$(ip -4 addr show | awk '/inet 192\.168\./{print $2}' | cut -d/ -f1 | head -1)
echo
echo "==============================================="
echo "Готово. С Windows подключайся так:"
echo "  В проводнике в адресной строке:"
echo "      \\\\${IP:-192.168.1.125}\\$SHARE_NAME"
echo "  Логин: $SMB_USER"
echo "  Пароль: тот, что ты задал на шаге [4/5]"
echo
echo "Локальный путь, куда будут падать файлы:"
echo "  $SHARE_DIR"
echo "==============================================="
