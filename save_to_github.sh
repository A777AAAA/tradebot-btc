#!/bin/bash
cd /root/TradingCons
echo "Текущий статус бота:"
docker ps --format "table {{.Names}}\t{{.Status}}"
echo ""
read -p "Бот работает нормально? Пушим на GitHub? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Отменено."
    exit 0
fi
git add -A
git commit -m "update $(date '+%d.%m.%Y %H:%M')"
git push origin main
echo "✅ Сохранено на GitHub!"
