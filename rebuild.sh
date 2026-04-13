#!/bin/bash
cd /root/TradingCons
echo "Остановка..."
docker-compose down
echo "Очистка мусора..."
docker system prune -f
echo "Сборка..."
docker-compose up --build -d
sleep 15
docker ps --format "table {{.Names}}\t{{.Status}}"
docker logs tradebot_new --tail 10 | grep -v "health\|werkzeug"
