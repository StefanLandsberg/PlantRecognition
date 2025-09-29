@echo off
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=ZA/ST=GP/L=JHB/O=PlantApp/CN=192.168.101.251"
echo Certificates generated successfully!