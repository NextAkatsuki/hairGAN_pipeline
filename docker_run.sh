docker build --tag hairgan_tensor:1.0 .

docker run --name hairgan_tensor -it -d -p 8000:8000 hairgan_tensor:1.0

docker logs hairgan_tensor