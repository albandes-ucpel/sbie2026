# 🐳 Tutorial: Instalação do Docker + Qdrant Server no Amazon Linux 2023

Este guia mostra como instalar o **Docker** em uma instância **Amazon Linux 2023**, subir o **Qdrant Server** em container e testar a instalação.

---

## 1) Atualizar pacotes
Conecte-se na sua EC2 e execute:

```bash
sudo yum update -y
```

---

## 2) Instalar o Docker
No Amazon Linux 2023 o Docker já está no repositório padrão:

```bash
sudo yum install -y docker
```

---

## 3) Iniciar e habilitar o serviço Docker
```bash
sudo systemctl enable --now docker
```

Verifique se o serviço está ativo:

```bash
systemctl status docker
```

---

## 4) Adicionar seu usuário ao grupo `docker`
Por padrão, apenas `root` pode usar Docker. Para liberar seu usuário (ex.: `ec2-user`):

```bash
sudo usermod -aG docker $USER
```

Aplique a mudança no grupo atual:

```bash
newgrp docker
```

Teste a instalação:

```bash
docker run hello-world
```

Se aparecer a mensagem **Hello from Docker!**, a instalação funcionou ✅.

---

## 5) Subir o Qdrant Server
Crie um diretório para persistir os dados do Qdrant:

```bash
mkdir -p ~/qdrant_storage
```

Execute o container do Qdrant:

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

- `-d` → roda em background  
- `--name qdrant` → nome do container  
- `-p 6333:6333` → expõe a porta 6333 no host  
- `-v ~/qdrant_storage:/qdrant/storage` → volume persistente  

---

## 6) Testar o Qdrant Server
Verifique se o container está rodando:

```bash
docker ps
```

Você deve ver algo parecido com:

```
CONTAINER ID   IMAGE                  STATUS         PORTS                  NAMES
abcd1234       qdrant/qdrant:latest   Up 10 seconds  0.0.0.0:6333->6333/tcp qdrant
```

Teste o endpoint HTTP:

```bash
curl -s http://localhost:6333/collections | jq
```

Se aparecer:

```json
{
  "collections": []
}
```

significa que o Qdrant está pronto 🎉.

---

## 7) Configurar sua aplicação para usar o Qdrant
No arquivo `.env` da aplicação, configure:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=literatura
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
API_KEY=dev-key
```

---

## 8) Comandos úteis
- **Parar o Qdrant**:
  ```bash
  docker stop qdrant
  ```
- **Iniciar novamente**:
  ```bash
  docker start qdrant
  ```
- **Ver logs do container**:
  ```bash
  docker logs -f qdrant
  ```
- **Remover container**:
  ```bash
  docker rm -f qdrant
  ```

---

✅ Agora o Docker e o Qdrant Server estão prontos no seu Amazon Linux 2023!
