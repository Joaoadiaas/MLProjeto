# Assistente de IA no Telegram (n8n)

Automacao no-code/low-code com **n8n** que transforma mensagens do Telegram em respostas
geradas por uma LLM. Projeto de estudo e portfolio, rodando 100% local com custo zero.

## Arquitetura

```
Telegram (usuario)
      | mensagem
      v
[Telegram Trigger]  --webhook-->  n8n
      v
[Groq - Chat Completion]  --HTTP-->  API da LLM (Llama 3.3 70B)
      v
[Enviar Resposta]  -->  Telegram (usuario)
```

## Stack

- **n8n** (self-hosted via Docker) - orquestracao
- **Groq** (free tier) - inferencia da LLM
- **Telegram Bot API** - interface com o usuario

## Como rodar

Veja o passo a passo completo em [`GUIA.md`](./GUIA.md).

Resumo:
1. `docker compose up -d`
2. Abrir http://localhost:5678
3. Importar `workflow_telegram_ia.json`
4. Configurar credenciais (Telegram + Groq)
5. Ativar e testar

## Estrutura

| Arquivo | Descricao |
|---|---|
| `docker-compose.yml` | Sobe o n8n localmente |
| `workflow_telegram_ia.json` | O fluxo de automacao (importavel no n8n) |
| `GUIA.md` | Passo a passo detalhado |
