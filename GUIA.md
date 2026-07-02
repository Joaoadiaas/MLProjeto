# Guia: Assistente de IA no Telegram com n8n (custo zero)

Seu primeiro projeto de automacao com n8n. Ao final voce tera um bot de Telegram
que responde qualquer mensagem usando uma IA (LLM), tudo orquestrado no n8n rodando
na sua propria maquina. Zero custo.

**Arquivos deste projeto:**
- `docker-compose.yml` -> sobe o n8n localmente
- `workflow_telegram_ia.json` -> o fluxo pronto pra importar
- `GUIA.md` -> este passo a passo

Tempo estimado: 30-45 min na primeira vez.

---

## Pre-requisitos (instalar uma vez)

1. **Docker Desktop** -> https://www.docker.com/products/docker-desktop/
   Instale, abra e deixe rodando (o icone da baleia fica na barra de tarefas).
2. **Conta no Telegram** (o app que voce ja usa no celular).
3. **Conta no Groq** (a IA gratuita) -> https://console.groq.com

Nao precisa saber programar. Voce so vai copiar, colar e clicar.

---

## Passo 1 - Criar o bot no Telegram (BotFather)

1. No Telegram, procure por **@BotFather** (o oficial tem selo azul de verificado).
2. Envie `/newbot`.
3. Escolha um **nome** (ex: "Meu Assistente IA") e um **username** que termine em `bot`
   (ex: `meu_assistente_ia_bot`).
4. O BotFather vai te devolver um **token**, algo como:
   `8123456789:AAH...seu-token-secreto...`
5. **Copie e guarde esse token.** E a "senha" do seu bot -- nao compartilhe.

---

## Passo 2 - Pegar a chave gratuita da IA (Groq)

1. Acesse https://console.groq.com e faca login (pode entrar com o Google).
2. No menu lateral, clique em **API Keys**.
3. Clique em **Create API Key**, de um nome (ex: "n8n") e crie.
4. **Copie a chave** (comeca com `gsk_...`). Ela so aparece uma vez -- guarde.

> Groq e gratuito e nao pede cartao. O modelo que vamos usar (`llama-3.3-70b-versatile`)
> e otimo e rapido. Se um dia quiser trocar por outro, e so mudar o nome do modelo no fluxo.

---

## Passo 3 - Subir o n8n na sua maquina

1. Abra a pasta `automacaoN8N` no VSCode (voce ja fez isso).
2. Confirme que os arquivos `docker-compose.yml` e `workflow_telegram_ia.json` estao la.
3. No VSCode, abra o **Terminal** (menu Terminal > New Terminal).
4. Rode:

   ```bash
   docker compose up -d
   ```

5. Aguarde o download (so na primeira vez, alguns minutos).
6. Para ver o endereco publico do tunnel, rode:

   ```bash
   docker compose logs -f n8n
   ```

   Procure por uma linha tipo:
   `Tunnel URL: https://algo-aleatorio.hooks.n8n.cloud/`
   Guarde que ela existe (o n8n usa automaticamente pro Telegram). Depois aperte `Ctrl+C`
   pra sair dos logs -- o n8n continua rodando.

7. Abra o navegador em **http://localhost:5678**
8. Na primeira vez o n8n pede pra criar uma conta LOCAL (email + senha, fica so na sua maquina).
   Preencha e entre.

> Para desligar depois: `docker compose down`. Seus dados ficam salvos na pasta `n8n_data`.

---

## Passo 4 - Importar o fluxo pronto

1. Ja dentro do n8n (http://localhost:5678), clique no menu **(tres pontinhos)** no canto
   superior direito > **Import from File...**
2. Selecione o arquivo `workflow_telegram_ia.json` da pasta do projeto.
3. Vai aparecer o fluxo com 3 blocos (nos):
   **Telegram Trigger -> Groq - Chat Completion -> Enviar Resposta**.

Agora falta so conectar suas credenciais (token e chave) em cada no.

---

## Passo 5 - Configurar as credenciais

### 5.1 - Telegram (usado em 2 nos)

1. Clique no no **Telegram Trigger**.
2. No campo **Credential**, clique em **Create New Credential**.
3. Cole o **token do BotFather** (Passo 1) no campo *Access Token* e salve.
4. Clique no no **Enviar Resposta** e, no campo Credential, **escolha a mesma credencial**
   do Telegram que voce acabou de criar.

### 5.2 - Groq (no no do meio)

1. Clique no no **Groq - Chat Completion**.
2. Em **Credential for Header Auth**, clique em **Create New Credential**.
3. Preencha:
   - **Name**: `Authorization`
   - **Value**: `Bearer gsk_suachaveaqui`  (a palavra `Bearer`, um espaco, e a chave do Groq)
4. Salve.

> Detalhe importante: e "Bearer " + a chave, tudo no mesmo campo Value.

---

## Passo 6 - Ativar e testar

1. No canto superior direito, ligue a chave **Active** (o fluxo precisa estar ativo pra
   receber mensagens do Telegram).
2. No Telegram, abra uma conversa com o **seu bot** (pelo username que voce criou).
3. Envie `/start` e depois qualquer pergunta, ex: *"Me explique o que e n8n em uma frase"*.
4. Em segundos o bot responde com a IA. Funcionou!

Se quiser ver o que aconteceu por dentro: no n8n, aba **Executions** mostra cada mensagem
processada, no por no. Otimo pra entender o fluxo (e pra debugar).

---

## Como funciona (o que voce aprendeu)

- **Telegram Trigger**: e um *webhook*. Quando alguem manda mensagem, o Telegram avisa o n8n
  na hora. Esse e o conceito mais importante do n8n -- reagir a eventos em tempo real.
- **Os dados "andam" entre os nos** como um JSON. O texto que a pessoa mandou fica em
  `message.text`, e cada no seguinte pode ler isso.
- **Groq - Chat Completion**: um no HTTP Request que chama a API da LLM. Enviamos um
  *prompt de sistema* (a personalidade do bot) + a mensagem do usuario, e recebemos a resposta.
- **Autenticacao por credencial**: a chave nunca fica escrita no fluxo -- fica guardada
  em seguranca no n8n (Header Auth). Boa pratica de verdade.
- **Enviar Resposta**: pega o texto que a IA gerou (`choices[0].message.content`) e devolve
  pro mesmo chat de onde veio a pergunta (`chat.id`).

Esse padrao **gatilho -> processa -> IA decide -> responde** e a base de quase toda
automacao que voce vai construir daqui pra frente.

---

## Ideias pra evoluir (e enriquecer o portfolio)

1. **Personalidade**: mude o texto do *system* no no do Groq pra criar um bot especialista
   (tutor de programacao, revisor de curriculo, coach de estudos).
2. **Memoria de conversa**: guarde as ultimas mensagens (ex: em Google Sheets ou no proprio
   n8n) e mande junto no prompt, pro bot lembrar do contexto.
3. **Historico**: adicione um no Google Sheets no final que salva pergunta + resposta + data.
4. **Audio**: aceite mensagens de voz -> transcreva (Groq tem Whisper gratuito) -> responda.
5. **Trocar a IA**: no mesmo no HTTP, dava pra apontar pro Gemini do Google. Mesma logica.

---

## Problemas comuns

- **Bot nao responde**: confira se o fluxo esta **Active** e se o Docker/n8n esta rodando.
- **Erro 401 no no do Groq**: o Value da credencial precisa ser `Bearer ` + a chave (com o espaco).
- **Nao consigo logar em http://localhost:5678**: confirme que subiu com o `docker-compose.yml`
  deste projeto (ele ja tem `N8N_SECURE_COOKIE=false`).
- **Tunnel mudou de endereco**: o tunnel gratuito e temporario e pode mudar ao reiniciar --
  o n8n reconfigura o webhook do Telegram sozinho quando o fluxo esta Active. Se travar,
  desative e reative o fluxo.
- **Ver logs**: `docker compose logs -f n8n`.

Bom projeto! Qualquer erro que aparecer no terminal, e so me mandar a mensagem que eu te ajudo.
