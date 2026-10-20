# Auto Cut AI

[English](README.md) | **Italiano**

Applicazione locale di montaggio assistito dall'AI per video informativi parlati in italiano. Carica una o più clip, descrivi il risultato desiderato, verifica la proposta editoriale e approvala esplicitamente prima del rendering finale in MP4.

Il modello linguistico decide la struttura narrativa. L'allineamento deterministico e FFmpeg eseguono queste decisioni, correggendo i bordi dei tagli per proteggere parole ed espressioni numeriche.

## Funzionalità

- Caricamento di più clip, anche a blocchi tramite l'interfaccia web.
- Trascrizione italiana con `faster-whisper`, modello `large-v3` e timestamp per parola.
- Selezione dei modelli Ollama locali, con preferenza per `qwen3:8b` quando disponibile.
- `NarrativeMap` strutturata e validata con Pydantic: ordine delle clip, importanza dei segmenti, intercalari e strategia editoriale.
- Sei livelli di sintesi, dalla pulizia minima alla riduzione più marcata.
- Proposte verificabili con trascrizione, contenuti mantenuti e rimossi, durata stimata ed Edit Decision List (EDL).
- Modifica dei testi e ricalcolo della proposta prima dell'approvazione.
- Approvazione esplicita con **Via libera** prima della composizione finale.
- Avanzamento in tempo reale tramite server-sent events (SSE), timer totale e di fase, persistenza dei lavori in SQLite.
- Controlli su formato e risoluzione, normalizzazione audio, transizioni ed effetti sonori, sottotitoli opzionali impressi nel video.
- Esportazione MP4 con video H.264 e audio AAC.

## Requisiti

- Python **3.11 o successivo** e Git.
- **FFmpeg e ffprobe** disponibili nel `PATH`. Per imprimere i sottotitoli serve una build FFmpeg con filtro ASS/libass.
- **Ollama** installato e in esecuzione localmente, con almeno un modello scaricato.
- RAM e spazio disco sufficienti per i modelli di trascrizione e linguistici. La trascrizione usa CPU e calcolo `int8` per impostazione predefinita; i tempi dipendono da hardware, modello e durata delle clip.

I pesi dei modelli non sono inclusi. Se lo snapshot Whisper locale configurato non esiste, l'applicazione usa il nome modello `large-v3`, con possibile download dei pesi al primo utilizzo. I download iniziali richiedono internet; l'elaborazione usa servizi locali.

## Installazione

Clona il repository e crea un ambiente virtuale:

```sh
git clone https://github.com/elcerqua/autoedit_AI.git
cd autoedit_AI
python -m venv .venv
```

Attivalo in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Oppure su Linux/macOS:

```sh
source .venv/bin/activate
```

Installa l'applicazione e scarica il modello Ollama predefinito:

```sh
python -m pip install -e .
ollama pull qwen3:8b
```

Assicurati che Ollama sia in esecuzione. Se l'applicazione desktop o il servizio non sono già attivi, esegui `ollama serve` in un altro terminale. Verifica gli strumenti richiesti:

```sh
ollama list
ffmpeg -version
ffprobe -version
```

## Avvio

Dalla cartella principale del repository, con l'ambiente virtuale attivo:

```sh
python -m uvicorn autocut_ai.app:app --host 127.0.0.1 --port 8000
```

Apri l'[interfaccia web](http://127.0.0.1:8000/) oppure la [documentazione API interattiva](http://127.0.0.1:8000/docs).

Il backend serve direttamente il frontend HTML/CSS/JavaScript incluso. Non servono Node.js, una compilazione del frontend o un server frontend separato.

Su Windows sono disponibili anche `run.bat` e `stop.bat`. Il launcher prova prima il Python incorporato della macchina di sviluppo originale, poi ripiega su `python`; usa il comando precedente per scegliere esplicitamente il tuo ambiente virtuale.

## Flusso di montaggio

1. Carica uno o più video e scrivi le indicazioni editoriali.
2. Scegli un modello Ollama locale, il livello di sintesi e le opzioni tecniche.
3. Premi **Analizza video** e segui avanzamento e timer.
4. Nello stato `waiting_for_approval`, verifica strategia, ordine proposto, contenuti selezionati ed EDL. Se necessario, modifica o ricalcola la proposta.
5. Premi **Via libera** per approvare la proposta e avviare la composizione.
6. Scarica l'MP4 quando il lavoro raggiunge lo stato `done`.

### Livelli di sintesi

| Livello | Obiettivo editoriale | Testo rimosso indicativo |
| --- | --- | --- |
| 0 | Nessuna sintesi editoriale; solo pulizia tecnica configurata | 0% |
| 1 | Rimuovere le ridondanze molto evidenti | 2% |
| 2 | Rimuovere ripetizioni e digressioni minori | 5% |
| 3 | Migliorare il ritmo mantenendo un parlato naturale | 10% |
| 4 | Accorciare i blocchi narrativi meno importanti | 15% |
| 5 | Conservare i contenuti essenziali con continuità naturale | 20% |

Sono obiettivi editoriali, non percentuali di riduzione garantite. Coerenza, informazioni essenziali, naturalezza e sicurezza dei tagli hanno la precedenza.

### Pipeline

```text
queued -> interpreting_prompt -> normalizing_clips -> transcribing
       -> narrative_analysis -> refining -> waiting_for_approval

Approvazione esplicita -> composing -> [mixing_music] -> done
Errori -> failed
```

La pipeline si ferma a `waiting_for_approval`. La composizione del video finale richiede l'approvazione di una proposta in quello stato. La fase `mixing_music` è attualmente predisposta senza un mix effettivo; vedi i limiti sotto.

## Opzioni di montaggio e configurazione

L'interfaccia espone transizioni (`cut`, `cross_dissolve`, `fade_black`, `zoom_cut`, `whip_pan`), pulizia dei silenzi, opzioni zoom, normalizzazione audio, effetti sonori da `Sounds/`, stili dei sottotitoli e formato di uscita (`source`, `16:9`, `9:16`, `1:1`). Le risoluzioni usano il formato `LARGHEZZAxALTEZZA`, per esempio `1920x1080`.

Le impostazioni di sicurezza includono margini prima e dopo il taglio, durata minima dei segmenti, soglie di pausa e protezione di numeri ed entità nominate. I valori predefiniti effettivamente selezionabili sono restituiti da `GET /editing-options`.

Le impostazioni applicative sono definite in [`autocut_ai/config.py`](autocut_ai/config.py) tramite `AppSettings`. Per personalizzare l'istanza puoi usare `create_app(AppSettings(...))`. Le opzioni includono percorso e dispositivo Whisper, eseguibile FFmpeg, percorsi dei dati, timeout e limiti di contesto Ollama. L'applicazione non carica automaticamente queste impostazioni da un file `.env`.

## API

| Metodo | Endpoint | Funzione |
| --- | --- | --- |
| `POST` | `/jobs` | Creare un lavoro da clip caricate e impostazioni |
| `GET` | `/jobs/{job_id}` | Leggere stato, timer e proposta |
| `GET` | `/jobs/{job_id}/events` | Ricevere gli eventi SSE di avanzamento |
| `POST` | `/jobs/{job_id}/approve` | Approvare e accodare la composizione finale |
| `PATCH` | `/jobs/{job_id}/proposal-texts` | Modificare i testi o rimuovere decisioni selezionate |
| `POST` | `/jobs/{job_id}/recalculate-proposal` | Ricalcolare la proposta prima dell'approvazione |
| `GET` | `/jobs/{job_id}/download` | Scaricare l'MP4 finale dopo il completamento |
| `GET` | `/llm/models` | Elencare i modelli Ollama locali e quello preferito |
| `GET` | `/editing-options` | Leggere opzioni supportate e valori predefiniti |
| `POST` | `/uploads/start` | Avviare una sessione di caricamento a blocchi |
| `PUT` | `/uploads/{upload_id}/clips/{clip_index}/chunk` | Caricare un blocco di una clip |
| `POST` | `/uploads/{upload_id}/jobs` | Creare un lavoro dai caricamenti completati |

Consulta `/docs` per gli schemi di richiesta e risposta. Se Ollama non è raggiungibile, `/llm/models` restituisce un errore chiaro con stato `503`; approvazione nello stato errato e download prima del completamento restituiscono `409`.

## Dati e struttura del progetto

```text
autocut_ai/       API FastAPI, schemi, persistenza, pipeline editoriale, rendering
frontend/        Interfaccia web servita da FastAPI
Sounds/          Effetti MP3 per le transizioni
tests/           Test unitari, di integrazione e test marcati su media reali
tests/fixtures/  Video di esempio 1.mp4 e 2.mp4
docs/            Note e piani di sviluppo
data/            Database lavori, caricamenti, risultati e file temporanei
models/          Cache locale opzionale dei modelli
```

I lavori sono persistiti in `data/jobs.sqlite3`. Le proposte vengono esportate in `proposal.txt` e `proposal.json`; il video finale si trova normalmente in `data/outputs/{job_id}/final.mp4`.

**Conservazione attuale dei file:** l'avvio di un nuovo caricamento o lavoro elimina le cartelle di caricamento e risultato degli altri lavori. La pulizia rimuove anche intermedi di rendering e cache temporanee, conservando le esportazioni della proposta corrente e il video finale completato. Scarica o copia i risultati prima di avviare un altro lavoro. I record SQLite possono rimanere anche dopo la rimozione dei relativi media.

Dati di esecuzione, modelli scaricati, cache, ambienti virtuali e file `.env` privati sono esclusi da Git.

## Test

Installa le dipendenze di sviluppo:

```sh
python -m pip install -e ".[dev]"
```

Esegui la suite ordinaria escludendo i test marcati come lenti o end-to-end:

```sh
python -m pytest -m "not slow and not e2e"
```

Esegui i test marcati su media reali, oppure l'intera suite:

```sh
python -m pytest -m "slow and e2e"
python -m pytest
```

Sono inclusi entrambi i video `tests/fixtures/1.mp4` e `tests/fixtures/2.mp4`. I test di rendering reale richiedono FFmpeg/ffprobe; la maggior parte dei test della pipeline simula i servizi pesanti. La copertura include validazione narrativa, sicurezza dei tagli, approvazione obbligatoria, elenco modelli, caricamenti, revisione della proposta, generazione sottotitoli e rendering.

## Limiti attuali

- Trascrizione e indicazioni editoriali sono orientate all'italiano; anche l'interfaccia è attualmente in italiano.
- Esistono i controlli per la musica e la fase `mixing_music`, ma il renderer attuale non implementa generazione musicale, mix di una traccia di sottofondo o ducking.
- Se il rendering delle transizioni fallisce, viene usata la concatenazione e il ripiego viene registrato nei log.
- La durata stimata può differire da quella renderizzata per le transizioni e i margini protettivi dei tagli.
- L'applicazione locale non include autenticazione e usa un esecutore semplice in background. La pulizia dei dati presuppone un flusso di lavoro attivo alla volta; non è un servizio multiutente concorrente.
- La modifica dei testi della proposta non genera una nuova voce. Il video finale usa le registrazioni originali.

## Licenza

Questo progetto è distribuito con licenza [GNU General Public License v3.0](LICENSE) (`GPL-3.0-only`). Il testo completo della licenza è disponibile nel file [LICENSE](LICENSE).
