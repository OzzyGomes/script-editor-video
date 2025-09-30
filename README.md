## AUTO EDIT - Editor Automático de Vídeo

Ferramenta para editar vídeos automaticamente: remove silêncios longos, fillers ("é..., ah..., hum..."), detecta regravações mantendo a última versão, melhora o áudio e opcionalmente gera legendas SRT. Também suporta modo de geração de legenda isolado (somente SRT) sem renderizar vídeo.

### Requisitos
- ffmpeg/ffprobe instalados no sistema
- Python 3.11+ com dependências:
```bash
pip install faster-whisper webrtcvad rapidfuzz moviepy pydub numpy unidecode imageio-ffmpeg
```

Observação: ao usar apenas a geração de SRT (`--srt_only`), o MoviePy não é carregado.

## Uso Básico (CLI)

- Cortar silêncios + fillers + regravações, melhorar áudio (padrão):
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --language pt
```

- Apenas melhorar áudio (sem cortes):
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --audio_only --audio_enhance on
```

- Exportar igualando bitrate do original:
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --quality_mode match_bitrate
```

- Exportar com CRF quase-lossless:
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --quality_mode crf --video_crf 16
```

- Gerar somente legenda SRT (sem renderizar vídeo):
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --srt_only --language pt
# Gera out.srt
```

- Renderizar vídeo e também salvar SRT:
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --write_srt --language pt
# Requer MoviePy instalado
```

## Parâmetros da CLI
- `--input` (obrigatório): arquivo de vídeo de entrada
- `--output` (obrigatório): arquivo de vídeo de saída (base usada para nome do SRT)
- `--language`: `auto|pt|en` (padrão: auto)
- `--whisper_model_size`: `tiny|base|small|medium|large` (padrão: small)

- VAD (detecção de fala):
  - `--vad_aggressiveness` 0-3 (padrão: 2)
  - `--speech_padding_ms` (padrão: 150)
  - `--min_speech_ms` (padrão: 200)
  - `--min_gap_ms` (padrão: 150)

- Transcrição:
  - `--silence_split_ms` pausa para dividir frases (padrão: 700)

- Regravações:
  - `--repeat_window_s` janela em segundos (padrão: 60.0)
  - `--repeat_similarity` 0-100 (padrão: 88.0)

- Fillers:
  - `--filler_pad_ms` (padrão: 100)
  - `--max_filler_len_ms` (padrão: 1200)
  - `--filler_regex` regex customizada

- Renderização / exportação:
  - `--audio_fade_ms` (padrão: 40)
  - `--join_gap_ms` (padrão: 120)
  - `--quality_mode` `match_bitrate|crf|lossless` (padrão: match_bitrate)
  - `--video_crf` (padrão: 18 para H.264)
  - `--video_preset` (padrão: slow)
  - `--target_bitrate` (ex.: 1200k)
  - `--pix_fmt` (padrão: input → preserva)
  - `--keep_fps_res` (padrão: True)

- Áudio:
  - `--audio_enhance` `on|off` (padrão: on)
  - `--audio_only` substitui somente áudio sem cortes de vídeo
  - `--denoise` `none|afftdn|arnndn` (padrão: afftdn)
  - `--denoise_strength` (padrão: 12)
  - `--deesser` `off|light|medium|strong` (padrão: medium)
  - `--highpass_hz` (padrão: 80)
  - `--compress_threshold_db` (padrão: -18)
  - `--compress_ratio` (padrão: 2.5)
  - `--loudnorm_i` (padrão: -16), `--loudnorm_tp` (padrão: -1.5), `--loudnorm_lra` (padrão: 11)
  - `--audio_bitrate` (padrão: 192k)

- Legenda:
  - `--write_srt` gerar SRT além do vídeo
  - `--srt_only` gerar apenas SRT e encerrar sem renderizar

## Principais Funções (alto nível)
- `probe_input(input_path)`
  - Usa ffprobe para extrair metadados de vídeo/áudio (codec, bitrate, fps, duração etc.)

- `load_media(input_path)`
  - Extrai o áudio mono PCM 16kHz com ffmpeg e retorna metadados

- `load_audio_data(audio_path)`
  - Retorna amostras de áudio como numpy a partir de um WAV via ffmpeg

- `run_vad(audio_wav_path, sr, ...)`
  - VAD (webrtcvad) para obter segmentos de fala, com padding e merge inteligente

- `transcribe_segments(audio_wav_path, segments, language, model_size, ...)`
  - Transcreve via faster-whisper com timestamps por palavra (opcional)

- `normalize_text(text)`
  - Normaliza texto para comparação de similaridade (lowercase, sem acentos/pontuação)

- `detect_fillers(sentences, cfg)`
  - Encontra intervalos curtos de fillers e retorna ranges a remover

- `detect_retakes(sentences, repeat_window_s, repeat_similarity)`
  - Detecta regravações e marca frases anteriores para exclusão, mantendo a última

- `build_final_timeline(vad_segments, excluded_indices, sentences, join_gap_ms)`
  - Constrói timeline final mantendo apenas trechos relevantes e unindo gaps pequenos

- `enhance_audio(in_wav, out_wav, cfg)`
  - Melhora áudio com pipeline ffmpeg (denoise, de-esser, compressor, loudnorm 2-pass)

- `render_video(input_path, kept_intervals, output_path, meta, video_quality, ...)`
  - Renderiza cortes com MoviePy (import feito dentro da função para evitar dependência quando não usado)
  - Também suporta modo áudio-apenas com remux via ffmpeg

- `write_srt(sentences_kept, srt_path)`
  - Gera arquivo `.srt` com timestamps e texto

- `main()`
  - CLI: orquestra fluxo, lê parâmetros, executa VAD → transcrição → detecções → timeline → renderização e/ou SRT

## Melhores Práticas
- Instale e valide ffmpeg/ffprobe no PATH:
```bash
ffmpeg -version
ffprobe -version
```
- Para apenas SRT, prefira `--srt_only` para evitar dependências desnecessárias (MoviePy).
- Para renderização, garanta `moviepy` e `imageio-ffmpeg` instalados:
```bash
python -m pip install --upgrade moviepy imageio-ffmpeg
```
- Escolha do `--quality_mode`:
  - `match_bitrate`: mantém taxa aproximada do original
  - `crf`: controle de qualidade por CRF (H.264: 16-23, menor = melhor)
  - `lossless`: sem perdas (arquivos grandes)
- Áudio: comece com `--audio_enhance on` e ajuste `--denoise_strength`, `--deesser` conforme necessidade.
- Transcrição: `--language pt` melhora precisão para PT-BR; use modelos maiores se necessário (`medium`/`large`).

## Exemplos Úteis
- PT-BR, CRF 18, salvar SRT:
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --language pt --quality_mode crf --video_crf 18 --write_srt
```

- Só SRT, sem renderização:
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --srt_only --language pt
```

- Somente melhorar áudio e remux:
```bash
python auto_edit.py --input in.mp4 --output out.mp4 --audio_only --audio_enhance on
```

## Troubleshooting
- Erro `ModuleNotFoundError: No module named 'moviepy.editor'`:
  - Para renderizar vídeo, instale/atualize: `python -m pip install --upgrade moviepy imageio-ffmpeg`
  - Se for apenas gerar SRT, use `--srt_only` (não requer MoviePy)

- ffmpeg/ffprobe não encontrados:
  - Instale ffmpeg e adicione ao PATH do sistema. Valide com `ffmpeg -version` e `ffprobe -version`.

- Performance da transcrição:
  - Use modelos `base`/`small` para velocidade, `medium`/`large` para maior qualidade.

- Sincronismo do SRT:
  - Ajuste `--speech_padding_ms` e `--silence_split_ms` para melhorar cortes e agrupamento de frases.


