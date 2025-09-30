#!/usr/bin/env python3
"""
AUTO EDIT - Editor Automático de Vídeo Avançado
==============================================

Script para remover pausas, fillers, regravações e melhorar qualidade de áudio/vídeo,
mantendo sempre a última ocorrência de frases repetidas.

INSTALAÇÃO:
1. Instale o ffmpeg no sistema
2. pip install faster-whisper webrtcvad rapidfuzz moviepy pydub numpy unidecode

USO BÁSICO:
# Cortar silêncios + regravações + melhorar áudio (padrão):
python auto_edit.py --input in.mp4 --output out.mp4 --language pt

# Apenas melhorar áudio (sem cortes):
python auto_edit.py --input in.mp4 --output out.mp4 --audio_only --audio_enhance on

# Exportar igualando bitrate do original:
python auto_edit.py --input in.mp4 --output out.mp4 --quality_mode match_bitrate

# Exportar com CRF quase-lossless:
python auto_edit.py --input in.mp4 --output out.mp4 --quality_mode crf --video_crf 16

FUNCIONALIDADES:
- Remove silêncios longos automaticamente
- Detecta e remove fillers/hesitações (é..., hum..., hã...)
- Detecta e remove regravações (mantém a última versão)
- Melhora qualidade de áudio (denoise, de-esser, normalização EBU R128)
- Preserva qualidade original do vídeo (mesmo codec, resolução, fps)
- Gera legendas SRT opcionais
- Aplica micro-fades para evitar clicks audíveis

AUTOR: Gerado automaticamente
VERSÃO: 2.0
"""

import argparse
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import warnings

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import fuzz
import rapidfuzz
from unidecode import unidecode

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")
warnings.filterwarnings("ignore", category=FutureWarning, module="moviepy")


class AutoEditor:
    """Editor automático de vídeo com detecção de voz e remoção de regravações."""
    
    def __init__(self, config: Dict):
        """Inicializa o editor com configurações."""
        self.config = config
        self.temp_files = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Limpa arquivos temporários."""
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Remove arquivos temporários criados durante o processamento."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Erro ao remover arquivo temporário {temp_file}: {e}")
        self.temp_files.clear()
    
    def create_temp_file(self, suffix: str = ".wav") -> str:
        """Cria um arquivo temporário e adiciona à lista de limpeza."""
        temp_file = tempfile.mktemp(suffix=suffix)
        self.temp_files.append(temp_file)
        return temp_file


def probe_input(input_path: str) -> Dict:
    """
    Analisa arquivo de entrada usando ffprobe para obter metadados completos.
    
    Args:
        input_path: Caminho para o arquivo de vídeo
        
    Returns:
        Dict com metadados do vídeo e áudio
    """
    logger.info(f"Analisando arquivo: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    
    # Verificar se ffprobe está disponível
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffprobe não encontrado. Instale o ffmpeg no sistema.")
    
    # Comando ffprobe para obter metadados detalhados
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", input_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao analisar arquivo: {result.stderr}")
    
    import json
    data = json.loads(result.stdout)
    
    # Extrair metadados de vídeo
    video_stream = None
    audio_stream = None
    
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream
    
    if not video_stream:
        raise RuntimeError("Nenhum stream de vídeo encontrado")
    
    # Metadados de vídeo
    v_codec = video_stream.get("codec_name", "unknown")
    v_bitrate = int(video_stream.get("bit_rate", 0)) if video_stream.get("bit_rate") else None
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    fps_str = video_stream.get("r_frame_rate", "0/1")
    fps = eval(fps_str) if "/" in fps_str else float(fps_str)
    pix_fmt = video_stream.get("pix_fmt", "yuv420p")
    profile = video_stream.get("profile", "")
    level = video_stream.get("level", "")
    
    # Metadados de áudio
    a_codec = audio_stream.get("codec_name", "unknown") if audio_stream else "unknown"
    a_sr = int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
    a_channels = int(audio_stream.get("channels", 0)) if audio_stream else 0
    a_bitrate = int(audio_stream.get("bit_rate", 0)) if audio_stream and audio_stream.get("bit_rate") else None
    
    # Duração total
    duration = float(data.get("format", {}).get("duration", 0))
    
    meta = {
        "v_codec": v_codec,
        "v_bitrate": v_bitrate,
        "width": width,
        "height": height,
        "fps": fps,
        "pix_fmt": pix_fmt,
        "profile": profile,
        "level": level,
        "a_codec": a_codec,
        "a_sr": a_sr,
        "a_channels": a_channels,
        "a_bitrate": a_bitrate,
        "duration": duration
    }
    
    logger.info(f"Metadados: {width}x{height}@{fps:.2f}fps, {v_codec}, {pix_fmt}")
    if a_codec != "unknown":
        logger.info(f"Áudio: {a_codec}, {a_sr}Hz, {a_channels}ch")
    
    return meta


def load_media(input_path: str) -> Dict:
    """
    Carrega vídeo e extrai áudio mono 16kHz PCM.
    
    Args:
        input_path: Caminho para o arquivo de vídeo
        
    Returns:
        Dict com metadados do vídeo e caminho do áudio extraído
    """
    logger.info(f"Carregando mídia: {input_path}")
    
    # Obter metadados usando ffprobe
    meta = probe_input(input_path)
    
    # Verificar se ffmpeg está disponível
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg não encontrado. Instale o ffmpeg no sistema.")
    
    try:
        # Extrair áudio com ffmpeg diretamente para melhor controle
        temp_audio_path = tempfile.mktemp(suffix=".wav")
        
        # Comando ffmpeg para extrair áudio mono 16kHz PCM
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ac", "1",  # mono
            "-ar", "16000",  # 16kHz
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
            "-y",  # sobrescrever arquivo de saída
            temp_audio_path
        ]
        
        logger.info("Extraindo áudio com ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Erro ao extrair áudio: {result.stderr}")
        
        logger.info(f"Áudio extraído: {meta['duration']:.2f}s, 16000Hz, mono")
        
        return {
            "audio_wav_path": temp_audio_path,
            "sr": 16000,
            "meta": meta
        }
        
    except Exception as e:
        logger.error(f"Erro ao carregar mídia: {e}")
        raise


def load_audio_data(audio_path: str) -> np.ndarray:
    """
    Carrega dados de áudio usando ffmpeg diretamente.
    
    Args:
        audio_path: Caminho para arquivo de áudio
        
    Returns:
        Array numpy com dados de áudio
    """
    # Usar ffmpeg para extrair dados brutos
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "s16le",  # 16-bit little-endian
        "-ac", "1",     # mono
        "-ar", "16000", # 16kHz
        "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao carregar áudio: {result.stderr}")
    
    # Converter bytes para array numpy
    audio_data = np.frombuffer(result.stdout, dtype=np.int16)
    return audio_data


def run_vad(audio_wav_path: str, sr: int, vad_aggressiveness: int = 2, 
           min_speech_ms: int = 200, min_gap_ms: int = 150, 
           speech_padding_ms: int = 150) -> List[Tuple[float, float]]:
    """
    Executa detecção de atividade de voz (VAD) no áudio.
    
    Args:
        audio_wav_path: Caminho para arquivo de áudio WAV
        sr: Sample rate do áudio
        vad_aggressiveness: Agressividade do VAD (0-3)
        min_speech_ms: Duração mínima de segmento de fala
        min_gap_ms: Duração mínima de gap entre segmentos
        speech_padding_ms: Padding para adicionar aos segmentos
        
    Returns:
        Lista de tuplas (start, end) em segundos
    """
    logger.info("Executando detecção de atividade de voz (VAD)...")
    
    # Carregar áudio usando ffmpeg diretamente
    audio_data = load_audio_data(audio_wav_path)
    
    # Configurar VAD
    vad = webrtcvad.Vad(vad_aggressiveness)
    
    # Parâmetros do VAD
    frame_duration_ms = 30  # 30ms por frame
    frame_size = int(sr * frame_duration_ms / 1000)
    
    # Converter para frames de 30ms
    frames = []
    for i in range(0, len(audio_data) - frame_size + 1, frame_size):
        frame = audio_data[i:i + frame_size]
        if len(frame) == frame_size:
            frames.append(frame.tobytes())
    
    # Detectar atividade de voz
    speech_frames = []
    for i, frame in enumerate(frames):
        try:
            is_speech = vad.is_speech(frame, sr)
            speech_frames.append((i * frame_duration_ms / 1000.0, is_speech))
        except Exception as e:
            logger.warning(f"Erro no frame {i}: {e}")
            speech_frames.append((i * frame_duration_ms / 1000.0, False))
    
    # Agrupar frames em segmentos
    segments = []
    current_start = None
    
    for timestamp, is_speech in speech_frames:
        if is_speech and current_start is None:
            current_start = timestamp
        elif not is_speech and current_start is not None:
            current_end = timestamp
            if (current_end - current_start) * 1000 >= min_speech_ms:
                segments.append((current_start, current_end))
            current_start = None
    
    # Adicionar último segmento se terminar com fala
    if current_start is not None:
        current_end = len(audio_data) / sr
        if (current_end - current_start) * 1000 >= min_speech_ms:
            segments.append((current_start, current_end))
    
    # Aplicar padding
    padding_s = speech_padding_ms / 1000.0
    padded_segments = []
    for start, end in segments:
        new_start = max(0, start - padding_s)
        new_end = min(len(audio_data) / sr, end + padding_s)
        padded_segments.append((new_start, new_end))
    
    # Unir segmentos muito próximos
    if not padded_segments:
        return []
    
    merged_segments = [padded_segments[0]]
    for start, end in padded_segments[1:]:
        last_start, last_end = merged_segments[-1]
        gap_ms = (start - last_end) * 1000
        
        if gap_ms <= min_gap_ms:
            # Unir segmentos
            merged_segments[-1] = (last_start, end)
        else:
            merged_segments.append((start, end))
    
    logger.info(f"VAD encontrou {len(merged_segments)} segmentos de fala")
    return merged_segments


def transcribe_segments(audio_wav_path: str, segments: List[Tuple[float, float]], 
                       language: str = "auto", model_size: str = "small", 
                       silence_split_ms: int = 700, word_level: bool = True) -> List[Dict]:
    """
    Transcreve segmentos de áudio usando Whisper com word timestamps.
    
    Args:
        audio_wav_path: Caminho para arquivo de áudio WAV
        segments: Lista de segmentos (start, end) em segundos
        language: Idioma para transcrição
        model_size: Tamanho do modelo Whisper
        silence_split_ms: Pausa para dividir frases em ms
        word_level: Incluir timestamps de palavras
        
    Returns:
        Lista de frases com timestamps e palavras
    """
    logger.info("Iniciando transcrição com Whisper...")
    
    if not segments:
        logger.warning("Nenhum segmento para transcrever")
        return []
    
    try:
        # Carregar modelo Whisper
        logger.info(f"Carregando modelo Whisper: {model_size}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        sentences = []
        
        for i, (start, end) in enumerate(segments):
            logger.info(f"Transcrevendo segmento {i+1}/{len(segments)}: {start:.2f}s - {end:.2f}s")
            
            # Extrair segmento de áudio com ffmpeg
            temp_segment_path = tempfile.mktemp(suffix=".wav")
            
            cmd = [
                "ffmpeg", "-i", audio_wav_path,
                "-ss", str(start),
                "-t", str(end - start),
                "-ac", "1",
                "-ar", "16000",
                "-y",
                temp_segment_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Erro ao extrair segmento {i+1}: {result.stderr}")
                continue
            
            try:
                # Transcrever segmento
                segments_result, info = model.transcribe(
                    temp_segment_path,
                    language=language if language != "auto" else None,
                    word_timestamps=word_level
                )
                
                # Processar resultados
                for segment_result in segments_result:
                    if segment_result.text.strip():
                        # Ajustar timestamps para o vídeo completo
                        adjusted_start = start + segment_result.start
                        adjusted_end = start + segment_result.end
                        
                        sentence_data = {
                            "text": segment_result.text.strip(),
                            "start": adjusted_start,
                            "end": adjusted_end,
                            "confidence": getattr(segment_result, 'avg_logprob', 0.0)
                        }
                        
                        # Adicionar timestamps de palavras se disponível
                        if word_level and hasattr(segment_result, 'words'):
                            words = []
                            for word in segment_result.words:
                                words.append({
                                    "text": word.word,
                                    "start": start + word.start,
                                    "end": start + word.end,
                                    "probability": getattr(word, 'probability', 0.0)
                                })
                            sentence_data["words"] = words
                        
                        sentences.append(sentence_data)
                
            finally:
                # Limpar arquivo temporário do segmento
                try:
                    os.remove(temp_segment_path)
                except:
                    pass
        
        # Normalizar e segmentar frases por pausas
        normalized_sentences = []
        for sentence in sentences:
            normalized_text = normalize_text(sentence["text"])
            if normalized_text and len(normalized_text) > 3:  # Filtrar textos muito curtos
                normalized_sentences.append({
                    **sentence,
                    "normalized_text": normalized_text
                })
        
        logger.info(f"Transcrição concluída: {len(normalized_sentences)} frases")
        return normalized_sentences
        
    except Exception as e:
        logger.error(f"Erro na transcrição: {e}")
        logger.info("Usando fallback: apenas VAD sem transcrição")
        return []


def detect_fillers(sentences: List[Dict], fillers_cfg: Dict) -> List[Tuple[float, float]]:
    """
    Detecta fillers/hesitações baseado em palavras e padrões regex.
    
    Args:
        sentences: Lista de frases com timestamps de palavras
        fillers_cfg: Configuração de detecção de fillers
        
    Returns:
        Lista de intervalos (start, end) para remover
    """
    logger.info("Detectando fillers e hesitações...")
    
    if not sentences:
        return []
    
    # Padrão regex para fillers em PT-BR
    filler_pattern = fillers_cfg.get("filler_regex", 
        r"\b(e+|ee+|eh+|h[au]+|hum+m+|hmm+|ah+|uh+|u+m+)\b")
    
    max_filler_len_ms = fillers_cfg.get("max_filler_len_ms", 1200)
    filler_pad_ms = fillers_cfg.get("filler_pad_ms", 100)
    
    intervals_to_remove = []
    
    for sentence in sentences:
        if "words" not in sentence:
            continue
            
        words = sentence["words"]
        if not words:
            continue
        
        # Agrupar palavras consecutivas que são fillers
        current_filler_start = None
        current_filler_end = None
        
        for i, word in enumerate(words):
            word_text = word.get("text", "").strip()
            if not word_text:
                continue
                
            # Normalizar texto para comparação
            normalized_text = unidecode(word_text.lower().strip())
            
            # Verificar se é filler
            if re.search(filler_pattern, normalized_text, re.IGNORECASE):
                word_start = word.get("start", 0)
                word_end = word.get("end", word_start)
                
                if current_filler_start is None:
                    # Início de um novo bloco de filler
                    current_filler_start = word_start
                    current_filler_end = word_end
                else:
                    # Continuar bloco existente
                    current_filler_end = word_end
            else:
                # Fim do bloco de filler
                if current_filler_start is not None:
                    filler_duration_ms = (current_filler_end - current_filler_start) * 1000
                    
                    if filler_duration_ms <= max_filler_len_ms:
                        # Adicionar padding
                        start_with_padding = max(0, current_filler_start - filler_pad_ms / 1000)
                        end_with_padding = current_filler_end + filler_pad_ms / 1000
                        
                        intervals_to_remove.append((start_with_padding, end_with_padding))
                        logger.info(f"Filler detectado: '{sentence['text'][:50]}...' ({filler_duration_ms:.0f}ms)")
                    
                    current_filler_start = None
                    current_filler_end = None
        
        # Verificar se há filler no final da frase
        if current_filler_start is not None:
            filler_duration_ms = (current_filler_end - current_filler_start) * 1000
            
            if filler_duration_ms <= max_filler_len_ms:
                start_with_padding = max(0, current_filler_start - filler_pad_ms / 1000)
                end_with_padding = current_filler_end + filler_pad_ms / 1000
                
                intervals_to_remove.append((start_with_padding, end_with_padding))
                logger.info(f"Filler detectado: '{sentence['text'][:50]}...' ({filler_duration_ms:.0f}ms)")
    
    logger.info(f"Detectados {len(intervals_to_remove)} fillers para remover")
    return intervals_to_remove


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparação de similaridade.
    
    Args:
        text: Texto original
        
    Returns:
        Texto normalizado
    """
    # Converter para minúsculas e remover acentos
    text = unidecode(text.lower().strip())
    
    # Remover pontuação excessiva
    text = re.sub(r'[.,!?;:]+', '', text)
    
    # Remover fillers comuns
    fillers = ['e...', 'ah...', 'uh...', 'hmm...', 'bem...', 'entao...', 'assim...']
    for filler in fillers:
        text = text.replace(filler, '')
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_retakes(sentences: List[Dict], repeat_window_s: float = 60.0, 
                  repeat_similarity: float = 88.0) -> Set[int]:
    """
    Detecta regravações (frases repetidas) e marca as anteriores para exclusão.
    
    Args:
        sentences: Lista de frases com timestamps
        repeat_window_s: Janela de tempo para buscar repetições
        repeat_similarity: Limiar de similaridade (0-100)
        
    Returns:
        Conjunto de índices de frases a excluir
    """
    logger.info("Detectando regravações...")
    
    if not sentences:
        return set()
    
    excluded_indices = set()
    
    for i, current_sentence in enumerate(sentences):
        if i in excluded_indices:
            continue
            
        current_text = current_sentence["normalized_text"]
        current_time = current_sentence["start"]
        
        # Buscar frases anteriores na janela de tempo
        window_start = current_time - repeat_window_s
        
        for j in range(i):
            if j in excluded_indices:
                continue
                
            prev_sentence = sentences[j]
            prev_time = prev_sentence["start"]
            
            # Verificar se está na janela de tempo
            if prev_time < window_start:
                continue
                
            prev_text = prev_sentence["normalized_text"]
            
            # Calcular similaridade
            similarity = fuzz.token_set_ratio(current_text, prev_text)
            
            if similarity >= repeat_similarity:
                logger.info(f"Regravação detectada: '{prev_text[:50]}...' -> '{current_text[:50]}...' (similaridade: {similarity:.1f}%)")
                excluded_indices.add(j)
    
    logger.info(f"Detectadas {len(excluded_indices)} regravações para remover")
    return excluded_indices


def enhance_audio(in_wav: str, out_wav: str, cfg: Dict) -> None:
    """
    Melhora qualidade de áudio usando filtros ffmpeg com 2-pass loudnorm.
    
    Args:
        in_wav: Caminho do arquivo WAV de entrada
        out_wav: Caminho do arquivo WAV de saída
        cfg: Configuração de melhoria de áudio
    """
    logger.info("Melhorando qualidade de áudio...")
    
    # Construir pipeline de filtros
    filters = []
    
    # Denoise
    if cfg.get("denoise") == "afftdn":
        denoise_strength = cfg.get("denoise_strength", 12)
        filters.append(f"afftdn=nr={denoise_strength}")
    elif cfg.get("denoise") == "arnndn":
        # Usar arnndn se disponível (requer modelo)
        filters.append("arnndn")
    
    # High-pass filter
    highpass_hz = cfg.get("highpass_hz", 80)
    filters.append(f"highpass=f={highpass_hz}")
    
    # De-esser
    deesser_mode = cfg.get("deesser", "medium")
    if deesser_mode != "off":
        deesser_strength = {"light": 0.5, "medium": 1.0, "strong": 1.5}.get(deesser_mode, 1.0)
        filters.append(f"deesser=i={deesser_strength}")
    
    # Compressão leve
    compress_thresh = cfg.get("compress_threshold_db", -18)
    compress_ratio = cfg.get("compress_ratio", 2.5)
    filters.append(f"acompressor=threshold={compress_thresh}dB:ratio={compress_ratio}:attack=10:release=60")
    
    # Loudness normalization (2-pass)
    lufs_target = cfg.get("loudnorm_i", -16)
    lufs_tp = cfg.get("loudnorm_tp", -1.5)
    lufs_lra = cfg.get("loudnorm_lra", 11)
    
    # Primeiro pass: medir loudness
    measure_cmd = [
        "ffmpeg", "-i", in_wav,
        "-af", f"loudnorm=I={lufs_target}:TP={lufs_tp}:LRA={lufs_lra}:print_format=json",
        "-f", "null", "-"
    ]
    
    logger.info("Medindo loudness (pass 1)...")
    result = subprocess.run(measure_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.warning("Erro na medição de loudness, usando valores padrão")
        measured_i = lufs_target
        measured_tp = lufs_tp
        measured_lra = lufs_lra
    else:
        # Extrair valores medidos do output
        import json
        try:
            # O ffmpeg pode imprimir múltiplas linhas, buscar a última com JSON válido
            for line in reversed(result.stderr.split('\n')):
                if line.strip().startswith('{'):
                    loudness_data = json.loads(line.strip())
                    measured_i = loudness_data.get('input_i', lufs_target)
                    measured_tp = loudness_data.get('input_tp', lufs_tp)
                    measured_lra = loudness_data.get('input_lra', lufs_lra)
                    break
            else:
                measured_i = lufs_target
                measured_tp = lufs_tp
                measured_lra = lufs_lra
        except:
            measured_i = lufs_target
            measured_tp = lufs_tp
            measured_lra = lufs_lra
    
    # Adicionar loudnorm ao pipeline
    filters.append(f"loudnorm=I={lufs_target}:TP={lufs_tp}:LRA={lufs_lra}:measured_I={measured_i}:measured_TP={measured_tp}:measured_LRA={measured_lra}")
    
    # Construir comando final
    filter_chain = ",".join(filters)
    
    cmd = [
        "ffmpeg", "-i", in_wav,
        "-af", filter_chain,
        "-ar", "48000",  # Upsample para melhor qualidade
        "-ac", "2",      # Estéreo
        "-y", out_wav
    ]
    
    logger.info(f"Aplicando filtros: {filter_chain}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao melhorar áudio: {result.stderr}")
    
    logger.info("Melhoria de áudio concluída")


def build_final_timeline(vad_segments: List[Tuple[float, float]], 
                        excluded_intervals: Set[int],
                        sentences: List[Dict],
                        join_gap_ms: int = 120) -> List[Tuple[float, float]]:
    """
    Constrói timeline final removendo intervalos excluídos.
    
    Args:
        vad_segments: Segmentos originais do VAD
        excluded_intervals: Índices de frases a excluir
        sentences: Lista de frases transcritas
        join_gap_ms: Gap máximo para unir segmentos
        
    Returns:
        Lista de intervalos finais a manter
    """
    logger.info("Construindo timeline final...")
    
    if not sentences:
        # Fallback: usar apenas VAD
        logger.info("Usando timeline baseada apenas em VAD")
        return vad_segments
    
    # Criar intervalos a excluir baseados nas frases
    exclude_ranges = []
    for i in excluded_intervals:
        if i < len(sentences):
            sentence = sentences[i]
            exclude_ranges.append((sentence["start"], sentence["end"]))
    
    # Ordenar intervalos de exclusão
    exclude_ranges.sort()
    
    # Aplicar exclusões aos segmentos VAD
    kept_intervals = []
    
    for vad_start, vad_end in vad_segments:
        current_start = vad_start
        
        for exclude_start, exclude_end in exclude_ranges:
            # Verificar sobreposição
            if exclude_start < vad_end and exclude_end > vad_start:
                # Adicionar parte antes da exclusão
                if exclude_start > current_start:
                    kept_intervals.append((current_start, exclude_start))
                
                # Atualizar início para depois da exclusão
                current_start = max(current_start, exclude_end)
        
        # Adicionar parte final se restou algo
        if current_start < vad_end:
            kept_intervals.append((current_start, vad_end))
    
    # Unir intervalos muito próximos
    if not kept_intervals:
        return []
    
    merged_intervals = [kept_intervals[0]]
    join_gap_s = join_gap_ms / 1000.0
    
    for start, end in kept_intervals[1:]:
        last_start, last_end = merged_intervals[-1]
        gap = start - last_end
        
        if gap <= join_gap_s:
            # Unir intervalos
            merged_intervals[-1] = (last_start, end)
        else:
            merged_intervals.append((start, end))
    
    logger.info(f"Timeline final: {len(merged_intervals)} intervalos mantidos")
    return merged_intervals


def render_video(input_path: str, kept_intervals: List[Tuple[float, float]], 
                output_path: str, meta: Dict, video_quality: Dict, 
                audio_fade_ms: int = 40, audio_wav_final: str = None) -> None:
    """
    Renderiza vídeo final mantendo apenas os intervalos especificados.
    
    Args:
        input_path: Caminho do vídeo original
        kept_intervals: Intervalos a manter
        output_path: Caminho do vídeo de saída
        meta: Metadados do vídeo original
        video_quality: Configuração de qualidade de vídeo
        audio_fade_ms: Duração do fade de áudio
        audio_wav_final: Caminho do áudio melhorado (opcional)
    """
    logger.info("Renderizando vídeo final...")
    
    if not kept_intervals:
        logger.warning("Nenhum intervalo para renderizar")
        return
    
    # Modo áudio-apenas: apenas substituir áudio
    if video_quality.get("audio_only", False):
        logger.info("Modo áudio-apenas: substituindo áudio sem cortes")
        
        if not audio_wav_final or not os.path.exists(audio_wav_final):
            raise RuntimeError("Arquivo de áudio melhorado não encontrado para modo áudio-apenas")
        
        # Converter áudio para formato compatível
        temp_audio = tempfile.mktemp(suffix=".aac")
        cmd = [
            "ffmpeg", "-i", audio_wav_final,
            "-c:a", "aac", "-b:a", video_quality.get("audio_bitrate", "192k"),
            "-y", temp_audio
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Erro ao converter áudio: {result.stderr}")
        
        # Fazer remux com stream-copy do vídeo
        cmd = [
            "ffmpeg", "-i", input_path, "-i", temp_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            "-movflags", "+faststart",
            "-y", output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Erro ao fazer remux: {result.stderr}")
        
        # Limpar arquivo temporário
        try:
            os.remove(temp_audio)
        except:
            pass
        
        logger.info("Remux concluído")
        return
    
    # Modo com cortes: usar MoviePy
    try:
        # Importar MoviePy apenas quando necessário (evita dependência para modos sem renderização)
        from moviepy import VideoFileClip, concatenate_videoclips
        # Carregar vídeo original
        video = VideoFileClip(input_path)
        
        # Criar clips para cada intervalo
        clips = []
        audio_fade_s = audio_fade_ms / 1000.0
        
        for i, (start, end) in enumerate(kept_intervals):
            logger.info(f"Processando intervalo {i+1}/{len(kept_intervals)}: {start:.2f}s - {end:.2f}s")
            
            # Criar subclip
            subclip = video.subclip(start, end)
            
            # Aplicar fades de áudio
            if subclip.audio:
                subclip = subclip.set_audio(
                    subclip.audio.audio_fadein(audio_fade_s).audio_fadeout(audio_fade_s)
                )
            
            clips.append(subclip)
        
        # Concatenar clips
        if len(clips) == 1:
            final_video = clips[0]
        else:
            final_video = concatenate_videoclips(clips, method="compose")
        
        # Configurar parâmetros de exportação baseados na qualidade
        quality_mode = video_quality.get("mode", "match_bitrate")
        v_codec = meta.get("v_codec", "libx264")
        pix_fmt = video_quality.get("pix_fmt", meta.get("pix_fmt", "yuv420p"))
        fps = meta.get("fps", 30)
        
        # Configurar parâmetros de qualidade
        ffmpeg_params = []
        
        # Formato de pixel
        if pix_fmt:
            ffmpeg_params.extend(["-pix_fmt", pix_fmt])
        
        # Preset
        preset = video_quality.get("video_preset", "slow")
        ffmpeg_params.extend(["-preset", preset])
        
        # Movflags
        ffmpeg_params.extend(["-movflags", "+faststart"])
        
        # Configurar codec e parâmetros
        if quality_mode == "match_bitrate":
            # Usar mesmo bitrate do original
            bitrate = meta.get("v_bitrate")
            if bitrate:
                bitrate_k = f"{bitrate // 1000}k"
                ffmpeg_params.extend(["-b:v", bitrate_k])
                logger.info(f"Qualidade: {quality_mode}, codec: {v_codec}, bitrate: {bitrate_k}")
            else:
                ffmpeg_params.extend(["-crf", "23"])
                logger.info(f"Qualidade: {quality_mode}, codec: {v_codec}, crf: 23 (fallback)")
        elif quality_mode == "crf":
            # Usar CRF
            crf = video_quality.get("video_crf", 18 if v_codec == "libx264" else 20)
            ffmpeg_params.extend(["-crf", str(crf)])
            logger.info(f"Qualidade: {quality_mode}, codec: {v_codec}, crf: {crf}")
        elif quality_mode == "lossless":
            # Lossless
            if v_codec == "libx264":
                ffmpeg_params.extend(["-crf", "0"])
            else:
                ffmpeg_params.extend(["-qp", "0"])
            logger.info(f"Qualidade: {quality_mode}, codec: {v_codec}, lossless")
        else:
            ffmpeg_params.extend(["-crf", "23"])
            logger.info(f"Qualidade: {quality_mode}, codec: {v_codec}, crf: 23 (fallback)")
        
        # Escrever vídeo final
        logger.info(f"Salvando vídeo: {output_path}")
        
        final_video.write_videofile(
            output_path,
            fps=fps,
            codec=v_codec,
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None,
            ffmpeg_params=ffmpeg_params
        )
        
        # Fechar clips
        final_video.close()
        video.close()
        
        logger.info("Renderização concluída")
        
    except Exception as e:
        logger.error(f"Erro ao renderizar vídeo: {e}")
        raise


def write_srt(sentences_kept: List[Dict], srt_path: str) -> None:
    """
    Gera arquivo SRT com as frases mantidas.
    
    Args:
        sentences_kept: Lista de frases a incluir no SRT
        srt_path: Caminho do arquivo SRT
    """
    logger.info(f"Gerando arquivo SRT: {srt_path}")
    
    if not sentences_kept:
        logger.warning("Nenhuma frase para incluir no SRT")
        return
    
    def format_timestamp(seconds: float) -> str:
        """Formata timestamp para SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences_kept, 1):
            start_time = format_timestamp(sentence["start"])
            end_time = format_timestamp(sentence["end"])
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{sentence['text']}\n\n")
    
    logger.info("Arquivo SRT gerado com sucesso")


def main():
    """Função principal com CLI."""
    parser = argparse.ArgumentParser(
        description="Editor automático de vídeo - Remove silêncios, fillers, regravações e melhora áudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Cortar silêncios + regravações + melhorar áudio (padrão):
  python auto_edit.py --input in.mp4 --output out.mp4 --language pt

  # Apenas melhorar áudio (sem cortes):
  python auto_edit.py --input in.mp4 --output out.mp4 --audio_only --audio_enhance on

  # Exportar igualando bitrate do original:
  python auto_edit.py --input in.mp4 --output out.mp4 --quality_mode match_bitrate

  # Exportar com CRF quase-lossless:
  python auto_edit.py --input in.mp4 --output out.mp4 --quality_mode crf --video_crf 16
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument("--input", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", required=True, help="Caminho do vídeo de saída")
    
    # Configurações gerais
    parser.add_argument("--language", choices=["auto", "pt", "en"], default="auto",
                       help="Idioma para transcrição (default: auto)")
    parser.add_argument("--whisper_model_size", default="small",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Tamanho do modelo Whisper (default: small)")
    
    # Configurações de VAD
    parser.add_argument("--vad_aggressiveness", type=int, choices=[0, 1, 2, 3], default=2,
                       help="Agressividade do VAD 0-3 (default: 2)")
    parser.add_argument("--speech_padding_ms", type=int, default=150,
                       help="Padding de fala em ms (default: 150)")
    parser.add_argument("--min_speech_ms", type=int, default=200,
                       help="Duração mínima de fala em ms (default: 200)")
    parser.add_argument("--min_gap_ms", type=int, default=150,
                       help="Gap mínimo entre segmentos em ms (default: 150)")
    
    # Configurações de transcrição
    parser.add_argument("--silence_split_ms", type=int, default=700,
                       help="Pausa para dividir frases em ms (default: 700)")
    
    # Configurações de detecção de regravações
    parser.add_argument("--repeat_window_s", type=float, default=60.0,
                       help="Janela de tempo para detectar repetições em s (default: 60)")
    parser.add_argument("--repeat_similarity", type=float, default=88.0,
                       help="Limiar de similaridade para repetições 0-100 (default: 88)")
    
    # Configurações de fillers
    parser.add_argument("--filler_pad_ms", type=int, default=100,
                       help="Padding para fillers em ms (default: 100)")
    parser.add_argument("--max_filler_len_ms", type=int, default=1200,
                       help="Duração máxima de filler em ms (default: 1200)")
    parser.add_argument("--filler_regex", type=str, default=None,
                       help="Regex customizado para detectar fillers")
    
    # Configurações de renderização
    parser.add_argument("--audio_fade_ms", type=int, default=40,
                       help="Duração do fade de áudio em ms (default: 40)")
    parser.add_argument("--join_gap_ms", type=int, default=120,
                       help="Gap máximo para unir segmentos em ms (default: 120)")
    
    # Qualidade de vídeo
    parser.add_argument("--quality_mode", choices=["match_bitrate", "crf", "lossless"], 
                       default="match_bitrate", help="Modo de qualidade de vídeo (default: match_bitrate)")
    parser.add_argument("--video_crf", type=int, default=18,
                       help="CRF para vídeo (default: 18 para H.264, 20 para H.265)")
    parser.add_argument("--video_preset", default="slow",
                       help="Preset de codificação (default: slow)")
    parser.add_argument("--target_bitrate", type=str, default=None,
                       help="Bitrate de saída (ex: 1000k)")
    parser.add_argument("--pix_fmt", default="input",
                       help="Formato de pixel (default: input)")
    parser.add_argument("--keep_fps_res", action="store_true", default=True,
                       help="Manter FPS e resolução do input (default: True)")
    
    # Áudio / melhoria
    parser.add_argument("--audio_enhance", choices=["on", "off"], default="on",
                       help="Melhorar qualidade de áudio (default: on)")
    parser.add_argument("--audio_only", action="store_true",
                       help="Apenas melhorar áudio (sem cortes)")
    parser.add_argument("--denoise", choices=["none", "afftdn", "arnndn"], default="afftdn",
                       help="Método de denoise (default: afftdn)")
    parser.add_argument("--denoise_strength", type=int, default=12,
                       help="Força do denoise (default: 12)")
    parser.add_argument("--deesser", choices=["off", "light", "medium", "strong"], 
                       default="medium", help="Força do de-esser (default: medium)")
    parser.add_argument("--highpass_hz", type=int, default=80,
                       help="Frequência do high-pass filter (default: 80)")
    parser.add_argument("--compress_threshold_db", type=float, default=-18,
                       help="Threshold de compressão em dB (default: -18)")
    parser.add_argument("--compress_ratio", type=float, default=2.5,
                       help="Ratio de compressão (default: 2.5)")
    parser.add_argument("--loudnorm_i", type=float, default=-16,
                       help="Target LUFS (default: -16)")
    parser.add_argument("--loudnorm_tp", type=float, default=-1.5,
                       help="True peak em dB (default: -1.5)")
    parser.add_argument("--loudnorm_lra", type=float, default=11,
                       help="LRA em LU (default: 11)")
    parser.add_argument("--audio_bitrate", default="192k",
                       help="Bitrate de áudio de saída (default: 192k)")
    
    # Opções
    parser.add_argument("--write_srt", action="store_true",
                       help="Gerar arquivo SRT com legendas")
    parser.add_argument("--srt_only", action="store_true",
                       help="Apenas gerar arquivo SRT e sair")
    
    args = parser.parse_args()
    
    # Validar arquivos
    if not os.path.exists(args.input):
        logger.error(f"Arquivo de entrada não encontrado: {args.input}")
        return 1
    
    # Criar diretório de saída se necessário
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configurações
    config = {
        "language": args.language,
        "whisper_model_size": args.whisper_model_size,
        "vad_aggressiveness": args.vad_aggressiveness,
        "speech_padding_ms": args.speech_padding_ms,
        "min_speech_ms": args.min_speech_ms,
        "min_gap_ms": args.min_gap_ms,
        "silence_split_ms": args.silence_split_ms,
        "repeat_window_s": args.repeat_window_s,
        "repeat_similarity": args.repeat_similarity,
        "audio_fade_ms": args.audio_fade_ms,
        "join_gap_ms": args.join_gap_ms,
        "audio_enhance": args.audio_enhance == "on",
        "audio_only": args.audio_only,
        "denoise": args.denoise,
        "denoise_strength": args.denoise_strength,
        "deesser": args.deesser,
        "highpass_hz": args.highpass_hz,
        "compress_threshold_db": args.compress_threshold_db,
        "compress_ratio": args.compress_ratio,
        "loudnorm_i": args.loudnorm_i,
        "loudnorm_tp": args.loudnorm_tp,
        "loudnorm_lra": args.loudnorm_lra,
        "audio_bitrate": args.audio_bitrate,
        "filler_pad_ms": args.filler_pad_ms,
        "max_filler_len_ms": args.max_filler_len_ms,
        "filler_regex": args.filler_regex
    }
    
    # Configurações de qualidade de vídeo
    video_quality = {
        "mode": args.quality_mode,
        "video_crf": args.video_crf,
        "video_preset": args.video_preset,
        "target_bitrate": args.target_bitrate,
        "pix_fmt": args.pix_fmt if args.pix_fmt != "input" else None,
        "keep_fps_res": args.keep_fps_res,
        "audio_only": args.audio_only,
        "audio_bitrate": args.audio_bitrate
    }
    
    # Configurações de fillers
    fillers_cfg = {
        "filler_pad_ms": args.filler_pad_ms,
        "max_filler_len_ms": args.max_filler_len_ms,
        "filler_regex": args.filler_regex or r"\b(e+|ee+|eh+|h[au]+|hum+m+|hmm+|ah+|uh+|u+m+)\b"
    }
    
    try:
        with AutoEditor(config) as editor:
            # 1. Carregar mídia
            media_info = load_media(args.input)
            original_duration = media_info["meta"]["duration"]
            editor.temp_files.append(media_info["audio_wav_path"])

            # Modo somente-legenda: gerar SRT e sair (sem renderização de vídeo)
            if args.srt_only:
                logger.info("Modo somente-legenda: gerando SRT e finalizando")
                vad_segments = run_vad(
                    media_info["audio_wav_path"],
                    media_info["sr"],
                    config["vad_aggressiveness"],
                    config["min_speech_ms"],
                    config["min_gap_ms"],
                    config["speech_padding_ms"]
                )
                sentences = transcribe_segments(
                    media_info["audio_wav_path"],
                    vad_segments,
                    config["language"],
                    config["whisper_model_size"],
                    config["silence_split_ms"],
                    True
                )
                if sentences:
                    srt_path = os.path.splitext(args.output)[0] + ".srt"
                    write_srt(sentences, srt_path)
                    logger.info(f"Legendas salvas: {srt_path}")
                else:
                    logger.warning("Nenhum conteúdo transcrito; nada para salvar em SRT")
                return 0
            
            # Modo áudio-apenas: pular VAD/transcrição
            if config["audio_only"]:
                logger.info("Modo áudio-apenas: melhorando áudio sem cortes")
                
                if config["audio_enhance"]:
                    # Melhorar áudio
                    enhanced_audio_path = tempfile.mktemp(suffix=".wav")
                    editor.temp_files.append(enhanced_audio_path)
                    enhance_audio(media_info["audio_wav_path"], enhanced_audio_path, config)
                    
                    # Renderizar vídeo com áudio melhorado
                    render_video(
                        args.input,
                        [(0, original_duration)],  # Intervalo completo
                        args.output,
                        media_info["meta"],
                        video_quality,
                        config["audio_fade_ms"],
                        enhanced_audio_path
                    )
                else:
                    # Apenas copiar arquivo
                    import shutil
                    shutil.copy2(args.input, args.output)
                
                logger.info("Processamento áudio-apenas concluído")
                return 0
            
            # 2. Detecção de voz
            vad_segments = run_vad(
                media_info["audio_wav_path"],
                media_info["sr"],
                config["vad_aggressiveness"],
                config["min_speech_ms"],
                config["min_gap_ms"],
                config["speech_padding_ms"]
            )
            
            # 3. Transcrição
            sentences = transcribe_segments(
                media_info["audio_wav_path"],
                vad_segments,
                config["language"],
                config["whisper_model_size"],
                config["silence_split_ms"],
                True  # word_level
            )
            
            # 4. Detecção de fillers
            filler_intervals = []
            if sentences:
                filler_intervals = detect_fillers(sentences, fillers_cfg)
            
            # 5. Detecção de regravações
            excluded_indices = set()
            if sentences:
                excluded_indices = detect_retakes(
                    sentences,
                    config["repeat_window_s"],
                    config["repeat_similarity"]
                )
            
            # 6. Timeline final
            final_intervals = build_final_timeline(
                vad_segments,
                excluded_indices,
                sentences,
                config["join_gap_ms"]
            )
            
            # 7. Melhorar áudio se solicitado
            audio_wav_final = None
            if config["audio_enhance"]:
                audio_wav_final = tempfile.mktemp(suffix=".wav")
                editor.temp_files.append(audio_wav_final)
                enhance_audio(media_info["audio_wav_path"], audio_wav_final, config)
            
            # 8. Renderizar vídeo
            render_video(
                args.input,
                final_intervals,
                args.output,
                media_info["meta"],
                video_quality,
                config["audio_fade_ms"],
                audio_wav_final
            )
            
            # 9. Gerar SRT se solicitado
            if args.write_srt and sentences:
                srt_path = os.path.splitext(args.output)[0] + ".srt"
                # Filtrar frases mantidas
                sentences_kept = [s for i, s in enumerate(sentences) if i not in excluded_indices]
                write_srt(sentences_kept, srt_path)
            
            # Estatísticas finais
            final_duration = sum(end - start for start, end in final_intervals)
            cuts_made = len(vad_segments) - len(final_intervals)
            retakes_removed = len(excluded_indices)
            fillers_removed = len(filler_intervals)
            
            logger.info("=" * 50)
            logger.info("PROCESSAMENTO CONCLUÍDO")
            logger.info("=" * 50)
            logger.info(f"Duração original: {original_duration:.2f}s")
            logger.info(f"Duração final: {final_duration:.2f}s")
            logger.info(f"Tempo removido: {original_duration - final_duration:.2f}s")
            logger.info(f"Segmentos cortados: {cuts_made}")
            logger.info(f"Regravações removidas: {retakes_removed}")
            logger.info(f"Fillers removidos: {fillers_removed}")
            logger.info(f"Arquivo salvo: {args.output}")
            if args.write_srt and sentences:
                logger.info(f"Legendas salvas: {srt_path}")
            
            return 0
            
    except Exception as e:
        logger.error(f"Erro durante processamento: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
