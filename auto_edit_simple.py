#!/usr/bin/env python3
"""
AUTO EDIT - Editor Automático de Vídeo (Versão Simplificada)
============================================================

Script para remover pausas sem fala e eliminar regravações de vídeos,
mantendo sempre a última ocorrência de frases repetidas.

Esta versão usa apenas ffmpeg diretamente para compatibilidade com Python 3.13.

INSTALAÇÃO:
1. Instale o ffmpeg no sistema
2. pip install moviepy==1.0.3 webrtcvad rapidfuzz faster-whisper numpy

USO:
python auto_edit_simple.py --input video.mp4 --output video_editado.mp4 --write_srt

AUTOR: Gerado automaticamente
VERSÃO: 1.1 (Compatível com Python 3.13)
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

# Verificar e importar dependências
try:
    import numpy as np
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    import webrtcvad
    from faster_whisper import WhisperModel
    from rapidfuzz import fuzz
    import rapidfuzz
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("\n📦 Para instalar as dependências necessárias, execute:")
    print("pip install moviepy==1.0.3 webrtcvad rapidfuzz faster-whisper numpy")
    print("\n💡 Certifique-se também de que o ffmpeg está instalado no sistema.")
    exit(1)

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


def load_media(input_path: str) -> Dict:
    """
    Carrega vídeo e extrai áudio mono 16kHz PCM usando ffmpeg diretamente.
    
    Args:
        input_path: Caminho para o arquivo de vídeo
        
    Returns:
        Dict com metadados do vídeo e caminho do áudio extraído
    """
    logger.info(f"Carregando mídia: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    
    # Verificar se ffmpeg está disponível
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg não encontrado. Instale o ffmpeg no sistema.")
    
    try:
        # Carregar vídeo com moviepy para obter metadados
        video = VideoFileClip(input_path)
        
        # Extrair metadados
        fps = video.fps
        width, height = video.size
        duration = video.duration
        
        # Obter codec de áudio
        audio_codec = None
        if hasattr(video, 'audio') and video.audio:
            audio_codec = getattr(video.audio, 'codec', 'unknown')
        
        video.close()
        
        # Extrair áudio com ffmpeg diretamente
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
        
        # Obter sample rate do arquivo extraído
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_streams", temp_audio_path
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode == 0:
            import json
            probe_data = json.loads(probe_result.stdout)
            sr = int(probe_data['streams'][0]['sample_rate'])
        else:
            sr = 16000  # fallback
        
        logger.info(f"Áudio extraído: {duration:.2f}s, {sr}Hz, mono")
        
        return {
            "audio_tmp_path": temp_audio_path,
            "sr": sr,
            "fps": fps,
            "width": width,
            "height": height,
            "codec_info": audio_codec,
            "duration": duration
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


def run_vad(audio_tmp_path: str, sr: int, vad_aggressiveness: int = 2, 
           min_speech_ms: int = 200, min_gap_ms: int = 150, 
           speech_padding_ms: int = 150) -> List[Tuple[float, float]]:
    """
    Executa detecção de atividade de voz (VAD) no áudio.
    
    Args:
        audio_tmp_path: Caminho para arquivo de áudio temporário
        sr: Sample rate do áudio
        vad_aggressiveness: Agressividade do VAD (0-3)
        min_speech_ms: Duração mínima de segmento de fala
        min_gap_ms: Duração mínima de gap entre segmentos
        speech_padding_ms: Padding para adicionar aos segmentos
        
    Returns:
        Lista de tuplas (start, end) em segundos
    """
    logger.info("Executando detecção de atividade de voz (VAD)...")
    
    # Carregar dados de áudio
    audio_data = load_audio_data(audio_tmp_path)
    
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


def transcribe_segments(audio_tmp_path: str, segments: List[Tuple[float, float]], 
                       language: str = "auto", model_size: str = "small") -> List[Dict]:
    """
    Transcreve segmentos de áudio usando Whisper.
    
    Args:
        audio_tmp_path: Caminho para arquivo de áudio
        segments: Lista de segmentos (start, end) em segundos
        language: Idioma para transcrição
        model_size: Tamanho do modelo Whisper
        
    Returns:
        Lista de frases com timestamps
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
                "ffmpeg", "-i", audio_tmp_path,
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
                    word_timestamps=True
                )
                
                # Processar resultados
                for segment_result in segments_result:
                    if segment_result.text.strip():
                        # Ajustar timestamps para o vídeo completo
                        adjusted_start = start + segment_result.start
                        adjusted_end = start + segment_result.end
                        
                        sentences.append({
                            "text": segment_result.text.strip(),
                            "start": adjusted_start,
                            "end": adjusted_end,
                            "confidence": getattr(segment_result, 'avg_logprob', 0.0)
                        })
                
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


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparação de similaridade.
    
    Args:
        text: Texto original
        
    Returns:
        Texto normalizado
    """
    # Converter para minúsculas
    text = text.lower().strip()
    
    # Remover pontuação excessiva
    text = re.sub(r'[.,!?;:]+', '', text)
    
    # Remover fillers comuns
    fillers = ['é...', 'ah...', 'uh...', 'hmm...', 'bem...', 'então...', 'assim...']
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
                output_path: str, fps: float, audio_fade_ms: int = 40) -> None:
    """
    Renderiza vídeo final mantendo apenas os intervalos especificados.
    
    Args:
        input_path: Caminho do vídeo original
        kept_intervals: Intervalos a manter
        output_path: Caminho do vídeo de saída
        fps: FPS do vídeo original
        audio_fade_ms: Duração do fade de áudio
    """
    logger.info("Renderizando vídeo final...")
    
    if not kept_intervals:
        logger.warning("Nenhum intervalo para renderizar")
        return
    
    try:
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
        
        # Escrever vídeo final
        logger.info(f"Salvando vídeo: {output_path}")
        final_video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
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
        description="Editor automático de vídeo - Remove silêncios e regravações (Versão Python 3.13)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python auto_edit_simple.py --input video.mp4 --output editado.mp4
  python auto_edit_simple.py --input video.mp4 --output editado.mp4 --write_srt --language pt
  python auto_edit_simple.py --input video.mp4 --output editado.mp4 --vad_aggressiveness 3 --repeat_similarity 85
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument("--input", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", required=True, help="Caminho do vídeo de saída")
    
    # Configurações de idioma e modelo
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
    
    # Configurações de renderização
    parser.add_argument("--audio_fade_ms", type=int, default=40,
                       help="Duração do fade de áudio em ms (default: 40)")
    parser.add_argument("--join_gap_ms", type=int, default=120,
                       help="Gap máximo para unir segmentos em ms (default: 120)")
    
    # Opções
    parser.add_argument("--write_srt", action="store_true",
                       help="Gerar arquivo SRT com legendas")
    parser.add_argument("--keep_silence_ms", type=int, default=0,
                       help="Manter silêncio residual em ms (default: 0)")
    parser.add_argument("--target_bitrate", type=str, default=None,
                       help="Bitrate de saída (ex: 1000k)")
    
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
        "keep_silence_ms": args.keep_silence_ms,
        "target_bitrate": args.target_bitrate
    }
    
    try:
        with AutoEditor(config) as editor:
            # 1. Carregar mídia
            media_info = load_media(args.input)
            original_duration = media_info["duration"]
            editor.temp_files.append(media_info["audio_tmp_path"])
            
            # 2. Detecção de voz
            vad_segments = run_vad(
                media_info["audio_tmp_path"],
                media_info["sr"],
                config["vad_aggressiveness"],
                config["min_speech_ms"],
                config["min_gap_ms"],
                config["speech_padding_ms"]
            )
            
            # 3. Transcrição
            sentences = transcribe_segments(
                media_info["audio_tmp_path"],
                vad_segments,
                config["language"],
                config["whisper_model_size"]
            )
            
            # 4. Detecção de regravações
            excluded_indices = set()
            if sentences:
                excluded_indices = detect_retakes(
                    sentences,
                    config["repeat_window_s"],
                    config["repeat_similarity"]
                )
            
            # 5. Timeline final
            final_intervals = build_final_timeline(
                vad_segments,
                excluded_indices,
                sentences,
                config["join_gap_ms"]
            )
            
            # 6. Renderizar vídeo
            render_video(
                args.input,
                final_intervals,
                args.output,
                media_info["fps"],
                config["audio_fade_ms"]
            )
            
            # 7. Gerar SRT se solicitado
            if args.write_srt and sentences:
                srt_path = os.path.splitext(args.output)[0] + ".srt"
                # Filtrar frases mantidas
                sentences_kept = [s for i, s in enumerate(sentences) if i not in excluded_indices]
                write_srt(sentences_kept, srt_path)
            
            # Estatísticas finais
            final_duration = sum(end - start for start, end in final_intervals)
            cuts_made = len(vad_segments) - len(final_intervals)
            retakes_removed = len(excluded_indices)
            
            logger.info("=" * 50)
            logger.info("PROCESSAMENTO CONCLUÍDO")
            logger.info("=" * 50)
            logger.info(f"Duração original: {original_duration:.2f}s")
            logger.info(f"Duração final: {final_duration:.2f}s")
            logger.info(f"Tempo removido: {original_duration - final_duration:.2f}s")
            logger.info(f"Segmentos cortados: {cuts_made}")
            logger.info(f"Regravações removidas: {retakes_removed}")
            logger.info(f"Arquivo salvo: {args.output}")
            if args.write_srt and sentences:
                logger.info(f"Legendas salvas: {srt_path}")
            
            return 0
            
    except Exception as e:
        logger.error(f"Erro durante processamento: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
