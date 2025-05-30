import numpy as np
from scipy.signal import butter, filtfilt
from typing import Dict, List, Tuple, Optional, Union
import logging
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EstiloEmocional(Enum):
    MISTICO = "mistico"
    ETEREO = "etereo" 
    TRIBAL = "tribal"
    GLITCH = "glitch"
    NEUTRA = "neutra"
    AMBIENTAL = "ambiental"
    INDUSTRIAL = "industrial"
    CELESTIAL = "celestial"

class TipoAcorde(Enum):
    MAYOR = "mayor"
    MENOR = "menor"
    SUSPENDIDO = "suspendido"
    SEPTIMA = "7"
    MENOR_NOVENA = "m9"
    MAYOR_SEPTIMA = "maj7"
    DISMINUIDO = "dim"
    AUMENTADO = "aug"

@dataclass
class ConfiguracionAudio:
    sample_rate: int = 44100
    duracion_ms: int = 1000
    gain_db: float = -12.0
    fade_in_ms: int = 50
    fade_out_ms: int = 50
    
    def __post_init__(self):
        if self.sample_rate <= 0 or self.duracion_ms <= 0:
            raise ValueError("Sample rate y duración deben ser positivos")
        if self.gain_db > 0:
            warnings.warn("Gain positivo puede causar distorsión")

class GeneradorArmonias:
    def __init__(self, config: Optional[ConfiguracionAudio] = None):
        self.config = config or ConfiguracionAudio()
        self._cache_ondas = {}
        self._escalas_cache = {}
        
        self._escalas_emocionales = {
            EstiloEmocional.MISTICO: [1, 6/5, 4/3, 3/2, 7/4, 2],
            EstiloEmocional.ETEREO: [1, 9/8, 5/4, 11/8, 3/2, 15/8],
            EstiloEmocional.TRIBAL: [1, 1.18, 1.33, 1.5, 1.78, 2],
            EstiloEmocional.GLITCH: [1, 1.1, 1.25, 1.42, 1.68, 1.92],
            EstiloEmocional.NEUTRA: [1, 5/4, 3/2, 2],
            EstiloEmocional.AMBIENTAL: [1, 1.125, 1.25, 1.5, 1.875, 2],
            EstiloEmocional.INDUSTRIAL: [1, 1.15, 1.4, 1.6, 1.85, 2.1],
            EstiloEmocional.CELESTIAL: [1, 1.059, 1.189, 1.335, 1.498, 1.682, 1.888, 2]
        }
        
        self._acordes = {
            TipoAcorde.MAYOR: [1, 5/4, 3/2], TipoAcorde.MENOR: [1, 6/5, 3/2],
            TipoAcorde.SUSPENDIDO: [1, 4/3, 3/2], TipoAcorde.SEPTIMA: [1, 5/4, 3/2, 7/4],
            TipoAcorde.MENOR_NOVENA: [1, 6/5, 3/2, 9/5], TipoAcorde.MAYOR_SEPTIMA: [1, 5/4, 3/2, 15/8],
            TipoAcorde.DISMINUIDO: [1, 6/5, 64/45], TipoAcorde.AUMENTADO: [1, 5/4, 8/5]
        }
    
    @lru_cache(maxsize=128)
    def _generar_onda_base(self, freq: float, duracion_samples: int, gain_db: float) -> np.ndarray:
        try:
            t = np.linspace(0, duracion_samples / self.config.sample_rate, duracion_samples, endpoint=False)
            return np.sin(2 * np.pi * freq * t) * (10 ** (gain_db / 20))
        except Exception as e:
            logger.error(f"Error generando onda: {e}")
            return np.zeros(duracion_samples)
    
    def _aplicar_envelope(self, signal: np.ndarray) -> np.ndarray:
        fade_in_s = int(self.config.fade_in_ms * self.config.sample_rate / 1000)
        fade_out_s = int(self.config.fade_out_ms * self.config.sample_rate / 1000)
        
        if len(signal) <= fade_in_s + fade_out_s:
            return signal
        
        signal[:fade_in_s] *= np.linspace(0, 1, fade_in_s)
        signal[-fade_out_s:] *= np.linspace(1, 0, fade_out_s)
        return signal
    
    def normalizar_seguro(self, wave: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
        if len(wave) == 0:
            return wave
        max_val = np.max(np.abs(wave))
        return wave if max_val == 0 else (wave / max_val) * (10 ** (headroom_db / 20))
    
    def aplicar_filtro_inteligente(self, signal: np.ndarray, cutoff: float = 800.0, 
                                 filter_type: str = 'lowpass', order: int = 6) -> np.ndarray:
        try:
            nyquist = self.config.sample_rate * 0.5
            if cutoff >= nyquist:
                cutoff = nyquist * 0.95
            b, a = butter(order, cutoff / nyquist, btype=filter_type)
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.error(f"Error en filtrado: {e}")
            return signal
    
    def obtener_frecuencias_escala(self, base_freq: float, estilo: EstiloEmocional) -> List[float]:
        if estilo not in self._escalas_emocionales:
            estilo = EstiloEmocional.NEUTRA
        return [base_freq * ratio for ratio in self._escalas_emocionales[estilo]]
    
    def obtener_frecuencias_acorde(self, base_freq: float, tipo: TipoAcorde) -> List[float]:
        if tipo not in self._acordes:
            tipo = TipoAcorde.MAYOR
        return [base_freq * ratio for ratio in self._acordes[tipo]]
    
    def generar_pad_avanzado(self, base_freq: float = 220.0, estilo: EstiloEmocional = EstiloEmocional.MISTICO,
                           modulacion: bool = True, stereo: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        try:
            duracion_samples = int(self.config.sample_rate * self.config.duracion_ms / 1000)
            frecuencias = self.obtener_frecuencias_escala(base_freq, estilo)
            
            capas = [self._generar_onda_base(freq, duracion_samples, self.config.gain_db - (i * 2)) 
                    for i, freq in enumerate(frecuencias)]
            pad_combinado = np.sum(capas, axis=0)
            
            if modulacion:
                t = np.linspace(0, self.config.duracion_ms / 1000, duracion_samples)
                mod1 = 0.9 + 0.1 * np.sin(2 * np.pi * 0.2 * t)
                mod2 = 1.0 + 0.05 * np.sin(2 * np.pi * 0.7 * t)
                pad_combinado *= mod1 * mod2
            
            pad_combinado = self.normalizar_seguro(self._aplicar_envelope(pad_combinado))
            
            if stereo:
                pad_derecho = self.aplicar_filtro_inteligente(pad_combinado, cutoff=1200)
                pad_izquierdo = self.aplicar_filtro_inteligente(pad_combinado, cutoff=900)
                return pad_izquierdo, pad_derecho
            
            return pad_combinado
        except Exception as e:
            logger.error(f"Error generando pad: {e}")
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
    
    def generar_acorde_pad(self, base_freq: float = 220.0, tipo: TipoAcorde = TipoAcorde.MAYOR,
                          voicing: str = "close") -> np.ndarray:
        try:
            duracion_samples = int(self.config.sample_rate * self.config.duracion_ms / 1000)
            frecuencias_base = self.obtener_frecuencias_acorde(base_freq, tipo)
            
            if voicing == "open":
                frecuencias = frecuencias_base + [f * 2 for f in frecuencias_base[1:]]
            elif voicing == "drop2" and len(frecuencias_base) > 1:
                frecuencias = frecuencias_base.copy()
                frecuencias[1] /= 2
            else:
                frecuencias = frecuencias_base
            
            capas = [self._generar_onda_base(freq, duracion_samples, self.config.gain_db - (i * 1.5)) 
                    for i, freq in enumerate(frecuencias)]
            acorde_combinado = self.normalizar_seguro(self._aplicar_envelope(np.sum(capas, axis=0)))
            return acorde_combinado
        except Exception as e:
            logger.error(f"Error generando acorde: {e}")
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
    
    def mezclar_elementos(self, elementos: List[np.ndarray], pesos: Optional[List[float]] = None) -> np.ndarray:
        if not elementos:
            return np.array([])
        
        longitud_min = min(len(elemento) for elemento in elementos)
        elementos_norm = [elemento[:longitud_min] for elemento in elementos]
        
        if pesos and len(pesos) == len(elementos):
            elementos_norm = [elem * peso for elem, peso in zip(elementos_norm, pesos)]
        
        return self.normalizar_seguro(np.sum(elementos_norm, axis=0))
    
    def generar_textura_compleja(self, base_freq: float = 220.0, num_capas: int = 5,
                               spread_octavas: float = 2.0) -> np.ndarray:
        try:
            estilos = list(EstiloEmocional)
            capas = []
            
            for i in range(num_capas):
                factor_freq = 1 + (i / (num_capas - 1)) * (spread_octavas - 1)
                freq_capa = base_freq * factor_freq
                estilo = estilos[i % len(estilos)]
                
                pad_capa = self.generar_pad_avanzado(base_freq=freq_capa, estilo=estilo, modulacion=True)
                cutoff = 2000 - (i * 300)
                pad_filtrado = self.aplicar_filtro_inteligente(pad_capa, cutoff=cutoff)
                capas.append(pad_filtrado)
            
            pesos = [1 / (i + 1) for i in range(num_capas)]
            return self.mezclar_elementos(capas, pesos)
        except Exception as e:
            logger.error(f"Error generando textura: {e}")
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
    
    def exportar_configuracion(self) -> Dict:
        return {
            "sample_rate": self.config.sample_rate, "duracion_ms": self.config.duracion_ms,
            "gain_db": self.config.gain_db, "escalas_disponibles": [e.value for e in EstiloEmocional],
            "acordes_disponibles": [t.value for t in TipoAcorde], "version": "v7_aurora_integrated"
        }
    
    def limpiar_cache(self):
        self._generar_onda_base.cache_clear()
        self._cache_ondas.clear()
        self._escalas_cache.clear()

def crear_generador_aurora(sample_rate: int = 44100, duracion_ms: int = 4000) -> GeneradorArmonias:
    config = ConfiguracionAudio(sample_rate=sample_rate, duracion_ms=duracion_ms, 
                               gain_db=-15.0, fade_in_ms=100, fade_out_ms=200)
    return GeneradorArmonias(config)

def generar_pack_emocional(base_freq: float = 220.0) -> Dict[str, np.ndarray]:
    generador = crear_generador_aurora(duracion_ms=6000)
    return {estilo.value: generador.generar_pad_avanzado(base_freq=base_freq, estilo=estilo, modulacion=True) 
            for estilo in EstiloEmocional}