import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache
import json
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TipoPad(Enum):
    SINE = "sine"
    SAW = "saw"
    SQUARE = "square"
    TRIANGLE = "triangle"
    PULSE = "pulse"
    STRING = "string"
    TRIBAL_PULSE = "tribal_pulse"
    SHIMMER = "shimmer"
    FADE_BLEND = "fade_blend"
    DIGITAL_SINE = "digital_sine"
    CENTER_PAD = "center_pad"
    DUST_PAD = "dust_pad"
    ICE_STRING = "ice_string"
    NEUROMORPHIC = "neuromorphic"
    BIOACOUSTIC = "bioacoustic"
    CRYSTALLINE = "crystalline"
    ORGANIC_FLOW = "organic_flow"
    QUANTUM_PAD = "quantum_pad"
    HARMONIC_SERIES = "harmonic_series"
    METALLIC = "metallic"
    VOCAL_PAD = "vocal_pad"
    GRANULAR = "granular"
    SPECTRAL = "spectral"
    FRACTAL = "fractal"

class EstiloRuido(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    FRACTAL = "fractal"
    ORGANIC = "organic"
    DIGITAL = "digital"
    NEURAL = "neural"
    ATMOSPHERIC = "atmospheric"
    GRANULAR = "granular"
    SPECTRAL = "spectral"
    CHAOS = "chaos"
    BROWN = "brown"
    PINK = "pink"
    WHITE = "white"
    BLUE = "blue"
    VIOLET = "violet"

class TipoFiltro(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    ALLPASS = "allpass"
    MORPHING = "morphing"
    NEUROMORPHIC = "neuromorphic"
    SPECTRAL = "spectral"
    FORMANT = "formant"
    COMB = "comb"
    PHASER = "phaser"
    FLANGER = "flanger"
    CHORUS = "chorus"
    REVERB = "reverb"

class CategoriaEstilo(Enum):
    MEDITATIVO = "meditativo"
    ENERGIZANTE = "energizante"
    CREATIVO = "creativo"
    TERAPEUTICO = "terapeutico"
    AMBIENTAL = "ambiental"
    EXPERIMENTAL = "experimental"
    TRADICIONAL = "tradicional"
    FUTURISTA = "futurista"
    ORGANICO = "organico"
    ESPIRITUAL = "espiritual"

@dataclass
class ConfiguracionFiltroAvanzada:
    tipo: TipoFiltro = TipoFiltro.LOWPASS
    frecuencia_corte: float = 500.0
    resonancia: float = 0.7
    sample_rate: int = 44100
    drive: float = 0.0
    wetness: float = 1.0
    modulacion_cutoff: bool = False
    velocidad_modulacion: float = 0.5
    profundidad_modulacion: float = 0.3
    envelope_follow: bool = False
    filtros_serie: List['ConfiguracionFiltroAvanzada'] = field(default_factory=list)
    mezcla_paralela: bool = False
    formant_frequencies: List[float] = field(default_factory=list)
    comb_delay_ms: float = 10.0
    phaser_stages: int = 4
    chorus_voices: int = 3
    reverb_room_size: float = 0.7
    reverb_decay: float = 0.5

@dataclass
class ConfiguracionRuidoAvanzada:
    duracion_sec: float
    estilo: EstiloRuido
    amplitud: float = 0.3
    sample_rate: int = 44100
    anchura_estereo: float = 1.0
    attack_ms: int = 100
    decay_ms: int = 200
    sustain_level: float = 0.8
    release_ms: int = 500
    filtro: Optional[ConfiguracionFiltroAvanzada] = None
    banda_frecuencia_min: float = 20.0
    banda_frecuencia_max: float = 20000.0
    granularidad: float = 1.0
    densidad: float = 1.0
    modulacion_temporal: bool = False
    patron_temporal: str = "constante"
    movimiento_espacial: bool = False
    patron_movimiento: str = "estatico"
    velocidad_movimiento: float = 0.1
    
    def __post_init__(self):
        if self.filtro is None:
            self.filtro = ConfiguracionFiltroAvanzada()

@dataclass
class ParametrosEspacialesAvanzados:
    pan: float = 0.0
    elevation: float = 0.0
    distance: float = 1.0
    width: float = 1.0
    movimiento_activo: bool = False
    tipo_movimiento: str = "circular"
    velocidad_movimiento: float = 0.1
    radio_movimiento: float = 1.0
    reverb_espacial: bool = False
    early_reflections: bool = False
    air_absorption: bool = False
    doppler_effect: bool = False

@dataclass
class PerfilEstiloCompleto:
    nombre: str
    categoria: CategoriaEstilo
    tipo_pad: TipoPad
    frecuencia_base: float = 220.0
    armónicos: List[float] = field(default_factory=list)
    nivel_db: float = -12.0
    modulacion_am: float = 0.0
    modulacion_fm: float = 0.0
    velocidad_modulacion: float = 0.5
    filtro: ConfiguracionFiltroAvanzada = field(default_factory=ConfiguracionFiltroAvanzada)
    ruido: Optional[ConfiguracionRuidoAvanzada] = None
    attack_ms: int = 100
    decay_ms: int = 500
    sustain_level: float = 0.7
    release_ms: int = 1000
    espacial: ParametrosEspacialesAvanzados = field(default_factory=ParametrosEspacialesAvanzados)
    neurotransmisores_asociados: List[str] = field(default_factory=list)
    efectos_emocionales: List[str] = field(default_factory=list)
    estados_mentales: List[str] = field(default_factory=list)
    descripcion: str = ""
    inspiracion: str = ""
    uso_recomendado: List[str] = field(default_factory=list)
    compatibilidad: List[str] = field(default_factory=list)
    incompatibilidad: List[str] = field(default_factory=list)
    complejidad_armonica: float = 0.5
    dinamismo_temporal: float = 0.5
    textura_espectral: str = "suave"
    validado: bool = False
    confidence_score: float = 0.0
    version: str = "v7.0"
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if self.frecuencia_base <= 0:
            raise ValueError("Frecuencia base debe ser positiva")
        if not self.armónicos:
            self._calcular_armonicos_automaticos()
    
    def _calcular_armonicos_automaticos(self):
        num_armonicos = int(3 + self.complejidad_armonica * 5)
        if self.tipo_pad == TipoPad.SINE:
            self.armónicos = [self.frecuencia_base * (2*i + 1) for i in range(num_armonicos)]
        elif self.tipo_pad == TipoPad.SAW:
            self.armónicos = [self.frecuencia_base * (i + 1) for i in range(num_armonicos)]
        elif self.tipo_pad == TipoPad.SQUARE:
            self.armónicos = [self.frecuencia_base * (2*i + 1) for i in range(num_armonicos)]
        elif self.tipo_pad == TipoPad.TRIANGLE:
            self.armónicos = [self.frecuencia_base * (2*i + 1) / ((2*i + 1)**2) for i in range(num_armonicos)]
        else:
            self.armónicos = [self.frecuencia_base * (i + 1) for i in range(num_armonicos)]

class GestorPerfilesEstilo:
    def __init__(self):
        self.perfiles = {}
        self.cache_generacion = {}
        self._inicializar_perfiles_predefinidos()
    
    def _inicializar_perfiles_predefinidos(self):
        perfiles_data = {
            "sereno": {
                "categoria": CategoriaEstilo.MEDITATIVO, "tipo_pad": TipoPad.SINE, "frecuencia_base": 6.0,
                "armónicos": [12.0, 18.0, 24.0], "nivel_db": -15.0, "modulacion_am": 0.1, "velocidad_modulacion": 0.05,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.LOWPASS, 400.0, 0.3, modulacion_cutoff=True, velocidad_modulacion=0.02),
                "attack_ms": 2000, "decay_ms": 3000, "sustain_level": 0.9, "release_ms": 5000,
                "espacial": ParametrosEspacialesAvanzados(width=2.0, reverb_espacial=True),
                "neurotransmisores_asociados": ["gaba", "serotonina", "melatonina"],
                "efectos_emocionales": ["calma_profunda", "serenidad", "paz_interior"],
                "estados_mentales": ["meditativo", "contemplativo", "relajado"],
                "uso_recomendado": ["meditación", "relajación", "terapia", "sueño"],
                "complejidad_armonica": 0.2, "dinamismo_temporal": 0.1, "textura_espectral": "suave",
                "validado": True, "confidence_score": 0.95
            },
            "organico": {
                "categoria": CategoriaEstilo.ORGANICO, "tipo_pad": TipoPad.ORGANIC_FLOW, "frecuencia_base": 8.5,
                "nivel_db": -10.0, "modulacion_am": 0.3, "modulacion_fm": 0.2, "velocidad_modulacion": 0.15,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.MORPHING, 800.0, 0.6, modulacion_cutoff=True, velocidad_modulacion=0.1),
                "ruido": ConfiguracionRuidoAvanzada(1.0, EstiloRuido.ORGANIC, 0.15, granularidad=0.7, patron_temporal="respiratorio"),
                "espacial": ParametrosEspacialesAvanzados(movimiento_activo=True, tipo_movimiento="pendulo", velocidad_movimiento=0.03),
                "neurotransmisores_asociados": ["anandamida", "serotonina", "oxitocina"],
                "efectos_emocionales": ["conexion_natural", "fluidez", "armonia"],
                "estados_mentales": ["flujo", "creatividad", "conexion"],
                "complejidad_armonica": 0.6, "dinamismo_temporal": 0.4, "textura_espectral": "orgánica",
                "validado": True, "confidence_score": 0.88
            },
            "etereo": {
                "categoria": CategoriaEstilo.ESPIRITUAL, "tipo_pad": TipoPad.SPECTRAL, "frecuencia_base": 10.0,
                "nivel_db": -12.0, "modulacion_am": 0.4, "modulacion_fm": 0.3, "velocidad_modulacion": 0.08,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.REVERB, reverb_room_size=0.9, reverb_decay=0.8, wetness=0.7),
                "espacial": ParametrosEspacialesAvanzados(elevation=0.5, width=1.8, movimiento_activo=True, tipo_movimiento="espiral", velocidad_movimiento=0.02),
                "neurotransmisores_asociados": ["anandamida", "serotonina", "dopamina"],
                "efectos_emocionales": ["expansion", "transcendencia", "libertad"],
                "estados_mentales": ["expandido", "elevado", "conectado"],
                "complejidad_armonica": 0.7, "dinamismo_temporal": 0.3, "textura_espectral": "etérea",
                "validado": True, "confidence_score": 0.90
            },
            "alienigena": {
                "categoria": CategoriaEstilo.EXPERIMENTAL, "tipo_pad": TipoPad.QUANTUM_PAD, "frecuencia_base": 13.2,
                "nivel_db": -8.0, "modulacion_fm": 0.8, "velocidad_modulacion": 1.5,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.SPECTRAL, 1200.0, 1.2, drive=0.4, modulacion_cutoff=True, velocidad_modulacion=0.3),
                "ruido": ConfiguracionRuidoAvanzada(1.0, EstiloRuido.CHAOS, 0.2, granularidad=0.3, modulacion_temporal=True),
                "espacial": ParametrosEspacialesAvanzados(movimiento_activo=True, tipo_movimiento="aleatorio", velocidad_movimiento=0.2, doppler_effect=True),
                "neurotransmisores_asociados": ["dopamina", "noradrenalina"],
                "efectos_emocionales": ["curiosidad", "extrañeza", "exploracion"],
                "estados_mentales": ["alterado", "experimental", "curioso"],
                "complejidad_armonica": 0.9, "dinamismo_temporal": 0.8, "textura_espectral": "digital",
                "validado": True, "confidence_score": 0.82
            },
            "tribal": {
                "categoria": CategoriaEstilo.ENERGIZANTE, "tipo_pad": TipoPad.TRIBAL_PULSE, "frecuencia_base": 12.0,
                "nivel_db": -9.0, "modulacion_am": 0.6, "velocidad_modulacion": 0.4,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.BANDPASS, 600.0, 0.8),
                "espacial": ParametrosEspacialesAvanzados(movimiento_activo=True, tipo_movimiento="circular", velocidad_movimiento=0.1),
                "neurotransmisores_asociados": ["dopamina", "adrenalina", "endorfina"],
                "efectos_emocionales": ["energia", "fuerza", "conexion_tribal"],
                "estados_mentales": ["energizado", "conectado", "poderoso"],
                "complejidad_armonica": 0.5, "dinamismo_temporal": 0.7, "textura_espectral": "percusiva",
                "validado": True, "confidence_score": 0.91
            },
            "mistico": {
                "categoria": CategoriaEstilo.ESPIRITUAL, "tipo_pad": TipoPad.SHIMMER, "frecuencia_base": 7.83,
                "nivel_db": -11.0, "modulacion_am": 0.3, "modulacion_fm": 0.4, "velocidad_modulacion": 0.12,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.CHORUS, chorus_voices=4, wetness=0.6),
                "neurotransmisores_asociados": ["anandamida", "serotonina", "oxitocina"],
                "efectos_emocionales": ["misterio", "conexion_universal", "reverencia"],
                "estados_mentales": ["mistico", "expandido", "reverente"],
                "complejidad_armonica": 0.6, "dinamismo_temporal": 0.4,
                "validado": True, "confidence_score": 0.87
            },
            "futurista": {
                "categoria": CategoriaEstilo.FUTURISTA, "tipo_pad": TipoPad.DIGITAL_SINE, "frecuencia_base": 14.5,
                "nivel_db": -7.0, "modulacion_fm": 0.7, "velocidad_modulacion": 0.8,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.PHASER, phaser_stages=6, modulacion_cutoff=True, velocidad_modulacion=0.4),
                "ruido": ConfiguracionRuidoAvanzada(1.0, EstiloRuido.DIGITAL, 0.1, granularidad=0.2),
                "neurotransmisores_asociados": ["dopamina", "acetilcolina", "noradrenalina"],
                "efectos_emocionales": ["innovacion", "precision", "futuro"],
                "estados_mentales": ["tecnologico", "preciso", "avanzado"],
                "complejidad_armonica": 0.8, "dinamismo_temporal": 0.9, "textura_espectral": "digital",
                "validado": True, "confidence_score": 0.85
            },
            "crystalline": {
                "categoria": CategoriaEstilo.AMBIENTAL, "tipo_pad": TipoPad.CRYSTALLINE, "frecuencia_base": 11.0,
                "nivel_db": -13.0, "modulacion_am": 0.2, "velocidad_modulacion": 0.06,
                "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.HIGHPASS, 200.0, 0.9, drive=0.1),
                "espacial": ParametrosEspacialesAvanzados(elevation=0.3, width=1.5, early_reflections=True),
                "neurotransmisores_asociados": ["acetilcolina", "serotonina"],
                "efectos_emocionales": ["claridad", "pureza", "precision"],
                "estados_mentales": ["claro", "enfocado", "puro"],
                "complejidad_armonica": 0.4, "dinamismo_temporal": 0.2, "textura_espectral": "cristalina",
                "validado": True, "confidence_score": 0.92
            }
        }
        
        for nombre, data in perfiles_data.items():
            self.perfiles[nombre] = PerfilEstiloCompleto(nombre=nombre.title(), **data)
    
    @lru_cache(maxsize=64)
    def obtener_perfil(self, nombre: str) -> Optional[PerfilEstiloCompleto]:
        return self.perfiles.get(nombre.lower())
    
    def obtener_perfiles_por_categoria(self, categoria: CategoriaEstilo) -> List[PerfilEstiloCompleto]:
        return [p for p in self.perfiles.values() if p.categoria == categoria]
    
    def obtener_perfiles_por_neurotransmisor(self, neurotransmisor: str) -> List[PerfilEstiloCompleto]:
        return [p for p in self.perfiles.values() if neurotransmisor.lower() in [n.lower() for n in p.neurotransmisores_asociados]]
    
    def obtener_perfiles_por_estado_mental(self, estado: str) -> List[PerfilEstiloCompleto]:
        return [p for p in self.perfiles.values() if any(estado.lower() in em.lower() for em in p.estados_mentales)]
    
    def generar_perfil_personalizado(self, nombre: str, categoria: CategoriaEstilo, neurotransmisores: List[str], complejidad: float = 0.5) -> PerfilEstiloCompleto:
        freq_base = self._calcular_frecuencia_de_neurotransmisores(neurotransmisores)
        tipo_pad = self._seleccionar_tipo_pad_por_categoria(categoria)
        filtro = self._configurar_filtro_por_complejidad(complejidad)
        
        return PerfilEstiloCompleto(
            nombre=nombre, categoria=categoria, tipo_pad=tipo_pad, frecuencia_base=freq_base,
            neurotransmisores_asociados=neurotransmisores, complejidad_armonica=complejidad,
            dinamismo_temporal=complejidad * 0.8, filtro=filtro, validado=False, confidence_score=0.6
        )
    
    def _calcular_frecuencia_de_neurotransmisores(self, neurotransmisores: List[str]) -> float:
        mapeo_freq = {"gaba": 6.0, "serotonina": 7.5, "anandamida": 5.5, "oxitocina": 8.0, "dopamina": 12.0,
                      "acetilcolina": 14.0, "noradrenalina": 13.5, "adrenalina": 13.8, "endorfina": 10.5}
        frecuencias = [mapeo_freq.get(n.lower(), 10.0) for n in neurotransmisores]
        return sum(frecuencias) / len(frecuencias) if frecuencias else 10.0
    
    def _seleccionar_tipo_pad_por_categoria(self, categoria: CategoriaEstilo) -> TipoPad:
        mapeo = {CategoriaEstilo.MEDITATIVO: TipoPad.SINE, CategoriaEstilo.ENERGIZANTE: TipoPad.SAW,
                 CategoriaEstilo.CREATIVO: TipoPad.ORGANIC_FLOW, CategoriaEstilo.TERAPEUTICO: TipoPad.SINE,
                 CategoriaEstilo.AMBIENTAL: TipoPad.SPECTRAL, CategoriaEstilo.EXPERIMENTAL: TipoPad.QUANTUM_PAD,
                 CategoriaEstilo.TRADICIONAL: TipoPad.STRING, CategoriaEstilo.FUTURISTA: TipoPad.DIGITAL_SINE,
                 CategoriaEstilo.ORGANICO: TipoPad.ORGANIC_FLOW, CategoriaEstilo.ESPIRITUAL: TipoPad.SHIMMER}
        return mapeo.get(categoria, TipoPad.SINE)
    
    def _configurar_filtro_por_complejidad(self, complejidad: float) -> ConfiguracionFiltroAvanzada:
        if complejidad < 0.3:
            return ConfiguracionFiltroAvanzada(TipoFiltro.LOWPASS, 400.0, 0.3)
        elif complejidad < 0.7:
            return ConfiguracionFiltroAvanzada(TipoFiltro.BANDPASS, 800.0, 0.6, modulacion_cutoff=True)
        else:
            return ConfiguracionFiltroAvanzada(TipoFiltro.MORPHING, 1200.0, 0.9, modulacion_cutoff=True, drive=0.2)
    
    def analizar_compatibilidad(self, perfil1: str, perfil2: str) -> Dict[str, Any]:
        p1, p2 = self.obtener_perfil(perfil1), self.obtener_perfil(perfil2)
        if not p1 or not p2:
            return {"error": "Uno o ambos perfiles no encontrados"}
        
        diff_freq = abs(p1.frecuencia_base - p2.frecuencia_base)
        compatibilidad_freq = 1.0 - min(diff_freq / 10.0, 1.0)
        neuro_comunes = set(p1.neurotransmisores_asociados) & set(p2.neurotransmisores_asociados)
        compatibilidad_neuro = len(neuro_comunes) / max(len(p1.neurotransmisores_asociados), 1)
        compatibilidad_categoria = 1.0 if p1.categoria == p2.categoria else 0.5
        score_total = (compatibilidad_freq + compatibilidad_neuro + compatibilidad_categoria) / 3
        
        return {
            "score_compatibilidad": score_total,
            "diferencia_frecuencial": diff_freq,
            "neurotransmisores_comunes": list(neuro_comunes),
            "recomendacion": self._generar_recomendacion_compatibilidad(score_total),
            "sugerencias_mezcla": self._sugerir_parametros_mezcla(p1, p2, score_total)
        }
    
    def _generar_recomendacion_compatibilidad(self, score: float) -> str:
        if score > 0.8: return "Excelente compatibilidad - Ideal para mezclas largas"
        elif score > 0.6: return "Buena compatibilidad - Funciona bien en transiciones"
        elif score > 0.4: return "Compatibilidad moderada - Usar con precaución"
        else: return "Baja compatibilidad - No recomendado para mezcla directa"
    
    def _sugerir_parametros_mezcla(self, p1: PerfilEstiloCompleto, p2: PerfilEstiloCompleto, score: float) -> Dict[str, float]:
        if score > 0.7:
            return {"peso_perfil1": 0.5, "peso_perfil2": 0.5, "crossfade_ms": 2000}
        elif score > 0.5:
            peso_p1 = 0.7 if p1.complejidad_armonica < p2.complejidad_armonica else 0.3
            return {"peso_perfil1": peso_p1, "peso_perfil2": 1-peso_p1, "crossfade_ms": 3000}
        else:
            return {"peso_perfil1": 0.8, "peso_perfil2": 0.2, "crossfade_ms": 5000}
    
    def exportar_configuracion_aurora(self) -> Dict[str, Any]:
        return {
            "version": "v7.0", "total_perfiles": len(self.perfiles),
            "categorias_disponibles": [cat.value for cat in CategoriaEstilo],
            "tipos_pad_disponibles": [tipo.value for tipo in TipoPad],
            "tipos_filtro_disponibles": [filtro.value for filtro in TipoFiltro],
            "estilos_ruido_disponibles": [ruido.value for ruido in EstiloRuido],
            "perfiles_validados": sum(1 for p in self.perfiles.values() if p.validado),
            "confidence_promedio": sum(p.confidence_score for p in self.perfiles.values()) / len(self.perfiles)
        }
    
    def limpiar_cache(self):
        self.obtener_perfil.cache_clear()
        self.cache_generacion.clear()
        logger.info("Cache de perfiles de estilo limpiado")

class StyleProfile:
    _gestor = None
    
    @staticmethod
    def get(nombre: str) -> Dict[str, str]:
        if StyleProfile._gestor is None:
            StyleProfile._gestor = GestorPerfilesEstilo()
        
        perfil = StyleProfile._gestor.obtener_perfil(nombre)
        if perfil:
            return {"pad_type": perfil.tipo_pad.value}
        else:
            estilos_legacy = {
                "sereno": {"pad_type": "sine"}, "organico": {"pad_type": "square"},
                "etereo": {"pad_type": "saw"}, "alienigena": {"pad_type": "pulse"},
                "neoclasico": {"pad_type": "string"}, "minimalista_emocional": {"pad_type": "triangle"},
                "tribal": {"pad_type": "tribal_pulse"}, "mistico": {"pad_type": "shimmer"},
                "sutil": {"pad_type": "fade_blend"}, "futurista": {"pad_type": "digital_sine"},
                "centered": {"pad_type": "center_pad"}, "shimmer": {"pad_type": "shimmer"},
                "warm_dust": {"pad_type": "dust_pad"}, "crystalline": {"pad_type": "ice_string"}
            }
            return estilos_legacy.get(nombre.lower(), {"pad_type": "sine"})

class FilterConfig:
    def __init__(self, cutoff_freq=500.0, sample_rate=44100, filter_type="lowpass"):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.filter_type = filter_type

class NoiseStyle:
    DYNAMIC = "DYNAMIC"
    STATIC = "STATIC"
    FRACTAL = "FRACTAL"

class NoiseConfig:
    def __init__(self, duration_sec, noise_style, stereo_width=1.0, filter_config=None, sample_rate=44100, amplitude=0.3):
        self.duration_sec = duration_sec
        self.noise_style = noise_style
        self.stereo_width = stereo_width
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.filter_config = filter_config or FilterConfig()

def crear_gestor_estilos() -> GestorPerfilesEstilo:
    return GestorPerfilesEstilo()

def obtener_perfil_completo(nombre: str) -> Optional[PerfilEstiloCompleto]:
    return crear_gestor_estilos().obtener_perfil(nombre)

def generar_perfil_desde_neurotransmisores(neurotransmisores: List[str], nombre: str = "personalizado") -> PerfilEstiloCompleto:
    gestor = crear_gestor_estilos()
    
    if any(n in ["gaba", "serotonina", "melatonina"] for n in neurotransmisores):
        categoria = CategoriaEstilo.MEDITATIVO
    elif any(n in ["dopamina", "adrenalina", "noradrenalina"] for n in neurotransmisores):
        categoria = CategoriaEstilo.ENERGIZANTE
    elif any(n in ["anandamida", "oxitocina"] for n in neurotransmisores):
        categoria = CategoriaEstilo.ESPIRITUAL
    else:
        categoria = CategoriaEstilo.AMBIENTAL
    
    return gestor.generar_perfil_personalizado(nombre, categoria, neurotransmisores)

def obtener_perfiles_recomendados_para_estado(estado_mental: str) -> List[str]:
    gestor = crear_gestor_estilos()
    perfiles = gestor.obtener_perfiles_por_estado_mental(estado_mental)
    return [p.nombre.lower() for p in perfiles]