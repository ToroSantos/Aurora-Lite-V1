import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import math, warnings
from datetime import datetime

class TipoEnvolvente(Enum):
    SUAVE = "suave"; CELESTIAL = "celestial"; INQUIETANTE = "inquietante"; GRAVE = "grave"; PLANA = "plana"
    LUMINOSA = "luminosa"; RITMICA = "ritmica"; VIBRANTE = "vibrante"; ETEREA = "eterea"; PULIDA = "pulida"
    LIMPIA = "limpia"; CALIDA = "calida"; TRANSPARENTE = "transparente"; BRILLANTE = "brillante"; FOCAL = "focal"
    NEUROMORFICA = "neuromorfica"; ORGANICA = "organica"; CUANTICA = "cuantica"; CRISTALINA = "cristalina"
    FLUIDA = "fluida"; ESPIRAL = "espiral"; FRACTAL = "fractal"; HOLOGRAFICA = "holografica"

class EscalaArmonica(Enum):
    SOLFEGGIO = "solfeggio"; EQUAL_TEMPERAMENT = "equal_temperament"; JUST_INTONATION = "just_intonation"
    PYTHAGOREAN = "pythagorean"; GOLDEN_RATIO = "golden_ratio"; FIBONACCI = "fibonacci"
    SCHUMANN = "schumann"; PLANETARY = "planetary"; CHAKRA = "chakra"; CUSTOM = "custom"

class PatronPan(Enum):
    ESTATICO = "estatico"; PENDULO = "pendulo"; CIRCULAR = "circular"; ESPIRAL = "espiral"
    RESPIRATORIO = "respiratorio"; CARDIACO = "cardiaco"; ALEATORIO = "aleatorio"
    ONDULATORIO = "ondulatorio"; ORBITAL = "orbital"; CUANTICO = "cuantico"

@dataclass
class ParametrosFade:
    fade_in_sec: float; fade_out_sec: float; curva_fade_in: str = "exponential"; curva_fade_out: str = "exponential"
    micro_fades: bool = False; fade_crossover: bool = False; adaptativo: bool = False
    def __post_init__(self):
        if self.fade_in_sec < 0 or self.fade_out_sec < 0: raise ValueError("Tiempos de fade deben ser positivos")

@dataclass
class ConfiguracionPanorama:
    dinamico: bool; patron: PatronPan = PatronPan.ESTATICO; velocidad: float = 0.1; amplitud: float = 1.0
    centro: float = 0.0; sincronizacion: str = "libre"; elevacion: float = 0.0; profundidad: float = 0.0
    rotacion_3d: bool = False; doppler_effect: bool = False

@dataclass
class EspectroArmonico:
    frecuencias_base: List[float]; escala: EscalaArmonica = EscalaArmonica.EQUAL_TEMPERAMENT
    temperamento: str = "440hz"; armonicos_naturales: List[float] = field(default_factory=list)
    armonicos_artificiales: List[float] = field(default_factory=list); subharmonicos: List[float] = field(default_factory=list)
    modulacion_armonica: bool = False; velocidad_modulacion: float = 0.1; profundidad_modulacion: float = 0.2
    patron_modulacion: str = "senoidal"; razon_aurea: bool = False; fibonacci_sequence: bool = False
    just_ratios: List[Tuple[int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        for f in self.frecuencias_base:
            if f <= 0: raise ValueError(f"Frecuencia debe ser positiva: {f}")
        if not self.armonicos_naturales:
            for fb in self.frecuencias_base: self.armonicos_naturales.extend([fb * i for i in range(2, 8)])
        if self.razon_aurea and not self.armonicos_artificiales:
            phi = (1 + math.sqrt(5)) / 2
            for fb in self.frecuencias_base: self.armonicos_artificiales.extend([fb * phi, fb * (phi ** 2), fb / phi])
        if self.fibonacci_sequence and not self.subharmonicos:
            fib = [1, 1, 2, 3, 5, 8, 13, 21]
            for fb in self.frecuencias_base: self.subharmonicos.extend([fb / f for f in fib[2:6]])

@dataclass
class PropiedadesTextura:
    rugosidad: float = 0.0; granularidad: float = 0.0; densidad: float = 1.0; brillantez: float = 0.5
    calidez: float = 0.5; fluctuacion: float = 0.0; respiracion: bool = False; pulsacion: float = 0.0
    amplitud_espacial: float = 1.0; cohesion: float = 0.8; dispersion: float = 0.2
    def __post_init__(self):
        for attr in ['rugosidad', 'granularidad', 'brillantez', 'calidez', 'fluctuacion']:
            if not 0 <= getattr(self, attr) <= 1: warnings.warn(f"{attr} fuera de rango [0,1]")

@dataclass
class MetadatosEsteticos:
    inspiracion: str = ""; referentes_artisticos: List[str] = field(default_factory=list)
    generos_musicales: List[str] = field(default_factory=list); colores_asociados: List[str] = field(default_factory=list)
    texturas_visuales: List[str] = field(default_factory=list); elementos_naturales: List[str] = field(default_factory=list)
    arquetipos_psicologicos: List[str] = field(default_factory=list); simbolismo: List[str] = field(default_factory=list)
    momentos_dia: List[str] = field(default_factory=list); estaciones: List[str] = field(default_factory=list)
    ambientes: List[str] = field(default_factory=list); actividades: List[str] = field(default_factory=list)

@dataclass
class PresetEstiloCompleto:
    nombre: str; fade: ParametrosFade; panorama: ConfiguracionPanorama; espectro: EspectroArmonico
    envolvente: TipoEnvolvente; textura: PropiedadesTextura = field(default_factory=PropiedadesTextura)
    metadatos: MetadatosEsteticos = field(default_factory=MetadatosEsteticos)
    neurotransmisores_asociados: List[str] = field(default_factory=list)
    preset_emocional_compatible: List[str] = field(default_factory=list); perfil_estilo_base: str = ""
    adaptabilidad: float = 0.5; complejidad_estetica: float = 0.5; intensidad_emocional: float = 0.5
    version: str = "v7.0"; validado: bool = False; confidence_score: float = 0.8; usage_popularity: float = 0.0
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        for attr, val in [('adaptabilidad', self.adaptabilidad), ('complejidad_estetica', self.complejidad_estetica), ('intensidad_emocional', self.intensidad_emocional)]:
            if not 0 <= val <= 1: warnings.warn(f"{attr} fuera de rango [0,1]")
        if not self.neurotransmisores_asociados: self._inferir_neurotransmisores()
    
    def _inferir_neurotransmisores(self):
        mapeo = {
            (TipoEnvolvente.SUAVE, TipoEnvolvente.CELESTIAL, TipoEnvolvente.LUMINOSA): ["serotonina", "oxitocina"],
            (TipoEnvolvente.RITMICA, TipoEnvolvente.VIBRANTE, TipoEnvolvente.PULIDA): ["dopamina", "norepinefrina"],
            (TipoEnvolvente.ETEREA, TipoEnvolvente.CUANTICA, TipoEnvolvente.HOLOGRAFICA): ["anandamida", "serotonina"],
            (TipoEnvolvente.GRAVE, TipoEnvolvente.CALIDA, TipoEnvolvente.ORGANICA): ["gaba", "melatonina"]
        }
        for envs, nts in mapeo.items():
            if self.envolvente in envs: self.neurotransmisores_asociados.extend(nts); break

class GestorPresetsEstilos:
    def __init__(self):
        self.presets = {}; self.cache_armonicos = {}; self._init_presets()
    
    def _init_presets(self):
        presets_data = {
            "etereo": {
                "fade": (3.0, 3.0, "sigmoid", "exponential", True),
                "panorama": (True, PatronPan.ONDULATORIO, 0.08, 0.7, 0.0, "libre", 0.3, 0.0, True, False),
                "espectro": ([432.0, 528.0], EscalaArmonica.SOLFEGGIO, "432hz", True, True, 0.05),
                "envolvente": TipoEnvolvente.SUAVE,
                "textura": (0.1, 0.2, 1.0, 0.7, 0.4, 0.0, True, 0.0, 1.5, 0.8, 0.2),
                "metadatos": ("Espacios infinitos y consciencia expandida", [], [], ["azul_celeste", "violeta_suave", "blanco_perlado"], [], ["nubes", "aurora", "espacio"], ["el_soñador", "el_visionario"], [], ["amanecer", "atardecer"], [], ["meditación", "espacios_abiertos"], []),
                "neurotransmisores": ["anandamida", "serotonina", "oxitocina"],
                "emocional_compatible": ["conexion_mistica", "expansion_creativa"],
                "complejidad": 0.7, "intensidad": 0.6, "validado": True, "confidence": 0.92
            },
            "sacro": {
                "fade": (2.0, 4.0, "exponential", "linear", False),
                "panorama": (False, PatronPan.ESTATICO, 0.1, 1.0, 0.0, "libre", 0.4, 0.3, False, False),
                "espectro": ([639.0, 963.0], EscalaArmonica.SOLFEGGIO, "440hz", False, False, 0.1),
                "envolvente": TipoEnvolvente.CELESTIAL,
                "textura": (0.0, 0.0, 1.0, 0.8, 0.6, 0.0, False, 0.0, 1.0, 0.9, 0.1),
                "metadatos": ("Espacios sagrados y reverencia espiritual", ["canto_gregoriano", "musica_sacra"], [], ["dorado", "blanco_puro", "azul_profundo"], [], ["luz", "montañas", "templos"], ["el_sabio", "el_sacerdote"], [], ["amanecer", "meditación"], [], ["templos", "espacios_sagrados", "ceremonia"], []),
                "neurotransmisores": ["serotonina", "oxitocina", "anandamida"],
                "emocional_compatible": ["conexion_mistica", "apertura_corazon"],
                "complejidad": 0.5, "intensidad": 0.8, "validado": True, "confidence": 0.88
            },
            "alienigena": {
                "fade": (2.0, 2.0, "custom", "custom", False),
                "panorama": (True, PatronPan.ALEATORIO, 0.3, 0.9, 0.0, "libre", 0.0, 0.0, True, True),
                "espectro": ([888.0, 936.0], EscalaArmonica.CUSTOM, "440hz", False, True, 0.4),
                "envolvente": TipoEnvolvente.INQUIETANTE,
                "textura": (0.6, 0.7, 1.0, 0.8, 0.2, 0.8, False, 0.5, 1.0, 0.8, 0.8),
                "metadatos": ("Inteligencia no-humana y lo desconocido", [], ["ambient_experimental", "drone", "noise"], ["verde_acido", "azul_electrico", "violeta_neon"], [], ["cristales", "fenomenos_extraños", "luces"], ["el_explorador", "el_rebelde"], [], [], [], ["laboratorio", "espacios_experimentales", "ciencia_ficcion"], []),
                "neurotransmisores": ["dopamina", "norepinefrina", "acetilcolina"],
                "emocional_compatible": ["expansion_creativa", "claridad_mental"],
                "complejidad": 0.9, "intensidad": 0.7, "validado": True, "confidence": 0.82
            },
            "sereno": {
                "fade": (4.0, 4.0, "sigmoid", "sigmoid", True),
                "panorama": (True, PatronPan.RESPIRATORIO, 0.05, 0.3, 0.0, "respiracion", 0.0, 0.0, False, False),
                "espectro": ([285.0, 417.0], EscalaArmonica.SOLFEGGIO, "440hz", False, True, 0.02),
                "envolvente": TipoEnvolvente.LUMINOSA,
                "textura": (0.0, 0.1, 1.0, 0.6, 0.8, 0.0, True, 0.0, 1.0, 0.9, 0.2),
                "metadatos": ("Momentos de paz interior y contemplación", [], [], ["azul_suave", "verde_natural", "dorado_cálido"], [], ["agua_quieta", "bosque", "luz_dorada"], ["el_sabio", "el_sanador"], [], ["amanecer", "atardecer", "noche"], [], ["meditación", "naturaleza", "hogar"], []),
                "neurotransmisores": ["gaba", "serotonina", "oxitocina"],
                "emocional_compatible": ["calma_profunda", "introspeccion_suave"],
                "complejidad": 0.4, "intensidad": 0.5, "validado": True, "confidence": 0.94
            },
            "tribal": {
                "fade": (1.0, 2.0, "linear", "exponential", False),
                "panorama": (True, PatronPan.CIRCULAR, 0.15, 0.8, 0.0, "cardiaco", 0.0, 0.0, False, False),
                "espectro": ([140.0, 220.0], EscalaArmonica.PYTHAGOREAN, "440hz", False, False, 0.1),
                "envolvente": TipoEnvolvente.RITMICA,
                "textura": (0.5, 0.3, 1.0, 0.4, 0.9, 0.0, False, 1.2, 1.0, 0.8, 0.2),
                "metadatos": ("Conexión tribal y fuerza ancestral", [], ["percusion_tribal", "world_music", "shamanic"], ["rojo_tierra", "ocre", "naranja_fuego"], [], ["fuego", "tierra", "tambores"], ["el_guerrero", "el_chamán"], [], [], [], ["ritual", "danza", "ceremonia_grupal"], []),
                "neurotransmisores": ["dopamina", "endorfina", "oxitocina"],
                "emocional_compatible": ["fuerza_tribal", "alegria_sostenida"],
                "complejidad": 0.6, "intensidad": 0.8, "validado": True, "confidence": 0.89
            },
            "crystalline": {
                "fade": (2.0, 2.0, "linear", "linear", False),
                "panorama": (True, PatronPan.ORBITAL, 0.12, 0.6, 0.0, "libre", 0.2, 0.0, False, False),
                "espectro": ([512.0, 1024.0], EscalaArmonica.EQUAL_TEMPERAMENT, "440hz", True, False, 0.1),
                "envolvente": TipoEnvolvente.TRANSPARENTE,
                "textura": (0.0, 0.0, 1.0, 0.9, 0.3, 0.0, False, 0.0, 1.0, 1.0, 0.0),
                "metadatos": ("Claridad cristalina y pureza absoluta", [], [], ["blanco_cristal", "azul_hielo", "transparente"], [], ["cristales", "hielo", "agua_pura"], ["el_perfeccionista", "el_visionario"], [], [], [], ["laboratorio", "espacios_minimalistas", "meditación"], []),
                "neurotransmisores": ["acetilcolina", "serotonina"],
                "emocional_compatible": ["claridad_mental", "regulacion_emocional"],
                "complejidad": 0.3, "intensidad": 0.4, "validado": True, "confidence": 0.91
            },
            "cuantico": {
                "fade": (5.0, 5.0, "sigmoid", "sigmoid", True),
                "panorama": (True, PatronPan.CUANTICO, 0.618, 0.8, 0.0, "libre", 0.0, 0.0, True, True),
                "espectro": ([7.83, 14.3, 20.8], EscalaArmonica.SCHUMANN, "440hz", True, True, 0.1),
                "envolvente": TipoEnvolvente.CUANTICA,
                "textura": (0.2, 0.618, 1.0, 0.7, 0.5, 0.3, False, 0.0, 1.618, 0.8, 0.6),
                "metadatos": ("Mecánica cuántica y coherencia del campo", [], [], ["azul_quantum", "violeta_cosmico", "dorado_phi"], [], ["campo_energetico", "ondas", "particulas"], ["el_cientifico_mistico", "el_explorador_dimensional"], [], [], [], ["investigación_avanzada", "meditación_profunda", "sanación"], []),
                "neurotransmisores": ["anandamida", "serotonina", "acetilcolina"],
                "emocional_compatible": ["resonancia_cuantica", "conexion_mistica"],
                "complejidad": 0.9, "intensidad": 0.6, "validado": False, "confidence": 0.75
            }
        }
        
        for nombre, data in presets_data.items():
            self.presets[nombre] = PresetEstiloCompleto(
                nombre=nombre.title(),
                fade=ParametrosFade(*data["fade"][:2], data["fade"][2], data["fade"][3], data["fade"][4] if len(data["fade"]) > 4 else False),
                panorama=ConfiguracionPanorama(*data["panorama"]),
                espectro=EspectroArmonico(data["espectro"][0], data["espectro"][1], data["espectro"][2], razon_aurea=data["espectro"][3], modulacion_armonica=data["espectro"][4], velocidad_modulacion=data["espectro"][5]),
                envolvente=data["envolvente"],
                textura=PropiedadesTextura(*data["textura"]),
                metadatos=MetadatosEsteticos(data["metadatos"][0], data["metadatos"][1], data["metadatos"][2], data["metadatos"][3], data["metadatos"][4], data["metadatos"][5], data["metadatos"][6], data["metadatos"][7], data["metadatos"][8], data["metadatos"][9], data["metadatos"][10], data["metadatos"][11]),
                neurotransmisores_asociados=data["neurotransmisores"],
                preset_emocional_compatible=data["emocional_compatible"],
                complejidad_estetica=data["complejidad"],
                intensidad_emocional=data["intensidad"],
                validado=data["validado"],
                confidence_score=data["confidence"]
            )
    
    @lru_cache(maxsize=64)
    def obtener_preset(self, nombre: str) -> Optional[PresetEstiloCompleto]:
        return self.presets.get(nombre.lower())
    
    def obtener_presets_por_complejidad(self, min_c: float, max_c: float) -> List[PresetEstiloCompleto]:
        return [p for p in self.presets.values() if min_c <= p.complejidad_estetica <= max_c]
    
    def obtener_presets_por_intensidad_emocional(self, min_i: float) -> List[PresetEstiloCompleto]:
        return [p for p in self.presets.values() if p.intensidad_emocional >= min_i]
    
    def obtener_presets_por_neurotransmisor(self, nt: str) -> List[PresetEstiloCompleto]:
        return [p for p in self.presets.values() if any(nt.lower() in n.lower() for n in p.neurotransmisores_asociados)]
    
    def obtener_presets_compatibles(self, pe: str) -> List[PresetEstiloCompleto]:
        return [p for p in self.presets.values() if any(pe.lower() in c.lower() for c in p.preset_emocional_compatible)]
    
    def analizar_compatibilidad_estetica(self, e1: str, e2: str) -> Dict[str, Any]:
        p1, p2 = self.obtener_preset(e1), self.obtener_preset(e2)
        if not p1 or not p2: return {"error": "Uno o ambos estilos no encontrados"}
        
        fo = self._calc_freq_overlap(p1.espectro, p2.espectro)
        ts = self._calc_texture_sim(p1.textura, p2.textura)
        pc = self._calc_pan_compat(p1.panorama, p2.panorama)
        sf = (fo + ts + pc) / 3
        
        return {
            "score_compatibilidad": sf, "solapamiento_frecuencial": fo, "similitud_textural": ts,
            "compatibilidad_panoramica": pc, "recomendacion": self._gen_rec(sf),
            "parametros_transicion": self._sugerir_trans(p1, p2, sf)
        }
    
    def _calc_freq_overlap(self, e1, e2) -> float:
        f1 = set(e1.frecuencias_base + e1.armonicos_naturales)
        f2 = set(e2.frecuencias_base + e2.armonicos_naturales)
        if not f1 or not f2: return 0.0
        overlap = sum(1 for f1_val in f1 if any(abs(f1_val - f2_val) < 10 for f2_val in f2))
        return overlap / max(len(f1), len(f2))
    
    def _calc_texture_sim(self, t1, t2) -> float:
        attrs = ['rugosidad', 'granularidad', 'brillantez', 'calidez', 'fluctuacion']
        return max(0.0, 1.0 - sum(abs(getattr(t1, a) - getattr(t2, a)) for a in attrs) / len(attrs))
    
    def _calc_pan_compat(self, p1, p2) -> float:
        score = 0.5 if p1.patron == p2.patron else (0.3 if p1.dinamico == p2.dinamico else 0.0)
        if p1.dinamico and p2.dinamico: score += max(0, 0.5 - abs(p1.velocidad - p2.velocidad))
        return min(1.0, score)
    
    def _gen_rec(self, s: float) -> str:
        return ("Excelente compatibilidad - Ideal para transiciones fluidas" if s > 0.8 else
                "Buena compatibilidad - Funciona bien en secuencias" if s > 0.6 else
                "Compatibilidad moderada - Requiere transición cuidadosa" if s > 0.4 else
                "Baja compatibilidad - Considerar estilo intermedio")
    
    def _sugerir_trans(self, p1, p2, s: float) -> Dict[str, Any]:
        return ({"tipo": "crossfade_directo", "duracion_s": 3.0, "curva": "sigmoid"} if s > 0.7 else
                {"tipo": "fade_through_neutral", "duracion_s": 5.0, "estilo_intermedio": "sereno"} if s > 0.5 else
                {"tipo": "transicion_compleja", "duracion_s": 8.0, "pasos": ["fade_out", "silencio", "fade_in"], "silencio_s": 2.0})
    
    def generar_preset_personalizado(self, nombre: str, carac: Dict[str, Any]) -> PresetEstiloCompleto:
        return PresetEstiloCompleto(
            nombre=nombre,
            fade=ParametrosFade(carac.get('fade_in', 3.0), carac.get('fade_out', 3.0)),
            panorama=ConfiguracionPanorama(dinamico=carac.get('dinamico', True)),
            espectro=EspectroArmonico(frecuencias_base=carac.get('frecuencias', [440.0, 880.0])),
            envolvente=TipoEnvolvente.SUAVE,
            complejidad_estetica=carac.get('complejidad', 0.5),
            intensidad_emocional=carac.get('intensidad', 0.5),
            validado=False, confidence_score=0.6
        )
    
    def exportar_configuracion_aurora(self) -> Dict[str, Any]:
        return {
            "version": "v7.0", "total_presets": len(self.presets),
            "tipos_envolvente": [e.value for e in TipoEnvolvente],
            "escalas_armonicas": [e.value for e in EscalaArmonica],
            "patrones_panorama": [p.value for p in PatronPan],
            "presets_validados": sum(1 for p in self.presets.values() if p.validado),
            "confidence_promedio": sum(p.confidence_score for p in self.presets.values()) / len(self.presets),
            "complejidad_promedio": sum(p.complejidad_estetica for p in self.presets.values()) / len(self.presets),
            "intensidad_promedio": sum(p.intensidad_emocional for p in self.presets.values()) / len(self.presets)
        }
    
    def limpiar_cache(self): self.obtener_preset.cache_clear(); self.cache_armonicos.clear()

def crear_gestor_estilos_esteticos() -> GestorPresetsEstilos: return GestorPresetsEstilos()
def obtener_preset_estetico_completo(nombre: str) -> Optional[PresetEstiloCompleto]: return crear_gestor_estilos_esteticos().obtener_preset(nombre)
def encontrar_estilo_por_emocion(pe: str) -> List[str]: return [p.nombre.lower() for p in crear_gestor_estilos_esteticos().obtener_presets_compatibles(pe)]
def analizar_transicion_estetica(ea: str, eo: str) -> Dict[str, Any]: return crear_gestor_estilos_esteticos().analizar_compatibilidad_estetica(ea, eo)

def obtener_estilo_detalles(nombre: str) -> Dict[str, Any]:
    preset = obtener_preset_estetico_completo(nombre)
    if not preset:
        legacy = {
            'etereo': {'fade': (3, 3), 'pan_dinamico': True, 'armonia': [432, 528], 'envolvente': 'suave'},
            'sacro': {'fade': (2, 4), 'pan_dinamico': False, 'armonia': [639, 963], 'envolvente': 'celestial'},
            'alienigena': {'fade': (2, 2), 'pan_dinamico': True, 'armonia': [888, 936], 'envolvente': 'inquietante'},
            'oscuro': {'fade': (1, 5), 'pan_dinamico': False, 'armonia': [111, 222], 'envolvente': 'grave'},
            'neutra': {'fade': (3, 3), 'pan_dinamico': False, 'armonia': [261.63, 329.63], 'envolvente': 'plana'},
            'sereno': {'fade': (4, 4), 'pan_dinamico': True, 'armonia': [285, 417], 'envolvente': 'luminosa'},
            'tribal': {'fade': (1, 2), 'pan_dinamico': True, 'armonia': [140, 220], 'envolvente': 'ritmica'},
            'mistico': {'fade': (3, 4), 'pan_dinamico': True, 'armonia': [396, 741], 'envolvente': 'vibrante'},
            'sutil': {'fade': (4, 4), 'pan_dinamico': True, 'armonia': [222, 333], 'envolvente': 'etérea'},
            'futurista': {'fade': (2, 2), 'pan_dinamico': True, 'armonia': [500, 600], 'envolvente': 'pulida'},
            'minimalista': {'fade': (3, 3), 'pan_dinamico': False, 'armonia': [250, 400], 'envolvente': 'limpia'},
            'warm_dust': {'fade': (3, 5), 'pan_dinamico': True, 'armonia': [128, 256], 'envolvente': 'cálida'},
            'crystalline': {'fade': (2, 2), 'pan_dinamico': True, 'armonia': [512, 1024], 'envolvente': 'transparente'},
            'shimmer': {'fade': (2, 4), 'pan_dinamico': True, 'armonia': [720, 840], 'envolvente': 'brillante'},
            'centered': {'fade': (2, 2), 'pan_dinamico': False, 'armonia': [333, 444], 'envolvente': 'focal'}
        }
        return legacy.get(nombre, {})
    return {
        'fade': (preset.fade.fade_in_sec, preset.fade.fade_out_sec),
        'pan_dinamico': preset.panorama.dinamico,
        'armonia': preset.espectro.frecuencias_base,
        'envolvente': preset.envolvente.value
    }

ESTILOS_DETALLES = {n: obtener_estilo_detalles(n) for n in ['etereo', 'sacro', 'alienigena', 'oscuro', 'neutra', 'sereno', 'tribal', 'mistico', 'sutil', 'futurista', 'minimalista', 'warm_dust', 'crystalline', 'shimmer', 'centered']}