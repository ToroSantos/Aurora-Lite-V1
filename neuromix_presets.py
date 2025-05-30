"""Sistema neuroacústico Aurora V7 - Versión optimizada"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from functools import lru_cache
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neurotransmisor(Enum):
    DOPAMINA = "dopamina"
    SEROTONINA = "serotonina"
    GABA = "gaba"
    OXITOCINA = "oxitocina"
    ANANDAMIDA = "anandamida"
    ACETILCOLINA = "acetilcolina"
    ENDORFINA = "endorfina"
    BDNF = "bdnf"
    ADRENALINA = "adrenalina"
    NOREPINEFRINA = "norepinefrina"
    CORTISOL = "cortisol"
    MELATONINA = "melatonina"

class EstadoEmocional(Enum):
    ENFOQUE = "enfoque"
    RELAJACION = "relajacion"
    GRATITUD = "gratitud"
    VISUALIZACION = "visualizacion"
    SOLTAR = "soltar"
    ACCION = "accion"
    CLARIDAD_MENTAL = "claridad_mental"
    SEGURIDAD_INTERIOR = "seguridad_interior"
    APERTURA_CORAZON = "apertura_corazon"
    ALEGRIA_SOSTENIDA = "alegria_sostenida"
    FUERZA_TRIBAL = "fuerza_tribal"
    CONEXION_MISTICA = "conexion_mistica"
    REGULACION_EMOCIONAL = "regulacion_emocional"
    EXPANSION_CREATIVA = "expansion_creativa"
    ESTADO_FLUJO = "estado_flujo"
    INTROSPECCION_SUAVE = "introspeccion_suave"
    SANACION_PROFUNDA = "sanacion_profunda"
    EQUILIBRIO_MENTAL = "equilibrio_mental"

@dataclass
class PerfilNeurotransmisor:
    nombre: str
    frecuencia_primaria: float
    frecuencias_armonicas: List[float] = field(default_factory=list)
    intensidad_base: float = 1.0
    duracion_efectiva_ms: int = 5000
    fase_onset_ms: int = 500
    descripcion: str = ""
    efectos_cognitivos: List[str] = field(default_factory=list)
    efectos_emocionales: List[str] = field(default_factory=list)
    interacciones: Dict[str, float] = field(default_factory=dict)

@dataclass
class PresetEmocional:
    nombre: str
    neurotransmisores: Dict[Neurotransmisor, float]
    frecuencia_base: float
    intensidad_global: float = 1.0
    duracion_ms: int = 6000
    modulacion_temporal: Optional[str] = None
    descripcion: str = ""
    categoria: str = "general"
    nivel_activacion: str = "medio"
    
    def __post_init__(self):
        self._validar_preset()
    
    def _validar_preset(self):
        if not self.neurotransmisores:
            raise ValueError("Preset debe tener al menos un neurotransmisor")
        for neuro, intensidad in self.neurotransmisores.items():
            if not 0 <= intensidad <= 1:
                warnings.warn(f"Intensidad de {neuro.value} fuera de rango [0,1]: {intensidad}")
        if self.frecuencia_base <= 0:
            raise ValueError("Frecuencia base debe ser positiva")

class GestorNeuroacustico:
    
    def __init__(self):
        self._inicializar_perfiles_neurotransmisores()
        self._inicializar_presets_emocionales()
        self._cache_combinaciones = {}
        
    def _inicializar_perfiles_neurotransmisores(self):
        self.perfiles_neuro = {
            Neurotransmisor.DOPAMINA: PerfilNeurotransmisor(
                "Dopamina", 396.0, [792.0, 1188.0], 0.8, 4000, 300,
                "Neurotransmisor de recompensa y motivación",
                ["enfoque", "motivación", "aprendizaje"],
                ["satisfacción", "alegría", "confianza"],
                {Neurotransmisor.SEROTONINA.value: 0.7, Neurotransmisor.GABA.value: -0.3}
            ),
            Neurotransmisor.SEROTONINA: PerfilNeurotransmisor(
                "Serotonina", 417.0, [834.0, 1251.0], 0.75, 6000, 800,
                "Regulador del estado de ánimo y bienestar",
                ["estabilidad", "claridad", "paciencia"],
                ["calma", "bienestar", "contentamiento"],
                {Neurotransmisor.GABA.value: 0.8, Neurotransmisor.CORTISOL.value: -0.6}
            ),
            Neurotransmisor.GABA: PerfilNeurotransmisor(
                "GABA", 72.0, [144.0, 216.0, 288.0], 0.9, 8000, 1200,
                "Principal neurotransmisor inhibitorio",
                ["relajación", "reducción_ansiedad", "quietud"],
                ["paz", "tranquilidad", "seguridad"],
                {Neurotransmisor.ADRENALINA.value: -0.9, Neurotransmisor.CORTISOL.value: -0.7}
            ),
            Neurotransmisor.OXITOCINA: PerfilNeurotransmisor(
                "Oxitocina", 528.0, [1056.0, 1584.0], 0.85, 5000, 600,
                "Hormona de conexión y confianza",
                ["empatía", "conexión", "confianza"],
                ["amor", "seguridad", "pertenencia"],
                {Neurotransmisor.SEROTONINA.value: 0.6, Neurotransmisor.ENDORFINA.value: 0.7}
            ),
            Neurotransmisor.ANANDAMIDA: PerfilNeurotransmisor(
                "Anandamida", 111.0, [222.0, 333.0, 444.0], 0.7, 7000, 1000,
                "Cannabinoide endógeno de la dicha",
                ["creatividad", "perspectiva_amplia", "intuición"],
                ["euforia_suave", "apertura", "liberación"],
                {Neurotransmisor.DOPAMINA.value: 0.5, Neurotransmisor.GABA.value: 0.4}
            ),
            Neurotransmisor.ACETILCOLINA: PerfilNeurotransmisor(
                "Acetilcolina", 320.0, [640.0, 960.0], 0.8, 3500, 200,
                "Neurotransmisor de atención y aprendizaje",
                ["concentración", "memoria", "alerta"],
                ["determinación", "claridad", "presencia"],
                {Neurotransmisor.DOPAMINA.value: 0.6, Neurotransmisor.BDNF.value: 0.8}
            ),
            Neurotransmisor.ENDORFINA: PerfilNeurotransmisor(
                "Endorfina", 528.0, [1056.0, 1584.0, 2112.0], 0.85, 6000, 400,
                "Opiáceo natural del bienestar",
                ["resistencia", "superación", "fluidez"],
                ["euforia", "satisfacción", "fortaleza"],
                {Neurotransmisor.DOPAMINA.value: 0.8, Neurotransmisor.OXITOCINA.value: 0.7}
            ),
            Neurotransmisor.BDNF: PerfilNeurotransmisor(
                "BDNF", 285.0, [570.0, 855.0], 0.75, 4500, 600,
                "Factor neurotrófico de crecimiento",
                ["neuroplasticidad", "aprendizaje", "adaptación"],
                ["crecimiento", "renovación", "vitalidad"],
                {Neurotransmisor.ACETILCOLINA.value: 0.8, Neurotransmisor.DOPAMINA.value: 0.6}
            ),
            Neurotransmisor.ADRENALINA: PerfilNeurotransmisor(
                "Adrenalina", 741.0, [1482.0, 2223.0], 0.9, 2000, 100,
                "Hormona de activación y energía",
                ["alerta_máxima", "reacción_rápida", "enfoque_intenso"],
                ["energía", "determinación", "coraje"],
                {Neurotransmisor.GABA.value: -0.8, Neurotransmisor.CORTISOL.value: 0.6}
            ),
            Neurotransmisor.NOREPINEFRINA: PerfilNeurotransmisor(
                "Norepinefrina", 693.0, [1386.0, 2079.0], 0.8, 3000, 150,
                "Neurotransmisor de alerta y atención",
                ["vigilancia", "concentración", "decisión"],
                ["alerta", "confianza", "determinación"],
                {Neurotransmisor.DOPAMINA.value: 0.7, Neurotransmisor.ADRENALINA.value: 0.8}
            ),
            Neurotransmisor.MELATONINA: PerfilNeurotransmisor(
                "Melatonina", 108.0, [216.0, 324.0], 0.9, 10000, 2000,
                "Hormona reguladora del sueño",
                ["relajación_profunda", "preparación_descanso"],
                ["serenidad", "paz_interior", "soltar"],
                {Neurotransmisor.GABA.value: 0.8, Neurotransmisor.ADRENALINA.value: -0.9}
            )
        }
    
    def _inicializar_presets_emocionales(self):
        self.presets = {
            EstadoEmocional.ENFOQUE: PresetEmocional(
                "Enfoque Profundo", {Neurotransmisor.DOPAMINA: 0.8, Neurotransmisor.ACETILCOLINA: 0.9, Neurotransmisor.NOREPINEFRINA: 0.6},
                14.5, 0.85, 4000, "constante", "Concentración sostenida y claridad mental", "cognitivo", "alto"
            ),
            EstadoEmocional.RELAJACION: PresetEmocional(
                "Relajación Profunda", {Neurotransmisor.GABA: 0.9, Neurotransmisor.SEROTONINA: 0.8, Neurotransmisor.MELATONINA: 0.5},
                8.0, 0.9, 8000, "descendente", "Relajación completa y liberación del estrés", "bienestar", "bajo"
            ),
            EstadoEmocional.ESTADO_FLUJO: PresetEmocional(
                "Estado de Flujo", {Neurotransmisor.DOPAMINA: 0.9, Neurotransmisor.NOREPINEFRINA: 0.7, Neurotransmisor.ENDORFINA: 0.6, Neurotransmisor.ANANDAMIDA: 0.4},
                12.0, 0.8, 6000, "ondulatoria", "Rendimiento óptimo y inmersión total", "performance", "alto"
            ),
            EstadoEmocional.CONEXION_MISTICA: PresetEmocional(
                "Conexión Mística", {Neurotransmisor.ANANDAMIDA: 0.8, Neurotransmisor.SEROTONINA: 0.6, Neurotransmisor.OXITOCINA: 0.7, Neurotransmisor.GABA: 0.5},
                5.0, 0.7, 10000, "expansiva", "Expansión de consciencia y conexión universal", "espiritual", "medio"
            ),
            EstadoEmocional.SANACION_PROFUNDA: PresetEmocional(
                "Sanación Profunda", {Neurotransmisor.OXITOCINA: 0.9, Neurotransmisor.ENDORFINA: 0.8, Neurotransmisor.GABA: 0.7, Neurotransmisor.BDNF: 0.6},
                6.5, 0.85, 12000, "sanadora", "Regeneración y sanación integral", "terapeutico", "medio"
            ),
            EstadoEmocional.EXPANSION_CREATIVA: PresetEmocional(
                "Expansión Creativa", {Neurotransmisor.DOPAMINA: 0.8, Neurotransmisor.ACETILCOLINA: 0.7, Neurotransmisor.ANANDAMIDA: 0.6, Neurotransmisor.BDNF: 0.7},
                11.5, 0.75, 7000, "creativa", "Inspiración y expresión creativa libre", "creativo", "medio-alto"
            ),
            EstadoEmocional.GRATITUD: PresetEmocional(
                "Gratitud Profunda", {Neurotransmisor.OXITOCINA: 0.9, Neurotransmisor.ENDORFINA: 0.8, Neurotransmisor.SEROTONINA: 0.7},
                9.5, 0.8, 6000, categoria="emocional", nivel_activacion="medio"
            ),
            EstadoEmocional.EQUILIBRIO_MENTAL: PresetEmocional(
                "Equilibrio Mental", {Neurotransmisor.SEROTONINA: 0.8, Neurotransmisor.GABA: 0.7, Neurotransmisor.DOPAMINA: 0.5, Neurotransmisor.ACETILCOLINA: 0.6},
                10.0, 0.75, 8000, categoria="bienestar", nivel_activacion="medio"
            )
        }
    
    @lru_cache(maxsize=64)
    def obtener_frecuencias_por_estado(self, estado: EstadoEmocional) -> List[float]:
        try:
            if estado not in self.presets:
                logger.warning(f"Estado {estado.value} no encontrado, usando enfoque")
                estado = EstadoEmocional.ENFOQUE
            preset = self.presets[estado]
            frecuencias = [preset.frecuencia_base]
            for neuro, intensidad in preset.neurotransmisores.items():
                if neuro in self.perfiles_neuro:
                    perfil = self.perfiles_neuro[neuro]
                    frecuencias.append(perfil.frecuencia_primaria * intensidad)
                    if intensidad > 0.6:
                        for armonico in perfil.frecuencias_armonicas[:2]:
                            frecuencias.append(armonico * intensidad * 0.5)
            return sorted(list(set(frecuencias)))
        except Exception as e:
            logger.error(f"Error obteniendo frecuencias para {estado.value}: {e}")
            return [220.0, 440.0]
    
    def calcular_interacciones_neurotransmisores(self, neurotransmisores: Dict[Neurotransmisor, float]) -> Dict[str, float]:
        interacciones_resultantes = {}
        for neuro1, intensidad1 in neurotransmisores.items():
            if neuro1 not in self.perfiles_neuro:
                continue
            perfil1 = self.perfiles_neuro[neuro1]
            modificador_total = 0
            for neuro2, intensidad2 in neurotransmisores.items():
                if neuro1 == neuro2:
                    continue
                if neuro2.value in perfil1.interacciones:
                    factor_interaccion = perfil1.interacciones[neuro2.value]
                    modificador_total += factor_interaccion * intensidad2
            intensidad_ajustada = max(0, min(1, intensidad1 + (modificador_total * 0.1)))
            interacciones_resultantes[neuro1.value] = intensidad_ajustada
        return interacciones_resultantes
    
    def generar_preset_personalizado(self, nombre: str, neurotransmisores: Dict[str, float], frecuencia_base: float, duracion_ms: int = 6000) -> Optional[PresetEmocional]:
        try:
            neuros_enum = {}
            for neuro_str, intensidad in neurotransmisores.items():
                try:
                    neuro_enum = Neurotransmisor(neuro_str.lower())
                    neuros_enum[neuro_enum] = intensidad
                except ValueError:
                    logger.warning(f"Neurotransmisor {neuro_str} no reconocido, ignorando")
            if not neuros_enum:
                raise ValueError("No se pudieron procesar los neurotransmisores")
            return PresetEmocional(nombre, neuros_enum, frecuencia_base, duracion_ms=duracion_ms, categoria="personalizado")
        except Exception as e:
            logger.error(f"Error creando preset personalizado: {e}")
            return None
    
    def obtener_preset_por_categoria(self, categoria: str) -> List[PresetEmocional]:
        return [preset for preset in self.presets.values() if preset.categoria == categoria]
    
    def obtener_presets_por_activacion(self, nivel: str) -> List[PresetEmocional]:
        return [preset for preset in self.presets.values() if preset.nivel_activacion == nivel]
    
    def exportar_configuracion_aurora(self) -> Dict:
        return {
            "version": "v7_neuroacustico",
            "neurotransmisores_disponibles": [n.value for n in Neurotransmisor],
            "estados_disponibles": [e.value for e in EstadoEmocional],
            "categorias": list(set(p.categoria for p in self.presets.values())),
            "niveles_activacion": ["bajo", "medio", "medio-alto", "alto"],
            "rango_frecuencias": {"min": 5.0, "max": 2223.0},
            "duraciones_recomendadas": {"corta": 3000, "media": 6000, "larga": 10000, "muy_larga": 15000}
        }
    
    def limpiar_cache(self):
        self.obtener_frecuencias_por_estado.cache_clear()
        self._cache_combinaciones.clear()
        logger.info("Cache neuroacústico limpiado")

def crear_gestor_neuroacustico() -> GestorNeuroacustico:
    return GestorNeuroacustico()

def obtener_frecuencias_estado(estado: str) -> List[float]:
    gestor = crear_gestor_neuroacustico()
    try:
        estado_enum = EstadoEmocional(estado.lower())
        return gestor.obtener_frecuencias_por_estado(estado_enum)
    except ValueError:
        logger.error(f"Estado {estado} no válido")
        return [220.0, 440.0]

def generar_pack_neuroacustico_completo() -> Dict[str, List[float]]:
    gestor = crear_gestor_neuroacustico()
    return {estado.value: gestor.obtener_frecuencias_por_estado(estado) for estado in EstadoEmocional}

def get_frecuencias_por_estado(preset: str) -> List[float]:
    warnings.warn("get_frecuencias_por_estado está deprecated, usa obtener_frecuencias_estado", DeprecationWarning)
    return obtener_frecuencias_estado(preset)

NEURO_FREQUENCIES = {n.value: crear_gestor_neuroacustico().perfiles_neuro[n].frecuencia_primaria for n in Neurotransmisor}
EMOCIONAL_PRESETS = {"enfoque": ["dopamina", "acetilcolina"], "relajacion": ["gaba", "serotonina"], "gratitud": ["oxitocina", "endorfina"], "visualizacion": ["acetilcolina", "bdnf"], "soltar": ["gaba", "anandamida"], "accion": ["adrenalina", "dopamina"]}