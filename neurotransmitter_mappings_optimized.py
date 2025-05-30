"""Sistema optimizado de mapeo neuroacústico para Aurora V7"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TipoOnda(Enum):
    SINE = "sine"
    SAW = "saw" 
    SQUARE = "square"
    TRIANGLE = "triangle"
    PULSE = "pulse"
    NOISE_WHITE = "white_noise"
    NOISE_PINK = "pink_noise"
    PAD_BLEND = "pad_blend"
    BINAURAL = "binaural"
    NEUROMORPHIC = "neuromorphic"

class TipoModulacion(Enum):
    AM = "am"
    FM = "fm"
    PM = "pm"
    HYBRID = "hybrid"
    NEUROMORPHIC = "neuromorphic"
    BINAURAL_BEAT = "binaural_beat"

@dataclass
class ParametrosEspaciales:
    pan: float = 0.0
    width: float = 1.0
    distance: float = 1.0
    elevation: float = 0.0
    movement_pattern: str = "static"
    movement_speed: float = 0.1

@dataclass
class ParametrosTempo:
    onset_ms: int = 100
    sustain_ms: int = 2000
    decay_ms: int = 500
    release_ms: int = 1000
    rhythm_pattern: str = "steady"

@dataclass
class ParametrosModulacion:
    tipo: TipoModulacion
    profundidad: float
    velocidad_hz: float
    fase_inicial: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.profundidad <= 1:
            warnings.warn(f"Profundidad {self.profundidad} fuera de rango [0,1]")
        if self.velocidad_hz < 0:
            raise ValueError("Velocidad de modulación debe ser positiva")

@dataclass
class ParametrosNeuroacusticos:
    nombre: str
    frecuencia_base: float
    tipo_onda: TipoOnda
    modulacion: ParametrosModulacion
    armonicos: List[float] = field(default_factory=list)
    nivel_db: float = -12.0
    panorama: ParametrosEspaciales = field(default_factory=ParametrosEspaciales)
    tempo: ParametrosTempo = field(default_factory=ParametrosTempo)
    descripcion_acustica: str = ""
    efectos_percibidos: List[str] = field(default_factory=list)
    compatibilidad: List[str] = field(default_factory=list)
    incompatibilidad: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.frecuencia_base <= 0:
            raise ValueError("Frecuencia base debe ser positiva")
        if not -60 <= self.nivel_db <= 0:
            warnings.warn(f"Nivel {self.nivel_db} dB fuera del rango recomendado [-60, 0]")

class GeneradorMapeoNeuroacustico:
    def __init__(self):
        self._mapeos = {}
        self._combinaciones_cache = {}
        self._init_mapeos_optimizados()
    
    def _init_mapeos_optimizados(self):
        """Inicialización optimizada de mapeos neuroacústicos"""
        
        # Dopamina - Motivación y Recompensa
        self._mapeos["dopamina"] = ParametrosNeuroacusticos(
            nombre="Dopamina - Motivación",
            frecuencia_base=12.0,
            tipo_onda=TipoOnda.SINE,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.FM,
                profundidad=0.6,
                velocidad_hz=0.3
            ),
            armonicos=[24.0, 36.0, 48.0],
            nivel_db=-9.0,
            panorama=ParametrosEspaciales(
                pan=0.2, width=1.2, movement_pattern="circular", movement_speed=0.05
            ),
            tempo=ParametrosTempo(
                onset_ms=200, sustain_ms=3000, rhythm_pattern="pulsed"
            ),
            descripcion_acustica="Onda senoidal con modulación FM suave, sensación ascendente",
            efectos_percibidos=["motivación", "enfoque", "satisfacción", "energía positiva"],
            compatibilidad=["acetilcolina", "norepinefrina", "endorfina"],
            incompatibilidad=["gaba_alta_dosis"]
        )
        
        # Serotonina - Bienestar y Estabilidad
        self._mapeos["serotonina"] = ParametrosNeuroacusticos(
            nombre="Serotonina - Bienestar",
            frecuencia_base=7.5,
            tipo_onda=TipoOnda.SAW,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.AM,
                profundidad=0.7,
                velocidad_hz=0.15
            ),
            armonicos=[15.0, 22.5, 30.0],
            nivel_db=-10.0,
            panorama=ParametrosEspaciales(
                width=1.5, movement_pattern="pendulum", movement_speed=0.03
            ),
            tempo=ParametrosTempo(
                onset_ms=800, sustain_ms=5000, rhythm_pattern="breathing"
            ),
            descripcion_acustica="Onda sierra suavizada con modulación AM respiratoria",
            efectos_percibidos=["calma", "bienestar", "estabilidad emocional", "claridad"],
            compatibilidad=["gaba", "oxitocina", "melatonina"],
            incompatibilidad=["adrenalina", "norepinefrina_alta"]
        )
        
        # GABA - Relajación Profunda
        self._mapeos["gaba"] = ParametrosNeuroacusticos(
            nombre="GABA - Relajación Profunda",
            frecuencia_base=6.0,
            tipo_onda=TipoOnda.TRIANGLE,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.AM,
                profundidad=0.8,
                velocidad_hz=0.1
            ),
            armonicos=[12.0, 18.0],
            nivel_db=-8.0,
            panorama=ParametrosEspaciales(width=2.0),
            tempo=ParametrosTempo(
                onset_ms=1500, sustain_ms=8000, decay_ms=2000
            ),
            descripcion_acustica="Onda triangular suave con filtrado profundo, efecto envolvente",
            efectos_percibidos=["relajación profunda", "reducción ansiedad", "paz interior"],
            compatibilidad=["serotonina", "melatonina", "anandamida"],
            incompatibilidad=["adrenalina", "norepinefrina", "dopamina_alta"]
        )
        
        # Oxitocina - Conexión y Confianza
        self._mapeos["oxitocina"] = ParametrosNeuroacusticos(
            nombre="Oxitocina - Conexión",
            frecuencia_base=8.0,
            tipo_onda=TipoOnda.PAD_BLEND,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.HYBRID,
                profundidad=0.5,
                velocidad_hz=0.2
            ),
            armonicos=[16.0, 24.0, 32.0, 40.0],
            nivel_db=-11.0,
            panorama=ParametrosEspaciales(
                width=1.8, movement_pattern="circular", movement_speed=0.02
            ),
            tempo=ParametrosTempo(
                onset_ms=600, sustain_ms=6000, rhythm_pattern="breathing"
            ),
            descripcion_acustica="Pad cálido con modulación híbrida, sensación envolvente",
            efectos_percibidos=["conexión", "confianza", "calidez emocional", "seguridad"],
            compatibilidad=["serotonina", "endorfina", "dopamina"],
            incompatibilidad=["cortisol"]
        )
        
        # Acetilcolina - Atención y Claridad
        self._mapeos["acetilcolina"] = ParametrosNeuroacusticos(
            nombre="Acetilcolina - Atención",
            frecuencia_base=14.0,
            tipo_onda=TipoOnda.SINE,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.FM,
                profundidad=0.6,
                velocidad_hz=0.8
            ),
            armonicos=[28.0, 42.0, 56.0, 70.0],
            nivel_db=-8.5,
            panorama=ParametrosEspaciales(
                pan=-0.3, movement_pattern="random", movement_speed=0.1
            ),
            tempo=ParametrosTempo(
                onset_ms=50, sustain_ms=2500, rhythm_pattern="pulsed"
            ),
            descripcion_acustica="Onda senoidal aguda con FM rápida, sensación de alerta",
            efectos_percibidos=["concentración", "claridad mental", "velocidad procesamiento"],
            compatibilidad=["dopamina", "norepinefrina"],
            incompatibilidad=["gaba_alta", "melatonina"]
        )
        
        # Norepinefrina - Alerta y Preparación
        self._mapeos["norepinefrina"] = ParametrosNeuroacusticos(
            nombre="Norepinefrina - Alerta",
            frecuencia_base=13.5,
            tipo_onda=TipoOnda.PULSE,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.FM,
                profundidad=0.5,
                velocidad_hz=1.2
            ),
            armonicos=[27.0, 40.5, 54.0],
            nivel_db=-9.5,
            panorama=ParametrosEspaciales(
                pan=0.4, movement_pattern="pendulum", movement_speed=0.15
            ),
            tempo=ParametrosTempo(
                onset_ms=30, sustain_ms=2000, rhythm_pattern="irregular"
            ),
            descripcion_acustica="Pulsos modulados con FM irregular, sensación de urgencia controlada",
            efectos_percibidos=["alerta máxima", "preparación", "energía dirigida"],
            compatibilidad=["acetilcolina", "dopamina", "adrenalina"],
            incompatibilidad=["gaba", "serotonina_alta"]
        )
        
        # Endorfina - Euforia Natural
        self._mapeos["endorfina"] = ParametrosNeuroacusticos(
            nombre="Endorfina - Euforia Natural",
            frecuencia_base=10.5,
            tipo_onda=TipoOnda.PAD_BLEND,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.AM,
                profundidad=0.6,
                velocidad_hz=0.25
            ),
            armonicos=[21.0, 31.5, 42.0, 52.5],
            nivel_db=-10.5,
            panorama=ParametrosEspaciales(
                width=1.6, movement_pattern="circular", movement_speed=0.04
            ),
            tempo=ParametrosTempo(
                onset_ms=400, sustain_ms=4500, rhythm_pattern="breathing"
            ),
            descripcion_acustica="Pad cálido con modulación suave, sensación de elevación",
            efectos_percibidos=["bienestar", "euforia controlada", "resistencia al dolor"],
            compatibilidad=["dopamina", "oxitocina", "serotonina"]
        )
        
        # Anandamida - Molécula de la Dicha
        self._mapeos["anandamida"] = ParametrosNeuroacusticos(
            nombre="Anandamida - Molécula de la Dicha",
            frecuencia_base=5.5,
            tipo_onda=TipoOnda.SINE,
            modulacion=ParametrosModulacion(
                tipo=TipoModulacion.NEUROMORPHIC,
                profundidad=0.4,
                velocidad_hz=0.08
            ),
            armonicos=[11.0, 16.5, 22.0, 27.5, 33.0],
            nivel_db=-13.0,
            panorama=ParametrosEspaciales(
                width=2.0, movement_pattern="organic", movement_speed=0.01
            ),
            tempo=ParametrosTempo(
                onset_ms=2000, sustain_ms=10000, rhythm_pattern="organic"
            ),
            descripcion_acustica="Textura orgánica evolutiva, sensación de expansión consciencia",
            efectos_percibidos=["dicha", "expansión", "liberación", "creatividad"],
            compatibilidad=["gaba", "serotonina", "oxitocina"],
            incompatibilidad=["adrenalina", "norepinefrina_alta"]
        )
    
    @lru_cache(maxsize=128)
    def obtener_mapeo(self, nt: str) -> Optional[ParametrosNeuroacusticos]:
        """Obtener mapeo para un neurotransmisor específico"""
        return self._mapeos.get(nt.lower())
    
    def obtener_mapeos_compatibles(self, nt: str) -> List[ParametrosNeuroacusticos]:
        """Obtener mapeos compatibles con un neurotransmisor"""
        mapeo_base = self.obtener_mapeo(nt)
        if not mapeo_base:
            return []
        
        compatibles = []
        for nt_compatible in mapeo_base.compatibilidad:
            mapeo_comp = self.obtener_mapeo(nt_compatible)
            if mapeo_comp:
                compatibles.append(mapeo_comp)
        
        return compatibles
    
    def generar_combinacion_inteligente(self, 
                                      neurotransmisores: List[str], 
                                      pesos: Optional[List[float]] = None) -> Dict[str, Any]:
        """Generar combinación inteligente de neurotransmisores"""
        if not neurotransmisores:
            return {}
        
        # Obtener mapeos válidos
        mapeos = []
        incompatibilidades = []
        
        for nt in neurotransmisores:
            mapeo = self.obtener_mapeo(nt)
            if mapeo:
                mapeos.append(mapeo)
                # Verificar incompatibilidades
                for otro_nt in neurotransmisores:
                    if otro_nt != nt and otro_nt in mapeo.incompatibilidad:
                        incompatibilidades.append((nt, otro_nt))
        
        if incompatibilidades:
            logger.warning(f"Incompatibilidades detectadas: {incompatibilidades}")
        
        # Pesos automáticos si no se proporcionan
        if not pesos:
            pesos = [1.0 / len(mapeos)] * len(mapeos)
        
        # Combinar parámetros
        parametros_combinados = self._combinar_parametros(mapeos, pesos)
        
        return {
            "neurotransmisores": neurotransmisores,
            "pesos": pesos,
            "mapeos": mapeos,
            "incompatibilidades": incompatibilidades,
            "parametros_combinados": parametros_combinados,
            "recomendaciones": self._generar_recomendaciones(mapeos, incompatibilidades)
        }
    
    def _combinar_parametros(self, 
                           mapeos: List[ParametrosNeuroacusticos], 
                           pesos: List[float]) -> Dict[str, Any]:
        """Combinar parámetros de múltiples mapeos"""
        if not mapeos:
            return {}
        
        # Frecuencia combinada (promedio ponderado)
        freq_combinada = sum(m.frecuencia_base * p for m, p in zip(mapeos, pesos))
        
        # Tipo de onda dominante (el de mayor peso)
        idx_dominante = pesos.index(max(pesos))
        tipo_dominante = mapeos[idx_dominante].tipo_onda
        
        # Nivel dB combinado
        nivel_combinado = sum(m.nivel_db * p for m, p in zip(mapeos, pesos))
        
        # Efectos esperados (unión de todos)
        efectos_combinados = list(set([
            efecto for mapeo in mapeos for efecto in mapeo.efectos_percibidos
        ]))
        
        return {
            "frecuencia_base_combinada": freq_combinada,
            "tipo_onda_dominante": tipo_dominante.value,
            "nivel_db_combinado": nivel_combinado,
            "efectos_esperados": efectos_combinados,
            "complejidad": len(mapeos),
            "balance_activacion": self._calcular_balance_activacion(mapeos, pesos)
        }
    
    def _calcular_balance_activacion(self, 
                                   mapeos: List[ParametrosNeuroacusticos], 
                                   pesos: List[float]) -> Dict[str, Any]:
        """Calcular balance de activación del sistema nervioso"""
        
        # Niveles de activación por neurotransmisor
        niveles_activacion = {
            "dopamina": 0.7,
            "acetilcolina": 0.8,
            "norepinefrina": 0.9,
            "adrenalina": 1.0,
            "serotonina": 0.3,
            "gaba": 0.1,
            "melatonina": 0.0,
            "anandamida": 0.2,
            "oxitocina": 0.4,
            "endorfina": 0.5
        }
        
        # Calcular activación total ponderada
        activacion_total = 0
        for mapeo, peso in zip(mapeos, pesos):
            nt_nombre = mapeo.nombre.split(" - ")[0].lower()
            nivel = niveles_activacion.get(nt_nombre, 0.5)
            activacion_total += nivel * peso
        
        # Categorizar nivel de activación
        if activacion_total < 0.2:
            categoria = "muy_relajante"
            recomendacion = "ideal_para_meditacion_profunda_o_descanso"
        elif activacion_total < 0.4:
            categoria = "relajante"
            recomendacion = "perfecto_para_relajacion_y_recuperacion"
        elif activacion_total < 0.6:
            categoria = "equilibrado"
            recomendacion = "excelente_para_trabajo_creativo_relajado"
        elif activacion_total < 0.8:
            categoria = "activante"
            recomendacion = "optimo_para_concentracion_y_productividad"
        else:
            categoria = "muy_activante"
            recomendacion = "recomendado_para_actividades_intensas_corta_duracion"
        
        return {
            "nivel_activacion": activacion_total,
            "categoria": categoria,
            "recomendacion_uso": recomendacion
        }
    
    def _generar_recomendaciones(self, 
                               mapeos: List[ParametrosNeuroacusticos], 
                               incompatibilidades: List[Tuple[str, str]]) -> List[str]:
        """Generar recomendaciones para la combinación"""
        recomendaciones = []
        
        if incompatibilidades:
            recomendaciones.extend([
                "⚠️ Combinación con incompatibilidades detectadas",
                "💡 Considere reducir intensidad o duración de uso"
            ])
        
        if len(mapeos) > 4:
            recomendaciones.append("🔄 Combinación muy compleja, considere simplificar")
        
        # Verificar diferencias temporales
        onsets = [m.tempo.onset_ms for m in mapeos]
        if max(onsets) - min(onsets) > 1000:
            recomendaciones.append("⏱️ Diferencias temporales grandes, ajustar sincronización")
        
        return recomendaciones
    
    def exportar_mapeo_aurora(self, nt: str) -> Optional[Dict[str, Any]]:
        """Exportar mapeo en formato Aurora V7"""
        mapeo = self.obtener_mapeo(nt)
        if not mapeo:
            return None
        
        return {
            "neurotransmisor": nt,
            "parametros_basicos": {
                "base_freq": mapeo.frecuencia_base,
                "wave_type": mapeo.tipo_onda.value,
                "mod_type": mapeo.modulacion.tipo.value,
                "depth": mapeo.modulacion.profundidad
            },
            "parametros_avanzados": {
                "harmonics": mapeo.armonicos,
                "level_db": mapeo.nivel_db,
                "spatial": {
                    "pan": mapeo.panorama.pan,
                    "width": mapeo.panorama.width,
                    "movement": mapeo.panorama.movement_pattern
                },
                "temporal": {
                    "onset_ms": mapeo.tempo.onset_ms,
                    "sustain_ms": mapeo.tempo.sustain_ms,
                    "rhythm": mapeo.tempo.rhythm_pattern
                }
            },
            "metadata": {
                "description": mapeo.descripcion_acustica,
                "effects": mapeo.efectos_percibidos,
                "compatible_with": mapeo.compatibilidad,
                "avoid_with": mapeo.incompatibilidad
            }
        }
    
    def obtener_todos_los_mapeos_aurora(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todos los mapeos en formato Aurora"""
        return {
            nombre: self.exportar_mapeo_aurora(nombre) 
            for nombre in self._mapeos.keys()
        }
    
    def limpiar_cache(self):
        """Limpiar cache del sistema"""
        self.obtener_mapeo.cache_clear()
        self._combinaciones_cache.clear()
        logger.info("Cache de mapeos neuroacústicos limpiado")

# Funciones de conveniencia
def crear_gestor_mapeos() -> GeneradorMapeoNeuroacustico:
    """Crear gestor de mapeos neuroacústicos"""
    return GeneradorMapeoNeuroacustico()

def obtener_parametros_neuroacusticos(nt: str) -> Optional[Dict[str, Any]]:
    """Obtener parámetros neuroacústicos para un neurotransmisor"""
    return crear_gestor_mapeos().exportar_mapeo_aurora(nt)

def generar_combinacion_personalizada(neurotransmisores: List[str], 
                                    pesos: Optional[List[float]] = None) -> Dict[str, Any]:
    """Generar combinación personalizada de neurotransmisores"""
    return crear_gestor_mapeos().generar_combinacion_inteligente(neurotransmisores, pesos)

# Compatibilidad con versiones anteriores
NEUROTRANSMITTER_MAP = {}

def _init_compatibilidad():
    """Inicializar mapeo de compatibilidad"""
    global NEUROTRANSMITTER_MAP
    gestor = crear_gestor_mapeos()
    
    for nombre in gestor._mapeos.keys():
        mapeo_aurora = gestor.exportar_mapeo_aurora(nombre)
        if mapeo_aurora:
            NEUROTRANSMITTER_MAP[nombre] = mapeo_aurora["parametros_basicos"]

# Inicializar compatibilidad
_init_compatibilidad()

def obtener_mapeo_simple(nt: str) -> Dict[str, Any]:
    """Función de compatibilidad con versiones anteriores"""
    warnings.warn(
        "obtener_mapeo_simple está deprecated, usa obtener_parametros_neuroacusticos", 
        DeprecationWarning
    )
    resultado = obtener_parametros_neuroacusticos(nt)
    return resultado["parametros_basicos"] if resultado else {}

if __name__ == "__main__":
    print("🧠 Sistema de Mapeo Neuroacústico Aurora V7")
    print("=" * 50)
    
    # Crear gestor
    gestor = crear_gestor_mapeos()
    
    # Información del sistema
    print(f"📊 Neurotransmisores disponibles: {len(gestor._mapeos)}")
    for nt in gestor._mapeos.keys():
        print(f"  • {nt}")
    
    # Ejemplo de uso
    print(f"\n🔬 Ejemplo: Dopamina")
    dopamina = gestor.obtener_mapeo("dopamina")
    if dopamina:
        print(f"  Frecuencia base: {dopamina.frecuencia_base} Hz")
        print(f"  Efectos: {', '.join(dopamina.efectos_percibidos[:3])}")
    
    # Ejemplo de combinación
    print(f"\n🧪 Ejemplo: Combinación dopamina + serotonina")
    combinacion = gestor.generar_combinacion_inteligente(["dopamina", "serotonina"])
    if combinacion:
        params = combinacion["parametros_combinados"]
        print(f"  Frecuencia combinada: {params['frecuencia_base_combinada']:.1f} Hz")
        print(f"  Balance: {params['balance_activacion']['categoria']}")
    
    print(f"\n✅ Sistema de mapeo neuroacústico listo!")
