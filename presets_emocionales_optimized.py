"""Sistema optimizado de presets emocionales para Aurora V7"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoriaEmocional(Enum):
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    SOCIAL = "social"
    CREATIVO = "creativo"
    ESPIRITUAL = "espiritual"
    TERAPEUTICO = "terapeutico"
    PERFORMANCE = "performance"

class NivelIntensidad(Enum):
    SUAVE = "suave"
    MODERADO = "moderado"
    INTENSO = "intenso"

@dataclass
class EfectosPsicofisiologicos:
    """Efectos psicofisiológicos medibles del preset"""
    atencion: float = 0.0
    memoria: float = 0.0
    concentracion: float = 0.0
    creatividad: float = 0.0
    calma: float = 0.0
    alegria: float = 0.0
    confianza: float = 0.0
    apertura: float = 0.0
    energia: float = 0.0
    empatia: float = 0.0
    conexion: float = 0.0
    
    def __post_init__(self):
        # Validar rango de efectos
        for campo, valor in self.__dict__.items():
            if not -1.0 <= valor <= 1.0:
                logger.warning(f"Efecto {campo} fuera de rango [-1,1]: {valor}")

@dataclass
class PresetEmocionalCompleto:
    """Preset emocional completo optimizado"""
    nombre: str
    descripcion: str
    categoria: CategoriaEmocional
    intensidad: NivelIntensidad = NivelIntensidad.MODERADO
    neurotransmisores: Dict[str, float] = field(default_factory=dict)
    frecuencia_base: float = 10.0
    frecuencias_armonicas: List[float] = field(default_factory=list)
    estilo_asociado: str = "sereno"
    efectos: EfectosPsicofisiologicos = field(default_factory=EfectosPsicofisiologicos)
    mejor_momento_uso: List[str] = field(default_factory=list)
    contextos_recomendados: List[str] = field(default_factory=list)
    contraindicaciones: List[str] = field(default_factory=list)
    presets_compatibles: List[str] = field(default_factory=list)
    nivel_evidencia: str = "experimental"
    confidence_score: float = 0.8
    version: str = "v7.0"
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        self._validar_preset()
        if not self.frecuencias_armonicas:
            self._calcular_frecuencias_armonicas()
        self._inferir_efectos_desde_neurotransmisores()
    
    def _validar_preset(self):
        """Validación básica del preset"""
        if self.frecuencia_base <= 0:
            raise ValueError("Frecuencia base debe ser positiva")
        
        for nt, intensidad in self.neurotransmisores.items():
            if not 0 <= intensidad <= 1:
                logger.warning(f"Intensidad de {nt} fuera de rango [0,1]: {intensidad}")
    
    def _calcular_frecuencias_armonicas(self):
        """Calcular frecuencias armónicas automáticamente"""
        self.frecuencias_armonicas = [
            self.frecuencia_base * i for i in [2, 3, 4, 5]
        ]
    
    def _inferir_efectos_desde_neurotransmisores(self):
        """Inferir efectos desde neurotransmisores configurados"""
        # Mapeo optimizado de neurotransmisores a efectos
        mapeo_efectos = {
            "dopamina": {"atencion": 0.7, "energia": 0.6, "confianza": 0.5},
            "serotonina": {"calma": 0.8, "alegria": 0.6, "apertura": 0.4},
            "gaba": {"calma": 0.9, "atencion": -0.3},
            "oxitocina": {"empatia": 0.9, "conexion": 0.8, "confianza": 0.7},
            "acetilcolina": {"atencion": 0.9, "memoria": 0.8, "concentracion": 0.8},
            "norepinefrina": {"atencion": 0.8, "energia": 0.7},
            "endorfina": {"alegria": 0.8, "energia": 0.7},
            "anandamida": {"creatividad": 0.8, "apertura": 0.9, "alegria": 0.6},
            "melatonina": {"calma": 0.9, "energia": -0.6}
        }
        
        # Calcular efectos combinados
        efectos_calculados = {}
        for campo in self.efectos.__dict__.keys():
            efecto_total = peso_total = 0
            
            for nt, intensidad in self.neurotransmisores.items():
                if nt in mapeo_efectos and campo in mapeo_efectos[nt]:
                    efecto_total += mapeo_efectos[nt][campo] * intensidad
                    peso_total += intensidad
            
            if peso_total > 0:
                efectos_calculados[campo] = np.tanh(efecto_total / peso_total)
        
        # Aplicar efectos calculados si no están ya definidos
        for campo, valor in efectos_calculados.items():
            if getattr(self.efectos, campo) == 0.0:
                setattr(self.efectos, campo, valor)

class GestorPresetsEmocionales:
    """Gestor optimizado de presets emocionales"""
    
    def __init__(self):
        self.presets = {}
        self._inicializar_presets_core()
    
    def _inicializar_presets_core(self):
        """Inicializar presets core optimizados"""
        
        # Presets fundamentales
        presets_data = {
            "claridad_mental": {
                "descripcion": "Lucidez cognitiva y enfoque sereno para trabajo mental intenso",
                "categoria": CategoriaEmocional.COGNITIVO,
                "intensidad": NivelIntensidad.MODERADO,
                "neurotransmisores": {"acetilcolina": 0.9, "dopamina": 0.6, "norepinefrina": 0.4},
                "frecuencia_base": 14.5,
                "estilo_asociado": "crystalline",
                "mejor_momento_uso": ["mañana", "tarde"],
                "contextos_recomendados": ["trabajo", "estudio", "resolución_problemas"],
                "presets_compatibles": ["expansion_creativa", "estado_flujo"],
                "nivel_evidencia": "validado",
                "confidence_score": 0.92
            },
            
            "seguridad_interior": {
                "descripcion": "Confianza centrada y estabilidad emocional profunda",
                "categoria": CategoriaEmocional.EMOCIONAL,
                "intensidad": NivelIntensidad.SUAVE,
                "neurotransmisores": {"gaba": 0.8, "serotonina": 0.7, "oxitocina": 0.5},
                "frecuencia_base": 8.0,
                "estilo_asociado": "sereno",
                "mejor_momento_uso": ["mañana", "noche"],
                "contextos_recomendados": ["meditación", "terapia", "desarrollo_personal"],
                "nivel_evidencia": "validado",
                "confidence_score": 0.89
            },
            
            "apertura_corazon": {
                "descripcion": "Aceptación amorosa y conexión emocional profunda",
                "categoria": CategoriaEmocional.SOCIAL,
                "intensidad": NivelIntensidad.MODERADO,
                "neurotransmisores": {"oxitocina": 0.9, "serotonina": 0.5, "endorfina": 0.6},
                "frecuencia_base": 7.2,
                "estilo_asociado": "organico",
                "mejor_momento_uso": ["tarde", "noche"],
                "contextos_recomendados": ["relaciones", "terapia_pareja", "trabajo_emocional"],
                "presets_compatibles": ["conexion_mistica", "introspeccion_suave"],
                "nivel_evidencia": "experimental",
                "confidence_score": 0.86
            },
            
            "estado_flujo": {
                "descripcion": "Rendimiento óptimo y disfrute del presente",
                "categoria": CategoriaEmocional.PERFORMANCE,
                "intensidad": NivelIntensidad.INTENSO,
                "neurotransmisores": {"dopamina": 0.9, "norepinefrina": 0.7, "endorfina": 0.5},
                "frecuencia_base": 12.0,
                "estilo_asociado": "futurista",
                "mejor_momento_uso": ["mañana", "tarde"],
                "contextos_recomendados": ["deporte", "arte", "programación", "música"],
                "presets_compatibles": ["expansion_creativa", "claridad_mental"],
                "nivel_evidencia": "validado",
                "confidence_score": 0.94
            },
            
            "conexion_mistica": {
                "descripcion": "Unidad espiritual y percepción expandida",
                "categoria": CategoriaEmocional.ESPIRITUAL,
                "intensidad": NivelIntensidad.INTENSO,
                "neurotransmisores": {"anandamida": 0.8, "serotonina": 0.6, "oxitocina": 0.7},
                "frecuencia_base": 5.0,
                "estilo_asociado": "mistico",
                "mejor_momento_uso": ["noche"],
                "contextos_recomendados": ["meditación_profunda", "ceremonia", "retiro"],
                "contraindicaciones": ["ansiedad_severa", "primera_experiencia"],
                "nivel_evidencia": "experimental",
                "confidence_score": 0.82
            },
            
            "regulacion_emocional": {
                "descripcion": "Balance emocional y estabilidad sostenida",
                "categoria": CategoriaEmocional.TERAPEUTICO,
                "neurotransmisores": {"gaba": 0.9, "serotonina": 0.6, "norepinefrina": 0.3},
                "frecuencia_base": 7.8,
                "estilo_asociado": "sereno",
                "contextos_recomendados": ["terapia", "manejo_estres", "recuperacion"],
                "nivel_evidencia": "clinico",
                "confidence_score": 0.91
            },
            
            "expansion_creativa": {
                "descripcion": "Inspiración y creatividad fluida",
                "categoria": CategoriaEmocional.CREATIVO,
                "neurotransmisores": {"dopamina": 0.8, "acetilcolina": 0.7, "anandamida": 0.6},
                "frecuencia_base": 11.5,
                "estilo_asociado": "etereo",
                "contextos_recomendados": ["arte", "escritura", "diseño", "música"],
                "confidence_score": 0.87
            },
            
            "calma_profunda": {
                "descripcion": "Relajación total, serenidad corporal y mental",
                "categoria": CategoriaEmocional.TERAPEUTICO,
                "intensidad": NivelIntensidad.SUAVE,
                "neurotransmisores": {"gaba": 0.9, "serotonina": 0.8, "melatonina": 0.7},
                "frecuencia_base": 6.5,
                "estilo_asociado": "sereno",
                "mejor_momento_uso": ["noche"],
                "contextos_recomendados": ["sueño", "recuperación", "trauma"],
                "presets_compatibles": ["regulacion_emocional"],
                "nivel_evidencia": "clinico",
                "confidence_score": 0.93
            }
        }
        
        # Crear presets
        for nombre, data in presets_data.items():
            self.presets[nombre] = PresetEmocionalCompleto(nombre=nombre.replace("_", " ").title(), **data)
    
    @lru_cache(maxsize=128)
    def obtener_preset(self, nombre: str) -> Optional[PresetEmocionalCompleto]:
        """Obtener preset por nombre (con cache)"""
        return self.presets.get(nombre.lower().replace(" ", "_"))
    
    def buscar_presets_por_efecto(self, efecto: str, umbral: float = 0.5) -> List[PresetEmocionalCompleto]:
        """Buscar presets por efecto específico"""
        resultados = []
        for preset in self.presets.values():
            if hasattr(preset.efectos, efecto):
                valor_efecto = getattr(preset.efectos, efecto)
                if valor_efecto >= umbral:
                    resultados.append(preset)
        
        return sorted(resultados, key=lambda p: getattr(p.efectos, efecto), reverse=True)
    
    def obtener_presets_por_categoria(self, categoria: CategoriaEmocional) -> List[PresetEmocionalCompleto]:
        """Obtener presets por categoría"""
        return [p for p in self.presets.values() if p.categoria == categoria]
    
    def obtener_presets_por_contexto(self, contexto: str) -> List[PresetEmocionalCompleto]:
        """Obtener presets por contexto de uso"""
        contexto_lower = contexto.lower()
        return [p for p in self.presets.values() 
                if any(contexto_lower in c.lower() for c in p.contextos_recomendados)]
    
    def analizar_compatibilidad_presets(self, preset1: str, preset2: str) -> Dict[str, Any]:
        """Analizar compatibilidad entre dos presets"""
        p1 = self.obtener_preset(preset1)
        p2 = self.obtener_preset(preset2)
        
        if not p1 or not p2:
            return {"error": "Uno o ambos presets no encontrados"}
        
        # Compatibilidad por neurotransmisores
        nt_comunes = set(p1.neurotransmisores.keys()) & set(p2.neurotransmisores.keys())
        compatibilidad_nt = len(nt_comunes) / max(len(p1.neurotransmisores), len(p2.neurotransmisores))
        
        # Compatibilidad frecuencial
        diff_freq = abs(p1.frecuencia_base - p2.frecuencia_base)
        compatibilidad_freq = 1.0 - min(diff_freq / 20.0, 1.0)
        
        # Score final
        score_final = (compatibilidad_nt + compatibilidad_freq) / 2
        
        return {
            "score_compatibilidad": score_final,
            "neurotransmisores_comunes": list(nt_comunes),
            "diferencia_frecuencial": diff_freq,
            "recomendacion": self._generar_recomendacion_compatibilidad(score_final)
        }
    
    def _generar_recomendacion_compatibilidad(self, score: float) -> str:
        """Generar recomendación basada en compatibilidad"""
        if score > 0.8:
            return "Excelente - Ideal para secuencias largas"
        elif score > 0.6:
            return "Buena - Funciona bien en transiciones"
        elif score > 0.4:
            return "Moderada - Usar transiciones suaves"
        else:
            return "Baja - No recomendado en secuencia directa"
    
    def generar_secuencia_inteligente(self, objetivo_final: str, duracion_total_min: int = 30) -> List[Dict[str, Any]]:
        """Generar secuencia inteligente hacia un objetivo"""
        preset_objetivo = self.obtener_preset(objetivo_final)
        if not preset_objetivo:
            return []
        
        secuencia = []
        duracion_por_preset = duracion_total_min // 3
        
        # Preset inicial (preparatorio)
        preset_inicial = self._seleccionar_preset_inicial(preset_objetivo)
        secuencia.append({
            "preset": preset_inicial.nombre,
            "duracion_min": duracion_por_preset,
            "posicion": "inicial"
        })
        
        # Preset intermedio (puente)
        if duracion_total_min > 15:
            preset_intermedio = self._seleccionar_preset_intermedio(preset_inicial, preset_objetivo)
            secuencia.append({
                "preset": preset_intermedio.nombre,
                "duracion_min": duracion_por_preset,
                "posicion": "intermedio"
            })
        
        # Preset final (objetivo)
        secuencia.append({
            "preset": preset_objetivo.nombre,
            "duracion_min": duracion_total_min - len(secuencia) * duracion_por_preset,
            "posicion": "final"
        })
        
        return secuencia
    
    def _seleccionar_preset_inicial(self, objetivo: PresetEmocionalCompleto) -> PresetEmocionalCompleto:
        """Seleccionar preset inicial apropiado"""
        mapeo_inicial = {
            CategoriaEmocional.ESPIRITUAL: "calma_profunda",
            CategoriaEmocional.COGNITIVO: "regulacion_emocional",
            CategoriaEmocional.PERFORMANCE: "claridad_mental"
        }
        
        nombre_inicial = mapeo_inicial.get(objetivo.categoria, "seguridad_interior")
        return self.obtener_preset(nombre_inicial) or self.obtener_preset("calma_profunda")
    
    def _seleccionar_preset_intermedio(self, inicial: PresetEmocionalCompleto, objetivo: PresetEmocionalCompleto) -> PresetEmocionalCompleto:
        """Seleccionar preset intermedio como puente"""
        # Buscar preset con frecuencia intermedia
        for preset in self.presets.values():
            freq_min = min(inicial.frecuencia_base, objetivo.frecuencia_base)
            freq_max = max(inicial.frecuencia_base, objetivo.frecuencia_base)
            
            if freq_min < preset.frecuencia_base < freq_max:
                return preset
        
        # Fallback
        return self.obtener_preset("seguridad_interior") or inicial
    
    def exportar_configuracion_aurora(self) -> Dict[str, Any]:
        """Exportar configuración para Aurora V7"""
        return {
            "version": "v7.0",
            "total_presets": len(self.presets),
            "categorias": [cat.value for cat in CategoriaEmocional],
            "intensidades": [nivel.value for nivel in NivelIntensidad],
            "presets_validados": sum(1 for p in self.presets.values() 
                                   if p.nivel_evidencia in ["validado", "clinico"]),
            "confidence_promedio": sum(p.confidence_score for p in self.presets.values()) / len(self.presets),
            "neurotransmisores_unicos": list(set().union(*[p.neurotransmisores.keys() for p in self.presets.values()])),
            "frecuencias_rango": {
                "min": min(p.frecuencia_base for p in self.presets.values()), 
                "max": max(p.frecuencia_base for p in self.presets.values())
            }
        }
    
    def limpiar_cache(self):
        """Limpiar cache del gestor"""
        self.obtener_preset.cache_clear()
        logger.info("Cache de presets emocionales limpiado")

# Clase de compatibilidad para versiones anteriores
class EmotionalPreset:
    """Clase de compatibilidad con versiones anteriores"""
    _gestor = None
    
    # Presets simplificados para compatibilidad
    PRESETS = {
        "claridad_mental": {
            "nt": {"acetilcolina": 0.9, "dopamina": 0.6}, 
            "frecuencia_base": 14.5,
            "descripcion": "Lucidez y enfoque sereno"
        },
        "seguridad_interior": {
            "nt": {"gaba": 0.8, "serotonina": 0.7}, 
            "frecuencia_base": 8.0,
            "descripcion": "Confianza centrada"
        },
        "apertura_corazon": {
            "nt": {"oxitocina": 0.9, "serotonina": 0.5}, 
            "frecuencia_base": 7.2,
            "descripcion": "Aceptación amorosa y conexión emocional"
        },
        "estado_flujo": {
            "nt": {"dopamina": 0.9, "norepinefrina": 0.7}, 
            "frecuencia_base": 12.0,
            "descripcion": "Rendimiento óptimo y disfrute del presente"
        },
        "calma_profunda": {
            "nt": {"serotonina": 0.8, "gaba": 0.7}, 
            "frecuencia_base": 6.5,
            "descripcion": "Relajación total, serenidad corporal y mental"
        }
    }
    
    @classmethod
    def get(cls, nombre: str) -> Optional[Dict[str, Any]]:
        """Obtener preset con compatibilidad V7"""
        if cls._gestor is None:
            cls._gestor = GestorPresetsEmocionales()
        
        # Intentar V7 primero
        preset_v7 = cls._gestor.obtener_preset(nombre)
        if preset_v7:
            return {
                "nt": preset_v7.neurotransmisores,
                "frecuencia_base": preset_v7.frecuencia_base,
                "descripcion": preset_v7.descripcion
            }
        
        # Fallback a presets legacy
        return cls.PRESETS.get(nombre, None)

# Funciones de conveniencia
def crear_gestor_presets() -> GestorPresetsEmocionales:
    """Crear gestor de presets emocionales"""
    return GestorPresetsEmocionales()

def obtener_preset_completo(nombre: str) -> Optional[PresetEmocionalCompleto]:
    """Obtener preset completo por nombre"""
    return crear_gestor_presets().obtener_preset(nombre)

def buscar_presets_por_objetivo(objetivo: str) -> List[str]:
    """Buscar presets por objetivo emocional"""
    gestor = crear_gestor_presets()
    
    # Mapeo de objetivos a efectos
    mapeo_objetivos = {
        "concentracion": "concentracion",
        "relajacion": "calma", 
        "energia": "energia",
        "creatividad": "creatividad",
        "confianza": "confianza",
        "conexion": "conexion",
        "alegria": "alegria"
    }
    
    efecto = mapeo_objetivos.get(objetivo.lower(), objetivo.lower())
    presets = gestor.buscar_presets_por_efecto(efecto)
    return [p.nombre for p in presets]

def generar_experiencia_personalizada(objetivos: List[str], duracion_min: int = 20) -> Dict[str, Any]:
    """Generar experiencia personalizada basada en objetivos"""
    gestor = crear_gestor_presets()
    
    # Seleccionar presets para cada objetivo
    presets_seleccionados = []
    for objetivo in objetivos:
        presets_objetivo = buscar_presets_por_objetivo(objetivo)
        if presets_objetivo:
            presets_seleccionados.append(presets_objetivo[0])
    
    if not presets_seleccionados:
        return {"error": "No se encontraron presets para los objetivos especificados"}
    
    # Generar secuencia
    secuencia = gestor.generar_secuencia_inteligente(presets_seleccionados[-1], duracion_min)
    
    return {
        "objetivos": objetivos,
        "secuencia_generada": secuencia,
        "duracion_total": duracion_min,
        "recomendaciones": [
            "Usar auriculares de calidad para mejor experiencia",
            "Ambiente tranquilo sin interrupciones",
            "Postura cómoda y relajada"
        ]
    }

if __name__ == "__main__":
    print("🧠 Sistema de Presets Emocionales Aurora V7")
    print("=" * 50)
    
    # Crear gestor
    gestor = crear_gestor_presets()
    config = gestor.exportar_configuracion_aurora()
    
    print(f"📊 {config['total_presets']} presets emocionales disponibles")
    print(f"✅ {config['presets_validados']} validados científicamente")
    print(f"🎯 Confidence promedio: {config['confidence_promedio']:.1%}")
    
    print(f"\n📋 Por categoría:")
    for categoria in CategoriaEmocional:
        presets_cat = gestor.obtener_presets_por_categoria(categoria)
        if presets_cat:
            print(f"  • {categoria.value.title()}: {len(presets_cat)}")
    
    print(f"\n🔬 Ejemplo: Claridad Mental")
    preset = gestor.obtener_preset("claridad_mental")
    if preset:
        print(f"  Frecuencia: {preset.frecuencia_base} Hz")
        print(f"  Neurotransmisores: {list(preset.neurotransmisores.keys())}")
        print(f"  Contextos: {', '.join(preset.contextos_recomendados[:2])}")
    
    print(f"\n🧪 Ejemplo: Secuencia para estado de flujo")
    secuencia = gestor.generar_secuencia_inteligente("estado_flujo", 20)
    for fase in secuencia:
        print(f"  • {fase['preset']} ({fase['duracion_min']}min) - {fase['posicion']}")
    
    print(f"\n✅ Sistema de presets emocionales listo!")
