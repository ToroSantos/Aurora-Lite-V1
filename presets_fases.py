import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from functools import lru_cache
from objective_templates_optimized import ConfiguracionCapaV7, TemplateObjetivoV7, CategoriaObjetivo, NivelComplejidad, ModoActivacion

class TipoFase(Enum):
    PREPARACION = "preparacion"
    ACTIVACION = "activacion"
    INTENCION = "intencion"
    VISUALIZACION = "visualizacion"
    MANIFESTACION = "manifestacion"
    COLAPSO = "colapso"
    INTEGRACION = "integracion"
    ANCLAJE = "anclaje"
    CIERRE = "cierre"

class NivelIntensidad(Enum):
    SUAVE = "suave"
    MODERADO = "moderado"
    INTENSO = "intenso"
    EXTREMO = "extremo"

class EstadoConciencia(Enum):
    ALERTA_RELAJADO = "alerta_relajado"
    CONCENTRACION_SUAVE = "concentracion_suave"
    ENFOQUE_LASER = "enfoque_laser"
    ESTADO_FLUJO = "estado_flujo"
    VISION_INTERIOR = "vision_interior"
    EXPANSION_CONSCIENCIA = "expansion_consciencia"
    INTEGRACION_PROFUNDA = "integracion_profunda"
    PRESENCIA_TOTAL = "presencia_total"

@dataclass
class TransicionFase:
    duracion_ms: int = 3000
    tipo_transicion: str = "suave"
    modulacion_cruzada: bool = True
    fade_inteligente: bool = True
    sincronizacion_ondas: bool = True
    curva_transicion: str = "sigmoid"
    factor_coherencia: float = 0.8
    preservar_momentum: bool = True

@dataclass
class FaseConfigV7:
    nombre: str
    tipo_fase: TipoFase
    descripcion: str = ""
    neurotransmisor_principal: str = "GABA"
    neurotransmisores_secundarios: Dict[str, float] = field(default_factory=dict)
    emocional_preset: str = "seguridad_interior"
    estilo: str = "minimalista"
    duracion_minutos: float = 10.0
    duracion_minima: float = 5.0
    duracion_maxima: float = 30.0
    beat_base: float = 8.0
    ondas_objetivo: List[str] = field(default_factory=lambda: ["alpha"])
    frecuencia_dominante: float = 10.0
    capas: Dict[str, ConfiguracionCapaV7] = field(default_factory=dict)
    estado_consciencia: EstadoConciencia = EstadoConciencia.ALERTA_RELAJADO
    nivel_intensidad: NivelIntensidad = NivelIntensidad.MODERADO
    nivel_atencion_requerido: str = "medio"
    efectos_esperados: List[str] = field(default_factory=list)
    sensaciones_comunes: List[str] = field(default_factory=list)
    transicion_entrada: TransicionFase = field(default_factory=TransicionFase)
    transicion_salida: TransicionFase = field(default_factory=TransicionFase)
    adaptable_duracion: bool = True
    adaptable_intensidad: bool = True
    parametros_ajustables: List[str] = field(default_factory=list)
    base_cientifica: str = "validado"
    estudios_referencia: List[str] = field(default_factory=list)
    nivel_confianza: float = 0.85
    version: str = "v7.0"
    compatibilidad_v6: bool = True
    complejidad_renderizado: str = "medio"
    veces_usada: int = 0
    efectividad_promedio: float = 0.0
    feedback_usuarios: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._validar_configuracion()
        self._configurar_capas_automaticas()
        self._calcular_metricas_automaticas()
    
    def _validar_configuracion(self):
        if self.duracion_minima >= self.duracion_maxima:
            raise ValueError(f"Duración inválida en fase {self.nombre}")
    
    def _configurar_capas_automaticas(self):
        if not self.capas:
            self.capas = self._generar_capas_por_tipo_fase()
    
    def _generar_capas_por_tipo_fase(self) -> Dict[str, ConfiguracionCapaV7]:
        configs = {
            TipoFase.PREPARACION: {"neuro_wave": {"enabled": True, "modulacion_profundidad": 0.3, "prioridad": 3}, "binaural": {"enabled": True, "fade_in_ms": 2000, "prioridad": 4}, "wave_pad": {"enabled": True, "style": "gentle_preparation", "prioridad": 2}, "textured_noise": {"enabled": True, "style": "soft_comfort", "prioridad": 1}, "heartbeat": {"enabled": False}},
            TipoFase.INTENCION: {"neuro_wave": {"enabled": True, "modulacion_profundidad": 0.6, "prioridad": 5}, "binaural": {"enabled": True, "movimiento_espacial": True, "prioridad": 4}, "wave_pad": {"enabled": True, "style": "focused_intention", "prioridad": 3}, "textured_noise": {"enabled": True, "style": "crystalline_focus", "prioridad": 2}, "heartbeat": {"enabled": True, "style": "determined_pulse", "prioridad": 1}},
            TipoFase.VISUALIZACION: {"neuro_wave": {"enabled": True, "modulacion_profundidad": 0.7, "evolucion_temporal": True, "prioridad": 5}, "binaural": {"enabled": True, "patron_movimiento": "vision_spiral", "prioridad": 4}, "wave_pad": {"enabled": True, "style": "mystic_visions", "width_estereo": 1.8, "prioridad": 3}, "textured_noise": {"enabled": True, "style": "ethereal_whispers", "prioridad": 2}, "heartbeat": {"enabled": False}},
            TipoFase.COLAPSO: {"neuro_wave": {"enabled": True, "modulacion_profundidad": 0.4, "curva_evolucion": "release", "prioridad": 4}, "binaural": {"enabled": False}, "wave_pad": {"enabled": True, "style": "gentle_release", "prioridad": 3}, "textured_noise": {"enabled": True, "style": "flowing_release", "prioridad": 2}, "heartbeat": {"enabled": True, "style": "calming_rhythm", "prioridad": 1}},
            TipoFase.ANCLAJE: {"neuro_wave": {"enabled": False}, "binaural": {"enabled": True, "movimiento_espacial": False, "prioridad": 3}, "wave_pad": {"enabled": True, "style": "heart_opening", "prioridad": 4}, "textured_noise": {"enabled": False}, "heartbeat": {"enabled": False}}
        }
        
        config_base = configs.get(self.tipo_fase, configs[TipoFase.PREPARACION])
        capas_configuradas = {}
        
        for nombre_capa, config in config_base.items():
            capa_config = ConfiguracionCapaV7(enabled=config.get("enabled", True), modulacion_profundidad=config.get("modulacion_profundidad", 0.5), prioridad=config.get("prioridad", 2), style=config.get("style"), fade_in_ms=config.get("fade_in_ms", 1000), fade_out_ms=config.get("fade_out_ms", 1500))
            
            if nombre_capa == "neuro_wave" and capa_config.enabled:
                capa_config.carrier = self.frecuencia_dominante * 15
                capa_config.neurotransmisores = {self.neurotransmisor_principal.lower(): 0.8}
                capa_config.neurotransmisores.update(self.neurotransmisores_secundarios)
            elif nombre_capa == "binaural" and capa_config.enabled:
                capa_config.freq_l = 100.0
                capa_config.freq_r = 100.0 + self.beat_base
                capa_config.movimiento_espacial = config.get("movimiento_espacial", False)
                capa_config.patron_movimiento = config.get("patron_movimiento", "static")
            elif nombre_capa == "wave_pad" and capa_config.enabled:
                capa_config.width_estereo = config.get("width_estereo", 1.0)
                capa_config.evolucion_temporal = config.get("evolucion_temporal", False)
            elif nombre_capa == "textured_noise" and capa_config.enabled:
                capa_config.modulacion_profundidad = config.get("modulacion_profundidad", 0.2)
            elif nombre_capa == "heartbeat" and capa_config.enabled:
                capa_config.modulacion_velocidad = 0.05
            
            for attr, valor in config.items():
                if hasattr(capa_config, attr) and attr not in ["enabled", "prioridad"]:
                    setattr(capa_config, attr, valor)
            
            capas_configuradas[nombre_capa] = capa_config
        
        return capas_configuradas
    
    def _calcular_metricas_automaticas(self):
        capas_activas = sum(1 for capa in self.capas.values() if capa.enabled)
        self.complejidad_renderizado = "bajo" if capas_activas <= 2 else "medio" if capas_activas <= 4 else "alto"
        mapeo_confianza = {"experimental": 0.7, "validado": 0.85, "clinico": 0.95}
        if self.base_cientifica in mapeo_confianza:
            self.nivel_confianza = mapeo_confianza[self.base_cientifica]
    
    def generar_template_objetivo(self) -> TemplateObjetivoV7:
        return TemplateObjetivoV7(nombre=f"Fase: {self.nombre.title()}", descripcion=self.descripcion or f"Fase {self.nombre} del proceso consciente", categoria=CategoriaObjetivo.ESPIRITUAL, complejidad=NivelComplejidad.INTERMEDIO, emotional_preset=self.emocional_preset, style=self.estilo, layers=self.capas, neurotransmisores_principales={self.neurotransmisor_principal.lower(): 0.8, **self.neurotransmisores_secundarios}, ondas_cerebrales_objetivo=self.ondas_objetivo, frecuencia_dominante=self.beat_base, duracion_recomendada_min=int(self.duracion_minutos), duracion_minima_min=int(self.duracion_minima), duracion_maxima_min=int(self.duracion_maxima), efectos_esperados=self.efectos_esperados, evidencia_cientifica=self.base_cientifica, nivel_confianza=self.nivel_confianza, autor="Aurora_V7_PhaseSystem")

@dataclass 
class SecuenciaConsciente:
    nombre: str
    descripcion: str
    fases: List[FaseConfigV7] = field(default_factory=list)
    duracion_total_min: float = 0.0
    transiciones_automaticas: bool = True
    pausa_entre_fases: bool = False
    duracion_pausa_s: float = 30.0
    adaptable_a_usuario: bool = True
    niveles_disponibles: List[str] = field(default_factory=lambda: ["principiante", "intermedio", "avanzado"])
    mejor_momento: List[str] = field(default_factory=list)
    ambiente_recomendado: str = "tranquilo_privado"
    preparacion_sugerida: List[str] = field(default_factory=list)
    categoria: str = "manifestacion"
    nivel_experiencia_requerido: str = "intermedio"
    efectividad_reportada: float = 0.0
    
    def __post_init__(self):
        if self.fases:
            self.duracion_total_min = sum(fase.duracion_minutos for fase in self.fases)
            if self.pausa_entre_fases:
                self.duracion_total_min += (len(self.fases) - 1) * (self.duracion_pausa_s / 60.0)
    
    def agregar_fase(self, fase: FaseConfigV7, posicion: Optional[int] = None):
        if posicion is None:
            self.fases.append(fase)
        else:
            self.fases.insert(posicion, fase)
        self.duracion_total_min = sum(f.duracion_minutos for f in self.fases)
        if self.pausa_entre_fases and len(self.fases) > 1:
            self.duracion_total_min += (len(self.fases) - 1) * (self.duracion_pausa_s / 60.0)
    
    def validar_secuencia(self) -> Dict[str, Any]:
        validacion = {"valida": True, "advertencias": [], "sugerencias": [], "coherencia_neuroacustica": 0.0, "flujo_emocional": 0.0}
        if not self.fases:
            validacion["valida"] = False
            validacion["advertencias"].append("Secuencia sin fases")
            return validacion
        
        for i in range(len(self.fases) - 1):
            if not self._transicion_es_compatible(self.fases[i], self.fases[i + 1]):
                validacion["advertencias"].append(f"Transición abrupta entre {self.fases[i].nombre} y {self.fases[i + 1].nombre}")
        
        coherencias = [fase.nivel_confianza for fase in self.fases]
        validacion["coherencia_neuroacustica"] = sum(coherencias) / len(coherencias)
        
        if self.duracion_total_min < 30:
            validacion["sugerencias"].append("Secuencia muy corta para efectos profundos")
        elif self.duracion_total_min > 120:
            validacion["advertencias"].append("Secuencia muy larga puede causar fatiga")
        
        return validacion
    
    def _transicion_es_compatible(self, fase1: FaseConfigV7, fase2: FaseConfigV7) -> bool:
        compatibilidades = {"GABA": ["Serotonina", "Oxitocina", "Melatonina"], "Dopamina": ["Acetilcolina", "Norepinefrina", "Anandamida"], "Serotonina": ["GABA", "Oxitocina", "Melatonina"], "Anandamida": ["Serotonina", "Dopamina", "Oxitocina"], "Oxitocina": ["Serotonina", "GABA", "Endorfina"]}
        nt1, nt2 = fase1.neurotransmisor_principal, fase2.neurotransmisor_principal
        return nt1 == nt2 or nt2 in compatibilidades.get(nt1, [])

class GestorFasesConscientes:
    def __init__(self):
        self.fases_base: Dict[str, FaseConfigV7] = {}
        self.secuencias_predefinidas: Dict[str, SecuenciaConsciente] = {}
        self.estadisticas_uso: Dict[str, Dict[str, Any]] = {}
        self._inicializar_fases_base()
        self._crear_secuencias_predefinidas()
    
    def _inicializar_fases_base(self):
        self.fases_base["preparacion"] = FaseConfigV7(nombre="preparacion", tipo_fase=TipoFase.PREPARACION, descripcion="Base segura y estado receptivo", neurotransmisor_principal="GABA", neurotransmisores_secundarios={"serotonina": 0.6, "oxitocina": 0.4}, emocional_preset="seguridad_interior", estilo="minimalista", duracion_minutos=10.0, beat_base=8.0, ondas_objetivo=["alpha", "theta"], frecuencia_dominante=8.0, estado_consciencia=EstadoConciencia.ALERTA_RELAJADO, nivel_intensidad=NivelIntensidad.SUAVE, efectos_esperados=["Relajación profunda", "Sensación de seguridad", "Apertura receptiva", "Calma mental", "Preparación interior"], sensaciones_comunes=["Sensación de hundimiento", "Respiración profunda", "Músculos relajados", "Mente clara y tranquila"], nivel_atencion_requerido="bajo", base_cientifica="clinico", nivel_confianza=0.95)
        
        self.fases_base["intencion"] = FaseConfigV7(nombre="intencion", tipo_fase=TipoFase.INTENCION, descripcion="Activación y enfoque del propósito consciente", neurotransmisor_principal="Dopamina", neurotransmisores_secundarios={"acetilcolina": 0.7, "norepinefrina": 0.5}, emocional_preset="estado_flujo", estilo="futurista", duracion_minutos=12.0, beat_base=12.0, ondas_objetivo=["beta", "alpha"], frecuencia_dominante=12.0, estado_consciencia=EstadoConciencia.ENFOQUE_LASER, nivel_intensidad=NivelIntensidad.MODERADO, efectos_esperados=["Claridad de propósito", "Motivación elevada", "Enfoque laser", "Determinación", "Activación de la voluntad"], sensaciones_comunes=["Energía dirigida", "Mente muy clara", "Sensación de poder", "Concentración intensa"], nivel_atencion_requerido="alto", base_cientifica="validado", nivel_confianza=0.88, transicion_entrada=TransicionFase(duracion_ms=4000, tipo_transicion="gradual"))
        
        self.fases_base["visualizacion"] = FaseConfigV7(nombre="visualizacion", tipo_fase=TipoFase.VISUALIZACION, descripcion="Creación vivida de la realidad deseada", neurotransmisor_principal="Anandamida", neurotransmisores_secundarios={"dopamina": 0.6, "serotonina": 0.5}, emocional_preset="conexion_mistica", estilo="mistico", duracion_minutos=15.0, duracion_minima=10.0, duracion_maxima=25.0, beat_base=7.0, ondas_objetivo=["theta", "alpha"], frecuencia_dominante=7.0, estado_consciencia=EstadoConciencia.VISION_INTERIOR, nivel_intensidad=NivelIntensidad.INTENSO, efectos_esperados=["Visiones vívidas", "Creatividad expandida", "Conexión con el futuro deseado", "Imaginación potenciada", "Experiencia multisensorial"], sensaciones_comunes=["Imágenes muy claras", "Emociones intensas", "Sensación de estar allí", "Conexión profunda con el deseo"], nivel_atencion_requerido="extremo", adaptable_duracion=True, parametros_ajustables=["intensidad_visual", "duracion", "profundidad"], base_cientifica="validado", nivel_confianza=0.82, transicion_entrada=TransicionFase(duracion_ms=5000, curva_transicion="exponential"))
        
        self.fases_base["colapso"] = FaseConfigV7(nombre="colapso", tipo_fase=TipoFase.COLAPSO, descripcion="Liberación consciente y entrega al proceso universal", neurotransmisor_principal="Serotonina", neurotransmisores_secundarios={"gaba": 0.7, "oxitocina": 0.5}, emocional_preset="regulacion_emocional", estilo="organico", duracion_minutos=10.0, beat_base=6.5, ondas_objetivo=["theta", "alpha"], frecuencia_dominante=6.5, estado_consciencia=EstadoConciencia.EXPANSION_CONSCIENCIA, nivel_intensidad=NivelIntensidad.MODERADO, efectos_esperados=["Liberación emocional", "Sensación de entrega", "Paz profunda", "Disolución de resistencias", "Confianza en el proceso"], sensaciones_comunes=["Liberación física", "Suspiro profundo", "Sensación de fluir", "Paz y confianza"], nivel_atencion_requerido="medio", base_cientifica="validado", nivel_confianza=0.90, transicion_entrada=TransicionFase(duracion_ms=3000, tipo_transicion="suave"))
        
        self.fases_base["anclaje"] = FaseConfigV7(nombre="anclaje", tipo_fase=TipoFase.ANCLAJE, descripcion="Anclaje energético y sellado del proceso consciente", neurotransmisor_principal="Oxitocina", neurotransmisores_secundarios={"serotonina": 0.6, "endorfina": 0.4}, emocional_preset="apertura_corazon", estilo="sutil", duracion_minutos=8.0, beat_base=7.2, ondas_objetivo=["alpha"], frecuencia_dominante=7.2, estado_consciencia=EstadoConciencia.PRESENCIA_TOTAL, nivel_intensidad=NivelIntensidad.SUAVE, efectos_esperados=["Apertura del corazón", "Conexión amorosa", "Integración completa", "Gratitud profunda", "Sellado energético"], sensaciones_comunes=["Calor en el pecho", "Sensación de amor", "Gratitud profunda", "Conexión universal"], nivel_atencion_requerido="bajo", base_cientifica="validado", nivel_confianza=0.93, transicion_entrada=TransicionFase(duracion_ms=2000, tipo_transicion="suave"), transicion_salida=TransicionFase(duracion_ms=4000, tipo_transicion="suave"))
        
        self.fases_base["activacion_cuantica"] = FaseConfigV7(nombre="activacion_cuantica", tipo_fase=TipoFase.ACTIVACION, descripcion="Activación de potencial cuántico y multidimensional", neurotransmisor_principal="Anandamida", neurotransmisores_secundarios={"dopamina": 0.8, "acetilcolina": 0.6}, emocional_preset="expansion_creativa", estilo="cuantico", duracion_minutos=20.0, duracion_minima=15.0, duracion_maxima=30.0, beat_base=12.0, ondas_objetivo=["gamma", "theta"], frecuencia_dominante=12.0, estado_consciencia=EstadoConciencia.EXPANSION_CONSCIENCIA, nivel_intensidad=NivelIntensidad.EXTREMO, efectos_esperados=["Activación cuántica", "Expansión multidimensional", "Conexión universal", "Potencial ilimitado", "Transformación profunda"], nivel_atencion_requerido="extremo", base_cientifica="experimental", nivel_confianza=0.75)
        
        self.fases_base["integracion_profunda"] = FaseConfigV7(nombre="integracion_profunda", tipo_fase=TipoFase.INTEGRACION, descripcion="Integración celular y energética profunda de la experiencia", neurotransmisor_principal="Serotonina", neurotransmisores_secundarios={"oxitocina": 0.7, "endorfina": 0.5, "gaba": 0.4}, emocional_preset="sanacion_integral", estilo="medicina_sagrada", duracion_minutos=18.0, beat_base=7.83, ondas_objetivo=["alpha", "theta"], frecuencia_dominante=7.83, estado_consciencia=EstadoConciencia.INTEGRACION_PROFUNDA, nivel_intensidad=NivelIntensidad.MODERADO, efectos_esperados=["Integración celular", "Sanación energética", "Renovación completa", "Armonización sistémica", "Sellado cuántico"], base_cientifica="validado", nivel_confianza=0.87)
    
    def _crear_secuencias_predefinidas(self):
        self.secuencias_predefinidas["manifestacion_clasica"] = SecuenciaConsciente(nombre="Manifestación Consciente Clásica", descripcion="Secuencia original de 5 fases para manifestación consciente", fases=[self.fases_base["preparacion"], self.fases_base["intencion"], self.fases_base["visualizacion"], self.fases_base["colapso"], self.fases_base["anclaje"]], categoria="manifestacion", nivel_experiencia_requerido="intermedio", mejor_momento=["mañana_temprano", "noche"], ambiente_recomendado="privado_silencioso", preparacion_sugerida=["Definir intención clara", "Espacio limpio y ordenado", "Sin interrupciones", "Hidratación adecuada"])
        
        self.secuencias_predefinidas["transformacion_cuantica"] = SecuenciaConsciente(nombre="Transformación Cuántica Avanzada", descripcion="Secuencia avanzada con activación cuántica e integración profunda", fases=[self.fases_base["preparacion"], self.fases_base["activacion_cuantica"], self.fases_base["visualizacion"], self.fases_base["colapso"], self.fases_base["integracion_profunda"], self.fases_base["anclaje"]], categoria="transformacion", nivel_experiencia_requerido="avanzado", mejor_momento=["noche"], ambiente_recomendado="sagrado_preparado", preparacion_sugerida=["Experiencia previa requerida", "Ayuno ligero recomendado", "Espacio energéticamente limpio", "Tiempo sin compromisos posteriores"])
        
        self.secuencias_predefinidas["intencion_express"] = SecuenciaConsciente(nombre="Intención Express", descripcion="Secuencia corta y efectiva para intenciones rápidas", fases=[FaseConfigV7(nombre="preparacion_rapida", tipo_fase=TipoFase.PREPARACION, neurotransmisor_principal="GABA", emocional_preset="seguridad_interior", estilo="minimalista", duracion_minutos=5.0, beat_base=8.0, descripcion="Preparación rápida y efectiva"), FaseConfigV7(nombre="intencion_visual", tipo_fase=TipoFase.MANIFESTACION, neurotransmisor_principal="Dopamina", neurotransmisores_secundarios={"anandamida": 0.6}, emocional_preset="estado_flujo", estilo="futurista", duracion_minutos=12.0, beat_base=10.0, descripcion="Intención y visualización combinadas"), FaseConfigV7(nombre="anclaje_rapido", tipo_fase=TipoFase.ANCLAJE, neurotransmisor_principal="Oxitocina", emocional_preset="apertura_corazon", estilo="sutil", duracion_minutos=5.0, beat_base=7.0, descripcion="Anclaje rápido y efectivo")], categoria="express", nivel_experiencia_requerido="principiante", mejor_momento=["cualquier_momento"], ambiente_recomendado="cualquier_lugar_tranquilo")
    
    @lru_cache(maxsize=64)
    def obtener_fase(self, nombre: str) -> Optional[FaseConfigV7]:
        return self.fases_base.get(nombre.lower())
    
    def obtener_secuencia(self, nombre: str) -> Optional[SecuenciaConsciente]:
        return self.secuencias_predefinidas.get(nombre.lower())
    
    def crear_secuencia_personalizada(self, nombre: str, fases_nombres: List[str], configuracion: Optional[Dict[str, Any]] = None) -> SecuenciaConsciente:
        config = configuracion or {}
        fases_personalizadas = []
        for nombre_fase in fases_nombres:
            fase = self.obtener_fase(nombre_fase)
            if fase:
                if nombre_fase in config:
                    fase_personalizada = self._personalizar_fase(fase, config[nombre_fase])
                    fases_personalizadas.append(fase_personalizada)
                else:
                    fases_personalizadas.append(fase)
        return SecuenciaConsciente(nombre=nombre, descripcion=config.get("descripcion", f"Secuencia personalizada: {nombre}"), fases=fases_personalizadas, categoria=config.get("categoria", "personalizada"), nivel_experiencia_requerido=config.get("nivel", "intermedio"))
    
    def _personalizar_fase(self, fase_base: FaseConfigV7, modificaciones: Dict[str, Any]) -> FaseConfigV7:
        import copy
        fase_personalizada = copy.deepcopy(fase_base)
        for campo, valor in modificaciones.items():
            if hasattr(fase_personalizada, campo):
                setattr(fase_personalizada, campo, valor)
        fase_personalizada._calcular_metricas_automaticas()
        return fase_personalizada
    
    def analizar_secuencia(self, secuencia: SecuenciaConsciente) -> Dict[str, Any]:
        analisis = {"duracion_total": secuencia.duracion_total_min, "numero_fases": len(secuencia.fases), "validacion": secuencia.validar_secuencia(), "neurotransmisores_utilizados": [], "estados_consciencia": [], "niveles_intensidad": [], "complejidad_promedio": 0.0, "recomendaciones": []}
        nts = set()
        estados = []
        intensidades = []
        complejidades = []
        for fase in secuencia.fases:
            nts.add(fase.neurotransmisor_principal)
            nts.update(fase.neurotransmisores_secundarios.keys())
            estados.append(fase.estado_consciencia.value)
            intensidades.append(fase.nivel_intensidad.value)
            capas_activas = sum(1 for capa in fase.capas.values() if capa.enabled)
            complejidades.append(capas_activas)
        analisis["neurotransmisores_utilizados"] = list(nts)
        analisis["estados_consciencia"] = estados
        analisis["niveles_intensidad"] = intensidades
        analisis["complejidad_promedio"] = sum(complejidades) / len(complejidades) if complejidades else 0
        if analisis["duracion_total"] > 90:
            analisis["recomendaciones"].append("Considerar dividir en dos sesiones")
        if analisis["complejidad_promedio"] > 4:
            analisis["recomendaciones"].append("Secuencia compleja - requiere experiencia")
        if len(nts) < 3:
            analisis["recomendaciones"].append("Considerar mayor variedad de neurotransmisores")
        return analisis
    
    def recomendar_secuencia(self, perfil_usuario: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        recomendaciones = []
        for nombre, secuencia in self.secuencias_predefinidas.items():
            compatibilidad = self._calcular_compatibilidad_secuencia(secuencia, perfil_usuario)
            if compatibilidad > 0.3:
                razon = self._generar_razon_recomendacion_secuencia(secuencia, perfil_usuario)
                recomendaciones.append((nombre, compatibilidad, razon))
        recomendaciones.sort(key=lambda x: x[1], reverse=True)
        return recomendaciones
    
    def _calcular_compatibilidad_secuencia(self, secuencia: SecuenciaConsciente, perfil: Dict[str, Any]) -> float:
        compatibilidad = 0.0
        experiencia = perfil.get("experiencia", "principiante")
        nivel_secuencia = secuencia.nivel_experiencia_requerido
        if experiencia == nivel_secuencia:
            compatibilidad += 0.4
        elif (experiencia == "intermedio" and nivel_secuencia == "principiante") or (experiencia == "avanzado" and nivel_secuencia in ["principiante", "intermedio"]):
            compatibilidad += 0.3
        elif experiencia == "principiante" and nivel_secuencia == "intermedio":
            compatibilidad += 0.2
        else:
            compatibilidad -= 0.2
        tiempo_disponible = perfil.get("tiempo_disponible_min", 60)
        if secuencia.duracion_total_min <= tiempo_disponible:
            compatibilidad += 0.2
        elif secuencia.duracion_total_min <= tiempo_disponible * 1.2:
            compatibilidad += 0.1
        else:
            compatibilidad -= 0.1
        objetivos = perfil.get("objetivos", [])
        categoria_secuencia = secuencia.categoria
        mapeo_objetivos = {"manifestacion": ["manifestar", "crear", "atraer", "lograr"], "sanacion": ["sanar", "curar", "equilibrar", "restaurar"], "meditacion": ["meditar", "calmar", "centrar", "paz"], "transformacion": ["transformar", "cambiar", "evolucionar", "crecer"]}
        objetivos_categoria = mapeo_objetivos.get(categoria_secuencia, [])
        for objetivo in objetivos:
            if any(palabra in objetivo.lower() for palabra in objetivos_categoria):
                compatibilidad += 0.15
        return max(0.0, compatibilidad)
    
    def _generar_razon_recomendacion_secuencia(self, secuencia: SecuenciaConsciente, perfil: Dict[str, Any]) -> str:
        razones = []
        experiencia = perfil.get("experiencia", "principiante")
        if secuencia.nivel_experiencia_requerido == experiencia:
            razones.append(f"Perfecto para nivel {experiencia}")
        tiempo_disponible = perfil.get("tiempo_disponible_min", 60)
        if secuencia.duracion_total_min <= tiempo_disponible:
            razones.append(f"Duración ideal ({int(secuencia.duracion_total_min)} min)")
        if secuencia.categoria in ["manifestacion", "transformacion"]:
            razones.append("Proceso de cambio profundo")
        elif secuencia.categoria == "express":
            razones.append("Resultados rápidos")
        return " • ".join(razones) if razones else "Compatible con tu perfil"
    
    def exportar_estadisticas(self) -> Dict[str, Any]:
        return {"version": "v7.0", "total_fases_base": len(self.fases_base), "total_secuencias": len(self.secuencias_predefinidas), "neurotransmisores_disponibles": list(set(fase.neurotransmisor_principal for fase in self.fases_base.values())), "duraciones_secuencias": {nombre: secuencia.duracion_total_min for nombre, secuencia in self.secuencias_predefinidas.items()}, "fases_por_tipo": {tipo.value: len([f for f in self.fases_base.values() if f.tipo_fase == tipo]) for tipo in TipoFase}}

class PhaseConfig:
    def __init__(self, nombre, neurotransmisor, emocional_preset, estilo, duracion_minutos=10, capas=None, beat_base=8.0):
        self.nombre = nombre
        self.neurotransmisor = neurotransmisor
        self.emocional_preset = emocional_preset
        self.estilo = estilo
        self.duracion_minutos = duracion_minutos
        self.capas = capas or {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}
        self.beat_base = beat_base

def _generar_phases_presets_v6():
    gestor = GestorFasesConscientes()
    phases_v6 = []
    fases_principales = ["preparacion", "intencion", "visualizacion", "colapso", "anclaje"]
    for nombre_fase in fases_principales:
        fase_v7 = gestor.obtener_fase(nombre_fase)
        if fase_v7:
            capas_v6 = {}
            for nombre_capa, config_capa in fase_v7.capas.items():
                capas_v6[nombre_capa] = config_capa.enabled
            phase_v6 = PhaseConfig(nombre=fase_v7.nombre, neurotransmisor=fase_v7.neurotransmisor_principal, emocional_preset=fase_v7.emocional_preset, estilo=fase_v7.estilo, duracion_minutos=fase_v7.duracion_minutos, capas=capas_v6, beat_base=fase_v7.beat_base)
            phases_v6.append(phase_v6)
    return phases_v6

PHASES_PRESETS = _generar_phases_presets_v6()

def crear_gestor_fases() -> GestorFasesConscientes:
    return GestorFasesConscientes()

def obtener_secuencia_manifestacion() -> SecuenciaConsciente:
    gestor = crear_gestor_fases()
    return gestor.obtener_secuencia("manifestacion_clasica")

def crear_secuencia_rapida() -> SecuenciaConsciente:
    gestor = crear_gestor_fases()
    return gestor.obtener_secuencia("intencion_express")

def personalizar_secuencia(fases_nombres: List[str], configuracion: Dict[str, Any] = None) -> SecuenciaConsciente:
    gestor = crear_gestor_fases()
    return gestor.crear_secuencia_personalizada("personalizada", fases_nombres, configuracion)

def analizar_eficacia_secuencia(secuencia: SecuenciaConsciente) -> Dict[str, Any]:
    gestor = crear_gestor_fases()
    return gestor.analizar_secuencia(secuencia)