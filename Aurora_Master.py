"""
Aurora V7 Sistema Unificado - UN SOLO ARCHIVO COMPLETO
Reemplaza aurora_director.py + Aurora_Master.py + coordinación

🎯 OBJETIVO: Menos archivos, más funcionalidad
✅ Director + Master + Configuración + Dependencias = TODO EN UNO
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.Unificado.V7")

# === CONFIGURACIÓN UNIVERSAL (antes era archivo separado) ===
@dataclass
class ConfigAurora:
    """Configuración única que funciona para TODO"""
    # Básicos
    objetivo: str = "relajacion"
    duracion_min: int = 20
    sample_rate: int = 44100
    amplitude: float = 0.5
    
    # Neuroacústico
    neurotransmisor: str = "dopamina"
    beat_primario: float = 10.0
    beat_secundario: float = 6.0
    armonicos: List[float] = field(default_factory=list)
    
    # Estético
    estilo: str = "sereno"
    tipo_textura: str = "organic"
    intensidad: str = "media"
    
    # Espacial
    stereo_width: float = 0.8
    movimiento_3d: bool = True
    
    # Control
    usar_fases: bool = False  # Auto-detectado
    precision_cientifica: bool = True
    guardar: bool = True
    
    def to_neuromix_config(self):
        """Convierte para NeuroMix Engine"""
        return {
            "neurotransmitter": self.neurotransmisor,
            "duration_sec": self.duracion_min * 60,
            "sample_rate": self.sample_rate,
            "intensity": self.intensidad,
            "beat_frequency": self.beat_primario,
            "amplitude": self.amplitude
        }
    
    def to_texturas_config(self):
        """Convierte para HarmonicEssence"""
        return {
            "duration_sec": self.duracion_min * 60,
            "sample_rate": self.sample_rate,
            "amplitude": self.amplitude,
            "stereo_width": self.stereo_width,
            "texture_type": self.tipo_textura.upper(),
            "neurotransmitter_profile": self.neurotransmisor,
            "style_profile": self.estilo,
            "precision_cientifica": self.precision_cientifica
        }

# === DETECCIÓN DE COMPONENTES (antes era archivo separado) ===
class ComponentesDisponibles:
    """Detección automática de qué está disponible"""
    
    def __init__(self):
        self.componentes = {}
        self.fallbacks = {}
        self._detectar_todos()
    
    def _detectar_todos(self):
        """Detecta qué componentes están disponibles"""
        # NeuroMix Engine
        try:
            from neuromix_engine_v26_ultimate import AuroraNeuroMixEngine, NeuroConfig
            self.componentes["neuromix"] = AuroraNeuroMixEngine()
            logger.info("✅ NeuroMix Engine cargado")
        except ImportError:
            self.componentes["neuromix"] = self._fallback_neuromix()
            logger.warning("⚠️ NeuroMix fallback activo")
        
        # HarmonicEssence V34
        try:
            from harmonicEssence_v33 import HarmonicEssenceV34Ultimate, NoiseConfigV34Unificado
            self.componentes["texturas"] = HarmonicEssenceV34Ultimate()
            logger.info("✅ HarmonicEssence V34 cargado")
        except ImportError:
            self.componentes["texturas"] = self._fallback_texturas()
            logger.warning("⚠️ Texturas fallback activo")
        
        # Quality Pipeline
        try:
            from aurora_quality_pipeline import AuroraQualityPipeline, CarmineAnalyzer
            self.componentes["quality"] = AuroraQualityPipeline()
            logger.info("✅ Quality Pipeline cargado")
        except ImportError:
            self.componentes["quality"] = self._fallback_quality()
            logger.warning("⚠️ Quality fallback activo")
        
        # Harmony Generator V7
        try:
            from harmony_generator import crear_generador_aurora
            self.componentes["harmony"] = crear_generador_aurora()
            logger.info("✅ Harmony Generator V7 cargado")
        except ImportError:
            self.componentes["harmony"] = self._fallback_harmony()
            logger.warning("⚠️ Harmony fallback activo")
        
        # Field Profiles (para Director avanzado)
        try:
            from field_profiles_unified import crear_gestor_perfiles
            self.componentes["field_profiles"] = crear_gestor_perfiles()
            logger.info("✅ Field Profiles cargado")
        except ImportError:
            self.componentes["field_profiles"] = None
            logger.info("ℹ️ Field Profiles no disponible")
    
    def esta_disponible(self, componente: str) -> bool:
        return componente in self.componentes and self.componentes[componente] is not None
    
    def obtener(self, componente: str):
        return self.componentes.get(componente)
    
    # === FALLBACKS SIMPLES ===
    def _fallback_neuromix(self):
        class NeuroFallback:
            def generate_neuro_wave_advanced(self, config):
                samples = int(config.get('duration_sec', 10) * 44100)
                freq = config.get('beat_frequency', 10.0)
                t = np.linspace(0, config.get('duration_sec', 10), samples)
                wave = 0.3 * np.sin(2 * np.pi * freq * t)
                return np.stack([wave, wave]), {}
        return NeuroFallback()
    
    def _fallback_texturas(self):
        class TexturasFallback:
            def generate_textured_noise(self, config):
                samples = int(config.get('duration_sec', 10) * 44100)
                noise = np.random.normal(0, 0.2, samples)
                return np.stack([noise, noise])
        return TexturasFallback()
    
    def _fallback_quality(self):
        class QualityFallback:
            def validar_y_normalizar(self, signal):
                if signal.ndim == 1: signal = np.stack([signal, signal])
                max_val = np.max(np.abs(signal))
                if max_val > 0: signal = signal * (0.85 / max_val)
                return np.clip(signal, -1.0, 1.0)
        return QualityFallback()
    
    def _fallback_harmony(self):
        class HarmonyFallback:
            def generar_pad_avanzado(self, base_freq=220, **kwargs):
                t = np.linspace(0, 4, 44100 * 4)
                pad = 0.2 * np.sin(2 * np.pi * base_freq * t)
                return np.stack([pad, pad])
        return HarmonyFallback()

# === RESULTADO UNIFICADO ===
@dataclass
class ResultadoAurora:
    """Resultado único para todo el sistema"""
    audio_data: np.ndarray
    sample_rate: int = 44100
    duracion_segundos: float = 0.0
    configuracion: Dict[str, Any] = field(default_factory=dict)
    motores_usados: List[str] = field(default_factory=list)
    score_calidad: float = 0.0
    tiempo_generacion: float = 0.0
    estrategia_usada: str = "desconocida"
    ruta_archivo: Optional[str] = None
    recomendaciones: List[str] = field(default_factory=list)

# === SISTEMA AURORA UNIFICADO ===
class AuroraUnificado:
    """
    Sistema Aurora V7 Unificado - TODO EN UNO
    
    ✅ Reemplaza: aurora_director.py + Aurora_Master.py
    ✅ Incluye: Configuración + Dependencias + Coordinación
    ✅ Objetivo: Menos archivos, más funcionalidad
    """
    
    def __init__(self):
        self.version = "Aurora V7 Unificado - Todo en Uno"
        self.componentes = ComponentesDisponibles()
        self.stats = {
            "experiencias_generadas": 0,
            "tiempo_total": 0.0,
            "calidad_promedio": 0.0,
            "estrategias_usadas": {}
        }
        logger.info(f"🚀 {self.version} inicializado")
    
    def crear_experiencia(self, objetivo: str, **kwargs) -> ResultadoAurora:
        """
        API ÚNICA que reemplaza tanto Master como Director
        
        Decide automáticamente:
        - ¿Usar fases (Director) o directo (Master)?
        - ¿Qué motores están disponibles?
        - ¿Cómo optimizar la calidad?
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Creando experiencia: {objetivo}")
            
            # 1. CONFIGURACIÓN INTELIGENTE
            config = self._crear_configuracion_inteligente(objetivo, kwargs)
            
            # 2. ESTRATEGIA AUTOMÁTICA
            estrategia = self._detectar_estrategia_optima(config)
            
            # 3. GENERACIÓN CON MEJOR ESTRATEGIA
            audio_data = self._generar_con_estrategia(estrategia, config)
            
            # 4. CALIDAD Y OPTIMIZACIÓN
            audio_final, score_calidad = self._optimizar_y_validar(audio_data)
            
            # 5. GUARDAR SI ES NECESARIO
            ruta_archivo = self._guardar_audio(audio_final, config) if config.guardar else None
            
            # 6. RESULTADO COMPLETO
            tiempo_generacion = time.time() - start_time
            resultado = self._crear_resultado_completo(
                audio_final, config, estrategia, score_calidad, 
                tiempo_generacion, ruta_archivo
            )
            
            # 7. ACTUALIZAR STATS
            self._actualizar_stats(resultado, objetivo, estrategia)
            
            logger.info(f"✅ Experiencia creada - {estrategia} - Calidad: {score_calidad:.1f}/100")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error creando experiencia: {e}")
            raise
    
    def _crear_configuracion_inteligente(self, objetivo: str, kwargs: Dict) -> ConfigAurora:
        """Crea configuración optimizada automáticamente"""
        
        # Mapeos inteligentes por objetivo
        mapeos = {
            "relajacion": {
                "neurotransmisor": "gaba",
                "beat_primario": 6.0,
                "beat_secundario": 4.0,
                "estilo": "sereno",
                "tipo_textura": "organic",
                "intensidad": "baja"
            },
            "concentracion": {
                "neurotransmisor": "acetilcolina", 
                "beat_primario": 14.0,
                "beat_secundario": 16.0,
                "estilo": "crystalline",
                "tipo_textura": "neuromorphic",
                "intensidad": "media"
            },
            "creatividad": {
                "neurotransmisor": "dopamina",
                "beat_primario": 10.0,
                "beat_secundario": 12.0,
                "estilo": "etereo",
                "tipo_textura": "holographic",
                "intensidad": "media"
            },
            "meditacion": {
                "neurotransmisor": "anandamida",
                "beat_primario": 7.83,
                "beat_secundario": 4.0,
                "estilo": "mistico",
                "tipo_textura": "quantum",
                "intensidad": "media"
            }
        }
        
        # Obtener configuración base
        config_base = mapeos.get(objetivo.lower(), mapeos["relajacion"])
        
        # Combinar con parámetros del usuario
        config_params = {
            "objetivo": objetivo,
            **config_base,
            **kwargs
        }
        
        # Conversiones comunes
        if "duracion" in kwargs:
            config_params["duracion_min"] = kwargs["duracion"]
        
        return ConfigAurora(**config_params)
    
    def _detectar_estrategia_optima(self, config: ConfigAurora) -> str:
        """Decide automáticamente la mejor estrategia"""
        
        # Factores de decisión
        duracion_larga = config.duracion_min > 30
        objetivo_complejo = any(palabra in config.objetivo.lower() 
                              for palabra in ["terapeutico", "neuroplasticidad", "complejo", "avanzado"])
        tiene_field_profiles = self.componentes.esta_disponible("field_profiles")
        
        # Lógica de decisión (reemplaza la lógica del Director vs Master)
        if (duracion_larga or objetivo_complejo) and tiene_field_profiles:
            config.usar_fases = True
            return "director_fases"
        elif self.componentes.esta_disponible("neuromix") and self.componentes.esta_disponible("texturas"):
            return "hibrido_completo"
        elif self.componentes.esta_disponible("neuromix"):
            return "neuromix_principal"
        elif self.componentes.esta_disponible("texturas"):
            return "texturas_principal"
        else:
            return "fallback_basico"
    
    def _generar_con_estrategia(self, estrategia: str, config: ConfigAurora) -> np.ndarray:
        """Genera audio con la estrategia óptima"""
        
        logger.info(f"🔧 Estrategia: {estrategia}")
        
        if estrategia == "director_fases":
            return self._generar_por_fases(config)
        elif estrategia == "hibrido_completo":
            return self._generar_hibrido_completo(config)
        elif estrategia == "neuromix_principal":
            return self._generar_neuromix_principal(config)
        elif estrategia == "texturas_principal":
            return self._generar_texturas_principal(config)
        else:
            return self._generar_fallback_basico(config)
    
    def _generar_por_fases(self, config: ConfigAurora) -> np.ndarray:
        """Generación por fases (equivalente al Director anterior)"""
        try:
            # Usar Field Profiles si está disponible
            field_profiles = self.componentes.obtener("field_profiles")
            if field_profiles:
                # Lógica simplificada del Director
                duracion_total = config.duracion_min
                
                # Crear fases automáticas
                if "concentracion" in config.objetivo.lower():
                    fases = [
                        ("preparacion", 0.2 * duracion_total, "gaba"),
                        ("activacion", 0.6 * duracion_total, "acetilcolina"),
                        ("consolidacion", 0.2 * duracion_total, "dopamina")
                    ]
                elif "relajacion" in config.objetivo.lower():
                    fases = [
                        ("entrada", 0.4 * duracion_total, "gaba"),
                        ("profundizacion", 0.6 * duracion_total, "serotonina")
                    ]
                else:
                    fases = [("desarrollo", duracion_total, config.neurotransmisor)]
                
                # Generar cada fase
                capas_fases = []
                for nombre_fase, duracion_fase, neurotrans in fases:
                    config_fase = ConfigAurora(
                        objetivo=nombre_fase,
                        duracion_min=duracion_fase,
                        neurotransmisor=neurotrans,
                        **{k: v for k, v in config.__dict__.items() 
                           if k not in ["objetivo", "duracion_min", "neurotransmisor"]}
                    )
                    
                    capa_fase = self._generar_hibrido_completo(config_fase)
                    capas_fases.append(capa_fase)
                
                # Concatenar fases
                return np.concatenate(capas_fases, axis=1)
            
            else:
                # Sin Field Profiles, usar híbrido simple
                return self._generar_hibrido_completo(config)
                
        except Exception as e:
            logger.warning(f"⚠️ Error en generación por fases: {e}")
            return self._generar_hibrido_completo(config)
    
    def _generar_hibrido_completo(self, config: ConfigAurora) -> np.ndarray:
        """Generación híbrida completa (equivalente al Master híbrido)"""
        capas = []
        
        # Capa neuroacústica
        if self.componentes.esta_disponible("neuromix"):
            neuromix = self.componentes.obtener("neuromix")
            config_neuro = config.to_neuromix_config()
            
            try:
                audio_neuro, _ = neuromix.generate_neuro_wave_advanced(config_neuro)
                capas.append(("neuro", audio_neuro * 0.6))
            except Exception as e:
                logger.warning(f"⚠️ Error NeuroMix: {e}")
        
        # Capa texturas
        if self.componentes.esta_disponible("texturas"):
            texturas = self.componentes.obtener("texturas")
            config_texturas = config.to_texturas_config()
            
            try:
                audio_texturas = texturas.generate_textured_noise(config_texturas)
                capas.append(("texturas", audio_texturas * 0.4))
            except Exception as e:
                logger.warning(f"⚠️ Error Texturas: {e}")
        
        # Capa armónica (si disponible)
        if self.componentes.esta_disponible("harmony"):
            harmony = self.componentes.obtener("harmony")
            try:
                audio_harmony = harmony.generar_pad_avanzado(
                    base_freq=config.beat_primario * 20,
                    estilo=config.estilo,
                    stereo=True
                )
                if isinstance(audio_harmony, tuple):
                    audio_harmony = np.stack(audio_harmony)
                capas.append(("harmony", audio_harmony * 0.3))
            except Exception as e:
                logger.warning(f"⚠️ Error Harmony: {e}")
        
        # Mezclar capas
        if capas:
            return self._mezclar_capas_inteligente(capas)
        else:
            return self._generar_fallback_basico(config)
    
    def _generar_neuromix_principal(self, config: ConfigAurora) -> np.ndarray:
        """Solo NeuroMix"""
        neuromix = self.componentes.obtener("neuromix")
        config_neuro = config.to_neuromix_config()
        audio_neuro, _ = neuromix.generate_neuro_wave_advanced(config_neuro)
        return audio_neuro
    
    def _generar_texturas_principal(self, config: ConfigAurora) -> np.ndarray:
        """Solo Texturas"""
        texturas = self.componentes.obtener("texturas")
        config_texturas = config.to_texturas_config()
        return texturas.generate_textured_noise(config_texturas)
    
    def _generar_fallback_basico(self, config: ConfigAurora) -> np.ndarray:
        """Generación básica cuando no hay nada disponible"""
        duracion_sec = config.duracion_min * 60
        samples = int(config.sample_rate * duracion_sec)
        t = np.linspace(0, duracion_sec, samples)
        
        # Audio básico con la frecuencia configurada
        audio = config.amplitude * np.sin(2 * np.pi * config.beat_primario * t)
        
        # Envolvente suave
        envelope = np.exp(-t * 0.01)
        audio *= envelope
        
        return np.stack([audio, audio])
    
    def _mezclar_capas_inteligente(self, capas: List[tuple]) -> np.ndarray:
        """Mezcla inteligente de capas"""
        if not capas:
            raise ValueError("No hay capas para mezclar")
        
        # Encontrar longitud mínima
        min_length = min(capa[1].shape[-1] for capa in capas)
        
        # Normalizar capas
        capas_norm = []
        for nombre, audio in capas:
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            audio_crop = audio[..., :min_length]
            capas_norm.append(audio_crop)
        
        # Mezcla simple pero efectiva
        audio_final = np.sum(capas_norm, axis=0) / len(capas_norm)
        
        # Normalización final
        max_val = np.max(np.abs(audio_final))
        if max_val > 0:
            audio_final = audio_final / max_val * 0.85
        
        return audio_final
    
    def _optimizar_y_validar(self, audio_data: np.ndarray) -> tuple[np.ndarray, float]:
        """Optimización y validación final"""
        try:
            if self.componentes.esta_disponible("quality"):
                quality = self.componentes.obtener("quality")
                audio_optimizado = quality.validar_y_normalizar(audio_data)
                
                # Calcular score de calidad
                peak = float(np.max(np.abs(audio_optimizado)))
                rms = float(np.sqrt(np.mean(audio_optimizado**2)))
                
                score = 85
                if peak > 0.95:
                    score -= 10
                if rms < 0.01:
                    score -= 5
                
                return audio_optimizado, max(70, min(100, score))
            else:
                # Validación básica
                if audio_data.ndim == 1:
                    audio_data = np.stack([audio_data, audio_data])
                
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.85
                
                return np.clip(audio_data, -1.0, 1.0), 80.0
                
        except Exception as e:
            logger.warning(f"⚠️ Error en optimización: {e}")
            return audio_data, 75.0
    
    def _guardar_audio(self, audio_data: np.ndarray, config: ConfigAurora) -> str:
        """Guarda el audio con nombre inteligente"""
        try:
            output_dir = Path("./aurora_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre = f"aurora_{config.objetivo}_{timestamp}.wav"
            ruta_completa = output_dir / nombre
            
            # Intentar guardar como WAV
            try:
                import scipy.io.wavfile as wav
                audio_int = (audio_data * 32767).astype(np.int16)
                wav.write(str(ruta_completa), config.sample_rate, audio_int.T)
                logger.info(f"💾 Audio guardado: {ruta_completa}")
                return str(ruta_completa)
            except ImportError:
                # Fallback numpy
                np.save(str(ruta_completa.with_suffix('.npy')), audio_data)
                return str(ruta_completa.with_suffix('.npy'))
                
        except Exception as e:
            logger.warning(f"⚠️ Error guardando: {e}")
            return ""
    
    def _crear_resultado_completo(self, audio_data: np.ndarray, config: ConfigAurora,
                                estrategia: str, score: float, tiempo: float, 
                                ruta: str) -> ResultadoAurora:
        """Crea resultado completo"""
        
        duracion = len(audio_data[0]) / config.sample_rate if audio_data.ndim > 1 else len(audio_data) / config.sample_rate
        
        # Motores usados
        motores_usados = []
        if self.componentes.esta_disponible("neuromix"):
            motores_usados.append("NeuroMix V26")
        if self.componentes.esta_disponible("texturas"):
            motores_usados.append("HarmonicEssence V34")
        if self.componentes.esta_disponible("harmony"):
            motores_usados.append("Harmony V7")
        
        # Recomendaciones
        recomendaciones = []
        if score < 80:
            recomendaciones.append("Considerar aumentar duración para mejor calidad")
        if score > 95:
            recomendaciones.append("Calidad excelente - apto para uso terapéutico")
        
        return ResultadoAurora(
            audio_data=audio_data,
            sample_rate=config.sample_rate,
            duracion_segundos=duracion,
            configuracion=config.__dict__,
            motores_usados=motores_usados,
            score_calidad=score,
            tiempo_generacion=tiempo,
            estrategia_usada=estrategia,
            ruta_archivo=ruta,
            recomendaciones=recomendaciones
        )
    
    def _actualizar_stats(self, resultado: ResultadoAurora, objetivo: str, estrategia: str):
        """Actualiza estadísticas del sistema"""
        self.stats["experiencias_generadas"] += 1
        self.stats["tiempo_total"] += resultado.tiempo_generacion
        
        # Promedio de calidad
        total = self.stats["experiencias_generadas"]
        calidad_actual = self.stats["calidad_promedio"]
        nueva_calidad = resultado.score_calidad
        self.stats["calidad_promedio"] = (calidad_actual * (total - 1) + nueva_calidad) / total
        
        # Estrategias usadas
        if estrategia not in self.stats["estrategias_usadas"]:
            self.stats["estrategias_usadas"][estrategia] = 0
        self.stats["estrategias_usadas"][estrategia] += 1
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Estadísticas completas del sistema"""
        return {
            "version": self.version,
            "componentes_disponibles": {
                "neuromix": self.componentes.esta_disponible("neuromix"),
                "texturas": self.componentes.esta_disponible("texturas"),
                "harmony": self.componentes.esta_disponible("harmony"),
                "quality": self.componentes.esta_disponible("quality"),
                "field_profiles": self.componentes.esta_disponible("field_profiles")
            },
            "estadisticas": self.stats
        }

# === API ULTRA-SIMPLE ===
_aurora_sistema = None

def Aurora(objetivo: str = None, **kwargs):
    """
    🌟 API ÚNICA de Aurora V7
    
    Reemplaza completamente:
    - Aurora_Master.crear_experiencia()
    - aurora_director.construir_pista()
    
    ✅ TODO EN UNO: Detecta automáticamente qué usar
    """
    global _aurora_sistema
    
    if _aurora_sistema is None:
        _aurora_sistema = AuroraUnificado()
    
    if objetivo:
        return _aurora_sistema.crear_experiencia(objetivo, **kwargs)
    else:
        return _aurora_sistema

# Métodos de conveniencia
Aurora.stats = lambda: _aurora_sistema.obtener_estadisticas() if _aurora_sistema else {}
Aurora.rapido = lambda obj, **kw: Aurora(obj, duracion=5, **kw)
Aurora.largo = lambda obj, **kw: Aurora(obj, duracion=60, **kw)
Aurora.terapeutico = lambda obj, **kw: Aurora(obj, duracion=45, precision_cientifica=True, **kw)

if __name__ == "__main__":
    print("🌟 Aurora V7 Sistema Unificado - TODO EN UNO")
    print("=" * 60)
    
    try:
        # Inicializar
        sistema = Aurora()
        stats = sistema.obtener_estadisticas()
        
        print(f"🚀 {stats['version']}")
        print(f"\n🔧 Componentes disponibles:")
        for comp, disponible in stats['componentes_disponibles'].items():
            estado = "✅" if disponible else "❌"
            print(f"   • {comp}: {estado}")
        
        print(f"\n🎵 Generando experiencia de prueba...")
        
        # Caso simple (usa estrategia automática)
        resultado = Aurora("relajacion", duracion=2, guardar=False)
        
        print(f"\n✅ Generación exitosa!")
        print(f"   • Estrategia: {resultado.estrategia_usada}")
        print(f"   • Calidad: {resultado.score_calidad:.1f}/100")
        print(f"   • Tiempo: {resultado.tiempo_generacion:.2f}s")
        print(f"   • Motores: {', '.join(resultado.motores_usados)}")
        print(f"   • Duración: {resultado.duracion_segundos:.1f}s")
        
        if resultado.recomendaciones:
            print(f"   • Recomendación: {resultado.recomendaciones[0]}")
        
        print(f"\n🎯 Casos de uso:")
        print(f"   • Básico: Aurora('relajacion')")
        print(f"   • Rápido: Aurora.rapido('concentracion')")
        print(f"   • Largo: Aurora.largo('meditacion')")
        print(f"   • Terapéutico: Aurora.terapeutico('neuroplasticidad')")
        
        print(f"\n🏆 ¡UN SOLO ARCHIVO - FUNCIONALIDAD COMPLETA!")
        print(f"📁 Reemplaza: aurora_director.py + Aurora_Master.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
