"""Aurora Quality Pipeline - Sistema de validación y corrección de audio optimizado"""
import numpy as np, warnings, time
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from functools import lru_cache

@dataclass
class ResultadoFase:
    audio: np.ndarray; metadata: Dict[str, Any]; dynamic_range: Optional[float] = None
    spectral_balance: Optional[Dict[str, float]] = None; phase_coherence: Optional[float] = None
    processing_time: Optional[float] = None

@dataclass
class AnalysisConfig:
    enable_spectral_analysis: bool = True; enable_dynamic_analysis: bool = True; enable_phase_analysis: bool = True
    quality_threshold: float = 85.0; fft_size: int = 4096; overlap: float = 0.5

class ModoProcesamiento:
    PROFESIONAL = "PROFESIONAL"; TEST = "TEST"; DEBUG = "DEBUG"; MASTERING = "MASTERING"
    REALTIME = "REALTIME"; BATCH = "BATCH"

class CarmineAnalyzer:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig(); self._analysis_cache = {}
        
    @staticmethod
    def evaluar(audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
        return CarmineAnalyzer().evaluar_avanzado(audio, sample_rate)
    
    def evaluar_avanzado(self, audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self._validate_audio(audio): return self._get_error_result("Audio inválido")
            
            resultado = {"score": 0, "nivel": "desconocido", "problemas": [], "recomendaciones": [], "metricas": {}}
            
            dynamic_score, dynamic_issues = self._analyze_dynamics(audio)
            resultado["metricas"]["dynamic_range"] = dynamic_score
            
            if self.config.enable_spectral_analysis:
                spectral_score, spectral_issues = self._analyze_spectrum(audio, sample_rate)
                resultado["metricas"]["spectral_balance"] = spectral_score; resultado["problemas"].extend(spectral_issues)
            
            if audio.ndim > 1 and self.config.enable_phase_analysis:
                phase_score, phase_issues = self._analyze_phase_coherence(audio)
                resultado["metricas"]["phase_coherence"] = phase_score; resultado["problemas"].extend(phase_issues)
            
            distortion_score, distortion_issues = self._analyze_distortion(audio)
            resultado["metricas"]["thd"] = distortion_score; resultado["problemas"].extend(distortion_issues)
            
            resultado["score"] = self._calculate_overall_score(resultado["metricas"])
            resultado["nivel"] = self._get_quality_level(resultado["score"])
            resultado["recomendaciones"] = self._generate_recommendations(resultado)
            resultado["processing_time"] = time.time() - start_time
            
            return resultado
        except Exception as e:
            warnings.warn(f"Error en análisis: {e}")
            return self._get_error_result(str(e))
    
    def _validate_audio(self, audio: np.ndarray) -> bool:
        return isinstance(audio, np.ndarray) and audio.size > 0 and not np.all(audio == 0)
    
    def _analyze_dynamics(self, audio: np.ndarray) -> Tuple[float, List[str]]:
        issues = []; mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        rms, peak = np.sqrt(np.mean(mono**2)), np.max(np.abs(mono))
        p99, p1 = np.percentile(np.abs(mono), 99), np.percentile(np.abs(mono), 1)
        dynamic_range = 20 * np.log10(p99 / (p1 + 1e-10))
        
        if dynamic_range < 6: issues.append("Rango dinámico muy limitado (over-compression)")
        elif dynamic_range > 60: issues.append("Rango dinámico excesivo (posible ruido)")
        
        crest_factor = peak / (rms + 1e-10)
        if crest_factor < 1.5: issues.append("Factor cresta bajo (posible limitación)")
        
        return min(100, max(0, (dynamic_range - 6) * 3)), issues
    
    def _analyze_spectrum(self, audio: np.ndarray, sample_rate: int) -> Tuple[Dict[str, float], List[str]]:
        issues = []; mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        fft = np.fft.rfft(mono, n=self.config.fft_size); magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(self.config.fft_size, 1/sample_rate)
        
        bands = {'sub': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500), 'mid': (500, 2000), 'high_mid': (2000, 4000), 'presence': (4000, 8000), 'air': (8000, sample_rate//2)}
        
        band_energies = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask): energy = np.mean(magnitude[mask]**2); band_energies[band_name] = 10 * np.log10(energy + 1e-10)
            else: band_energies[band_name] = -60
        
        total_energy = np.mean(list(band_energies.values()))
        for band, energy in band_energies.items():
            deviation = energy - total_energy
            if deviation > 6: issues.append(f"Exceso de energía en {band} (+{deviation:.1f}dB)")
            elif deviation < -12: issues.append(f"Déficit de energía en {band} ({deviation:.1f}dB)")
        
        return band_energies, issues
    
    def _analyze_phase_coherence(self, audio: np.ndarray) -> Tuple[float, List[str]]:
        if audio.ndim != 2: return 100, []
        issues = []; left, right = audio[0], audio[1]
        correlation = np.corrcoef(left, right)[0, 1]
        mid, side = (left + right) / 2, (left - right) / 2
        mid_energy, side_energy = np.mean(mid**2), np.mean(side**2)
        ms_ratio = 10 * np.log10(mid_energy / side_energy) if side_energy > 0 else 60
        
        if correlation < 0.3: issues.append("Baja correlación estéreo (posibles problemas de fase)")
        elif correlation > 0.95: issues.append("Correlación muy alta (imagen estéreo limitada)")
        
        if ms_ratio < -6: issues.append("Exceso de información lateral")
        elif ms_ratio > 20: issues.append("Imagen estéreo muy estrecha")
        
        return correlation * 100, issues
    
    def _analyze_distortion(self, audio: np.ndarray) -> Tuple[float, List[str]]:
        issues = []; mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        clipping_ratio = np.sum(np.abs(mono) > 0.99) / len(mono)
        if clipping_ratio > 0.001: issues.append(f"Clipping detectado ({clipping_ratio*100:.2f}%)")
        
        kurt = kurtosis(mono)
        if kurt > 10: issues.append("Alta curtosis (posible distorsión)")
        
        thd_estimate = min(100, clipping_ratio * 1000 + max(0, kurt - 3) * 2)
        return max(0, 100 - thd_estimate), issues
    
    def _calculate_overall_score(self, metricas: Dict[str, Any]) -> float:
        weights = {'dynamic_range': 0.3, 'phase_coherence': 0.2, 'thd': 0.3, 'spectral_balance': 0.2}
        total_score = total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metricas:
                value = metricas[metric]
                if isinstance(value, dict): values = list(value.values()); score = max(0, 100 - np.var(values) * 2)
                else: score = float(value)
                total_score += score * weight; total_weight += weight
        
        return total_score / max(total_weight, 0.1)
    
    def _get_quality_level(self, score: float) -> str:
        if score >= 90: return "excelente"
        elif score >= 80: return "muy bueno"
        elif score >= 70: return "bueno"
        elif score >= 60: return "aceptable"
        else: return "deficiente"
    
    def _generate_recommendations(self, resultado: Dict[str, Any]) -> List[str]:
        recomendaciones = []; score = resultado["score"]; problemas = resultado["problemas"]
        
        if score < 70: recomendaciones.append("Considerar reprocesamiento con diferentes parámetros")
        if any("compression" in p.lower() for p in problemas): recomendaciones.append("Reducir compresión para mejorar dinámica")
        if any("clipping" in p.lower() for p in problemas): recomendaciones.append("Reducir niveles para evitar saturación")
        if any("fase" in p.lower() or "phase" in p.lower() for p in problemas): recomendaciones.append("Revisar procesamiento estéreo")
        
        return recomendaciones
    
    def _get_error_result(self, error_msg: str) -> Dict[str, Any]:
        return {"score": 0, "nivel": "error", "problemas": [f"Error de análisis: {error_msg}"], "recomendaciones": ["Verificar formato y contenido del audio"], "metricas": {}}

class CorrectionStrategy:
    def __init__(self, mode: str = ModoProcesamiento.PROFESIONAL):
        self.mode = mode; self._correction_cache = {}
    
    @staticmethod
    def aplicar(audio: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        return CorrectionStrategy().aplicar_avanzado(audio, metadata)
    
    def aplicar_avanzado(self, audio: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        if not isinstance(audio, np.ndarray): return audio
        
        corrected = audio.copy(); problemas = metadata.get("problemas", []); score = metadata.get("score", 100)
        
        if score >= 85: return corrected
        
        try:
            if any("clipping" in p.lower() for p in problemas): corrected = self._soft_clip_correction(corrected)
            if any("compression" in p.lower() for p in problemas): corrected = self._dynamic_expansion(corrected)
            if any("energía" in p for p in problemas): corrected = self._spectral_correction(corrected)
            return corrected
        except Exception as e: warnings.warn(f"Error en corrección: {e}"); return audio
    
    def _soft_clip_correction(self, audio: np.ndarray) -> np.ndarray:
        threshold = 0.95; mask = np.abs(audio) > threshold
        audio[mask] = np.tanh(audio[mask] * 0.8) * threshold
        return audio
    
    def _dynamic_expansion(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2, axis=-1, keepdims=True))
        expansion_factor = 1.0 + 0.1 * (1 - rms / 0.5)
        return audio * expansion_factor
    
    def _spectral_correction(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            for channel in range(audio.shape[0]): audio[channel] = self._apply_gentle_eq(audio[channel])
        else: audio = self._apply_gentle_eq(audio)
        return audio
    
    def _apply_gentle_eq(self, signal: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        try: b, a = butter(1, 20/(22050), btype='high'); return filtfilt(b, a, signal)
        except: return signal

class AuroraQualityPipeline:
    def __init__(self, mode: str = ModoProcesamiento.PROFESIONAL):
        self.mode = mode; self.analyzer = CarmineAnalyzer(); self.estrategia = CorrectionStrategy(mode)
        self._processing_stats = {'total_processed': 0, 'avg_score': 0, 'processing_time': 0}
    
    def procesar(self, audio_func: Callable) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        try:
            audio = audio_func(); audio = self.validar_y_normalizar(audio)
            
            if self.mode == ModoProcesamiento.DEBUG: resultado = self.analyzer.evaluar_avanzado(audio)
            else: resultado = CarmineAnalyzer.evaluar(audio)
            
            if resultado["score"] < 85:
                if self.mode == ModoProcesamiento.PROFESIONAL: audio = self.estrategia.aplicar_avanzado(audio, resultado)
                else: audio = CorrectionStrategy.aplicar(audio, resultado)
            
            self._update_stats(resultado, time.time() - start_time)
            return audio, resultado
        except Exception as e:
            warnings.warn(f"Error en pipeline: {e}"); audio = audio_func()
            return audio, {"score": 50, "nivel": "error", "problemas": [str(e)]}
    
    def validar_y_normalizar(self, signal: np.ndarray) -> np.ndarray:
        if not isinstance(signal, np.ndarray): raise ValueError("❌ Señal inválida: debe ser un arreglo numpy")
        
        if signal.ndim == 1: signal = np.stack([signal, signal])
        elif signal.ndim != 2: raise ValueError("❌ Señal inválida: debe ser mono [N] o estéreo [2, N]")
        
        max_val = np.max(np.abs(signal))
        if max_val > 0: target_level = 0.95; signal = signal * (target_level / max_val)
        
        return np.clip(signal, -1.0, 1.0)
    
    def _update_stats(self, resultado: Dict[str, Any], processing_time: float):
        self._processing_stats['total_processed'] += 1
        current_avg = self._processing_stats['avg_score']; total = self._processing_stats['total_processed']
        new_score = resultado.get('score', 0)
        self._processing_stats['avg_score'] = (current_avg * (total-1) + new_score) / total
        self._processing_stats['processing_time'] = processing_time
    
    def get_stats(self) -> Dict[str, Any]: return self._processing_stats.copy()
    def reset_stats(self): self._processing_stats = {'total_processed': 0, 'avg_score': 0, 'processing_time': 0}

# === FUNCIONES DE CONVENIENCIA ===
def evaluar_calidad_audio(audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
    """Función simplificada para evaluar calidad de audio"""
    return CarmineAnalyzer.evaluar(audio, sample_rate)

def corregir_audio_automaticamente(audio: np.ndarray, modo: str = ModoProcesamiento.PROFESIONAL) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Función simplificada para corrección automática"""
    pipeline = AuroraQualityPipeline(mode=modo)
    return pipeline.procesar(lambda: audio)

def validar_audio_estereo(audio: np.ndarray) -> np.ndarray:
    """Validación básica de audio estéreo"""
    pipeline = AuroraQualityPipeline()
    return pipeline.validar_y_normalizar(audio)

if __name__ == "__main__":
    print("🔍 Aurora Quality Pipeline - Sistema de Validación Optimizado")
    print("=" * 60)
    
    # Generar audio de prueba
    sample_rate = 44100; duration = 2.0; t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Audio de prueba con algunos problemas
    left_channel = 0.9 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 0.1, len(t))
    right_channel = 0.8 * np.sin(2 * np.pi * 444 * t) + 0.1 * np.random.normal(0, 0.1, len(t))
    test_audio = np.stack([left_channel, right_channel])
    
    print(f"📊 Analizando audio de prueba...")
    resultado = evaluar_calidad_audio(test_audio, sample_rate)
    
    print(f"✅ Análisis completado:")
    print(f"   • Score de calidad: {resultado['score']:.1f}/100")
    print(f"   • Nivel: {resultado['nivel']}")
    print(f"   • Problemas detectados: {len(resultado['problemas'])}")
    
    if resultado['problemas']:
        print(f"   • Problemas: {', '.join(resultado['problemas'][:2])}")
    
    if resultado['recomendaciones']:
        print(f"   • Recomendaciones: {len(resultado['recomendaciones'])}")
    
    print(f"\n🔧 Probando corrección automática...")
    try:
        audio_corregido, analisis_corregido = corregir_audio_automaticamente(test_audio)
        print(f"✅ Corrección aplicada - Score mejorado: {analisis_corregido['score']:.1f}/100")
    except Exception as e:
        print(f"⚠️ Error en corrección: {e}")
    
    print(f"\n🚀 Aurora Quality Pipeline listo para validar audio!")
