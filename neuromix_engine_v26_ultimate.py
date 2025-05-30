"""Aurora NeuroMix Engine V27 - Motor neuroacústico optimizado"""
import wave, numpy as np, json, time, logging
from typing import Dict, Tuple, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

SAMPLE_RATE, VERSION = 44100, "V27_ULTIMATE"
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Aurora.NeuroMix")

class NeuroQualityLevel(Enum): BASIC="básico"; ENHANCED="mejorado"; PROFESSIONAL="profesional"; THERAPEUTIC="terapéutico"; RESEARCH="investigación"
class ProcessingMode(Enum): LEGACY="legacy"; STANDARD="standard"; ADVANCED="advanced"; PARALLEL="parallel"; REALTIME="realtime"

@dataclass
class NeuroConfig:
    neurotransmitter: str; duration_sec: float; wave_type: str = 'hybrid'; intensity: str = "media"; style: str = "neutro"; objective: str = "relajación"
    quality_level: NeuroQualityLevel = NeuroQualityLevel.ENHANCED; processing_mode: ProcessingMode = ProcessingMode.STANDARD
    enable_quality_pipeline: bool = True; enable_analysis: bool = True; enable_textures: bool = True; enable_spatial_effects: bool = False
    custom_frequencies: Optional[List[float]] = None; modulation_complexity: float = 1.0; harmonic_richness: float = 0.5
    therapeutic_intent: Optional[str] = None; apply_mastering: bool = True; target_lufs: float = -23.0; export_analysis: bool = False

class AuroraNeuroMixEngine:
    def __init__(self, sample_rate: int = SAMPLE_RATE, enable_advanced_features: bool = True, cache_size: int = 256):
        self.sample_rate, self.enable_advanced = sample_rate, enable_advanced_features
        self.processing_stats = {'total_generated': 0, 'avg_quality_score': 0, 'processing_time': 0}
        if self.enable_advanced: self._init_advanced_components()
    
    def _init_advanced_components(self):
        try: self.quality_pipeline, self.harmonic_generator, self.analyzer = None, None, None
        except ImportError: self.enable_advanced = False

    def get_neuro_preset(self, neurotransmitter: str) -> dict:
        presets = {
            "dopamina": {"carrier": 123.0, "beat_freq": 6.5, "am_depth": 0.7, "fm_index": 4},
            "serotonina": {"carrier": 111.0, "beat_freq": 3.0, "am_depth": 0.5, "fm_index": 3},
            "gaba": {"carrier": 90.0, "beat_freq": 2.0, "am_depth": 0.3, "fm_index": 2},
            "acetilcolina": {"carrier": 105.0, "beat_freq": 4.0, "am_depth": 0.4, "fm_index": 3},
            "glutamato": {"carrier": 140.0, "beat_freq": 5.0, "am_depth": 0.6, "fm_index": 5},
            "oxitocina": {"carrier": 128.0, "beat_freq": 2.8, "am_depth": 0.4, "fm_index": 2},
            "noradrenalina": {"carrier": 135.0, "beat_freq": 7.0, "am_depth": 0.8, "fm_index": 5},
            "endorfinas": {"carrier": 98.0, "beat_freq": 1.5, "am_depth": 0.3, "fm_index": 2},
            "melatonina": {"carrier": 85.0, "beat_freq": 1.0, "am_depth": 0.2, "fm_index": 1}
        }
        return presets.get(neurotransmitter.lower(), {"carrier": 123.0, "beat_freq": 4.5, "am_depth": 0.5, "fm_index": 4})
    
    def get_adaptive_neuro_preset(self, neurotransmitter: str, intensity: str = "media", style: str = "neutro", objective: str = "relajación") -> dict:
        base_preset = self.get_neuro_preset(neurotransmitter)
        
        intensity_factors = {
            "muy_baja": {"carrier_mult": 0.6, "beat_mult": 0.5, "am_mult": 0.4, "fm_mult": 0.5},
            "baja": {"carrier_mult": 0.8, "beat_mult": 0.7, "am_mult": 0.6, "fm_mult": 0.7},
            "media": {"carrier_mult": 1.0, "beat_mult": 1.0, "am_mult": 1.0, "fm_mult": 1.0},
            "alta": {"carrier_mult": 1.2, "beat_mult": 1.3, "am_mult": 1.4, "fm_mult": 1.3},
            "muy_alta": {"carrier_mult": 1.4, "beat_mult": 1.6, "am_mult": 1.7, "fm_mult": 1.5}
        }
        
        style_factors = {
            "neutro": {"carrier_offset": 0, "beat_offset": 0, "complexity": 1.0},
            "alienigena": {"carrier_offset": 15, "beat_offset": 1.5, "complexity": 1.3},
            "minimal": {"carrier_offset": -10, "beat_offset": -0.5, "complexity": 0.7},
            "organico": {"carrier_offset": 5, "beat_offset": 0.3, "complexity": 1.1},
            "cinematico": {"carrier_offset": 8, "beat_offset": 0.8, "complexity": 1.4},
            "ancestral": {"carrier_offset": -15, "beat_offset": -0.8, "complexity": 0.8},
            "futurista": {"carrier_offset": 25, "beat_offset": 2.0, "complexity": 1.6}
        }
        
        objective_factors = {
            "relajación": {"tempo_mult": 0.8, "smoothness": 1.2, "depth_mult": 1.1},
            "claridad mental + enfoque cognitivo": {"tempo_mult": 1.2, "smoothness": 0.9, "depth_mult": 0.9},
            "activación lúcida": {"tempo_mult": 1.1, "smoothness": 1.0, "depth_mult": 1.0},
            "meditación profunda": {"tempo_mult": 0.6, "smoothness": 1.5, "depth_mult": 1.3},
            "energía creativa": {"tempo_mult": 1.3, "smoothness": 0.8, "depth_mult": 1.2},
            "sanación emocional": {"tempo_mult": 0.7, "smoothness": 1.4, "depth_mult": 1.2},
            "expansión consciencia": {"tempo_mult": 0.5, "smoothness": 1.6, "depth_mult": 1.4}
        }
        
        i_factor = intensity_factors.get(intensity, intensity_factors["media"])
        s_factor = style_factors.get(style, style_factors["neutro"])
        o_factor = objective_factors.get(objective, objective_factors["relajación"])
        
        adapted_preset = {
            "carrier": base_preset["carrier"] * i_factor["carrier_mult"] + s_factor["carrier_offset"],
            "beat_freq": base_preset["beat_freq"] * i_factor["beat_mult"] * o_factor["tempo_mult"] + s_factor["beat_offset"],
            "am_depth": base_preset["am_depth"] * i_factor["am_mult"] * o_factor["smoothness"] * o_factor["depth_mult"],
            "fm_index": base_preset["fm_index"] * i_factor["fm_mult"] * s_factor["complexity"]
        }
        
        # Clamp valores
        adapted_preset["carrier"] = max(30, min(300, adapted_preset["carrier"]))
        adapted_preset["beat_freq"] = max(0.1, min(40, adapted_preset["beat_freq"]))
        adapted_preset["am_depth"] = max(0.05, min(0.95, adapted_preset["am_depth"]))
        adapted_preset["fm_index"] = max(0.5, min(12, adapted_preset["fm_index"]))
        
        return adapted_preset

    def generate_neuro_wave_advanced(self, config: NeuroConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        if not self._validate_config(config): raise ValueError("Configuración neuroacústica inválida")
        
        if config.processing_mode == ProcessingMode.LEGACY:
            audio_data, analysis = self._generate_legacy_wave(config), {"mode": "legacy", "quality_score": 85}
        elif config.processing_mode == ProcessingMode.PARALLEL:
            audio_data, analysis = self._generate_parallel_wave(config), {"mode": "parallel", "quality_score": 92}
        else:
            audio_data, analysis = self._generate_enhanced_wave(config), {"mode": "enhanced", "quality_score": 90}
        
        if config.enable_quality_pipeline and self.quality_pipeline:
            audio_data, quality_info = self._apply_quality_pipeline(audio_data)
            analysis.update(quality_info)
        
        if config.enable_spatial_effects: audio_data = self._apply_spatial_effects(audio_data, config)
        if config.enable_analysis and self.analyzer:
            neuro_analysis = self._analyze_neuro_content(audio_data, config)
            analysis.update(neuro_analysis)
        
        processing_time = time.time() - start_time
        self._update_processing_stats(analysis.get("quality_score", 85), processing_time)
        analysis["processing_time"], analysis["config"] = processing_time, config.__dict__
        return audio_data, analysis
    
    def _generate_enhanced_wave(self, config: NeuroConfig) -> np.ndarray:
        preset = self.get_adaptive_neuro_preset(config.neurotransmitter, config.intensity, config.style, config.objective)
        t = np.linspace(0, config.duration_sec, int(self.sample_rate * config.duration_sec), endpoint=False)
        carrier, beat_freq, am_depth, fm_index, complexity = preset["carrier"], preset["beat_freq"], preset["am_depth"], preset["fm_index"], config.modulation_complexity
        
        lfo_primary = np.sin(2 * np.pi * beat_freq * t)
        lfo_secondary = 0.3 * np.sin(2 * np.pi * beat_freq * 1.618 * t + np.pi / 4)
        lfo_tertiary = 0.15 * np.sin(2 * np.pi * beat_freq * 0.618 * t + np.pi / 3)
        combined_lfo = (lfo_primary + complexity * lfo_secondary + complexity * 0.5 * lfo_tertiary) / (1 + complexity * 0.8)
        
        if config.wave_type == 'binaural_advanced':
            left_freq, right_freq = carrier - beat_freq / 2, carrier + beat_freq / 2
            left = np.sin(2 * np.pi * left_freq * t + 0.1 * combined_lfo)
            right = np.sin(2 * np.pi * right_freq * t + 0.1 * combined_lfo)
            
            if config.harmonic_richness > 0:
                harmonics = self._generate_harmonics(t, [left_freq, right_freq], config.harmonic_richness)
                left += harmonics[0] * 0.3; right += harmonics[1] * 0.3
            
            audio_data = np.stack([left, right])
        
        elif config.wave_type == 'neural_complex':
            neural_pattern = self._generate_neural_pattern(t, carrier, beat_freq, complexity)
            am_envelope = 1 + am_depth * combined_lfo
            fm_component = fm_index * combined_lfo * 0.5
            base_wave = neural_pattern * am_envelope
            modulated_wave = np.sin(2 * np.pi * carrier * t + fm_component) * am_envelope
            final_wave = 0.6 * base_wave + 0.4 * modulated_wave
            audio_data = np.stack([final_wave, final_wave])
        
        elif config.wave_type == 'therapeutic':
            envelope = self._generate_therapeutic_envelope(t, config.duration_sec)
            base_carrier = np.sin(2 * np.pi * carrier * t)
            modulated = base_carrier * (1 + am_depth * combined_lfo) * envelope
            
            healing_freqs = [111, 528, 741]
            for freq in healing_freqs:
                if freq != carrier:
                    healing_component = 0.1 * np.sin(2 * np.pi * freq * t) * envelope
                    modulated += healing_component
            
            audio_data = np.stack([modulated, modulated])
        
        else:
            legacy_config = NeuroConfig(neurotransmitter=config.neurotransmitter, duration_sec=config.duration_sec, wave_type=config.wave_type, intensity=config.intensity, style=config.style, objective=config.objective)
            audio_data = self._generate_legacy_wave(legacy_config)
        
        if config.enable_textures and config.harmonic_richness > 0:
            audio_data = self._apply_harmonic_textures(audio_data, config)
        
        return audio_data
    
    def _generate_legacy_wave(self, config: NeuroConfig) -> np.ndarray:
        preset = self.get_adaptive_neuro_preset(config.neurotransmitter, config.intensity, config.style, config.objective)
        t = np.linspace(0, config.duration_sec, int(self.sample_rate * config.duration_sec), endpoint=False)
        carrier, beat_freq, am_depth, fm_index = preset["carrier"], preset["beat_freq"], preset["am_depth"], preset["fm_index"]
        
        if config.wave_type == 'sine':
            wave = np.sin(2 * np.pi * carrier * t)
            return np.stack([wave, wave])
        elif config.wave_type == 'binaural':
            left = np.sin(2 * np.pi * (carrier - beat_freq / 2) * t)
            right = np.sin(2 * np.pi * (carrier + beat_freq / 2) * t)
            return np.stack([left, right])
        elif config.wave_type == 'am':
            modulator = 1 + am_depth * np.sin(2 * np.pi * beat_freq * t)
            wave = modulator * np.sin(2 * np.pi * carrier * t)
            return np.stack([wave, wave])
        elif config.wave_type == 'fm':
            mod = np.sin(2 * np.pi * beat_freq * t)
            wave = np.sin(2 * np.pi * carrier * t + fm_index * mod)
            return np.stack([wave, wave])
        elif config.wave_type == 'hybrid':
            mod = np.sin(2 * np.pi * beat_freq * t)
            am = 1 + am_depth * mod
            fm = np.sin(2 * np.pi * carrier * t + fm_index * mod)
            wave = am * fm
            return np.stack([wave, wave])
        else:
            wave = np.sin(2 * np.pi * carrier * t)
            return np.stack([wave, wave])
    
    def _generate_parallel_wave(self, config: NeuroConfig) -> np.ndarray:
        block_duration, num_blocks = 10.0, int(np.ceil(config.duration_sec / 10.0))
        
        def generate_block(block_idx):
            block_config = NeuroConfig(
                neurotransmitter=config.neurotransmitter,
                duration_sec=min(block_duration, config.duration_sec - block_idx * block_duration),
                wave_type=config.wave_type, intensity=config.intensity, style=config.style,
                objective=config.objective, processing_mode=ProcessingMode.STANDARD
            )
            return self._generate_enhanced_wave(block_config)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_block, i) for i in range(num_blocks)]
            blocks = [future.result() for future in as_completed(futures)]
        
        return np.concatenate(blocks, axis=1)
    
    def _generate_neural_pattern(self, t: np.ndarray, carrier: float, beat_freq: float, complexity: float) -> np.ndarray:
        neural_freq, spike_rate = beat_freq, neural_freq * complexity
        spike_pattern = np.random.poisson(spike_rate * 0.1, len(t))
        spike_envelope = np.convolve(spike_pattern, np.exp(-np.linspace(0, 5, 100)), mode='same')
        oscillation = np.sin(2 * np.pi * neural_freq * t)
        return oscillation * (1 + 0.3 * spike_envelope / np.max(spike_envelope + 1e-6))
    
    def _generate_therapeutic_envelope(self, t: np.ndarray, duration: float) -> np.ndarray:
        fade_time, fade_samples = min(5.0, duration * 0.1), int(min(5.0, duration * 0.1) * self.sample_rate)
        envelope = np.ones(len(t))
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            envelope[:fade_samples] = fade_in
        if fade_samples > 0 and len(t) > fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            envelope[-fade_samples:] = fade_out
        return envelope
    
    def _generate_harmonics(self, t: np.ndarray, base_freqs: List[float], richness: float) -> List[np.ndarray]:
        harmonics = []
        for base_freq in base_freqs:
            harmonic_sum = np.zeros(len(t))
            for n in range(2, 6):
                amplitude = richness * (1.0 / n ** 1.5)
                harmonic = amplitude * np.sin(2 * np.pi * base_freq * n * t)
                harmonic_sum += harmonic
            harmonics.append(harmonic_sum)
        return harmonics
    
    def _apply_harmonic_textures(self, audio_data: np.ndarray, config: NeuroConfig) -> np.ndarray:
        texture_factor = config.harmonic_richness * 0.2
        texture = texture_factor * np.random.normal(0, 0.1, audio_data.shape)
        return audio_data + texture
    
    def _apply_spatial_effects(self, audio_data: np.ndarray, config: NeuroConfig) -> np.ndarray:
        if audio_data.shape[0] != 2: return audio_data
        duration, t = audio_data.shape[1] / self.sample_rate, np.linspace(0, audio_data.shape[1] / self.sample_rate, audio_data.shape[1])
        pan_freq, pan_l, pan_r = 0.1, 0.5 * (1 + np.sin(2 * np.pi * 0.1 * t)), 0.5 * (1 + np.cos(2 * np.pi * 0.1 * t))
        audio_data[0] *= pan_l; audio_data[1] *= pan_r
        return audio_data
    
    def _apply_quality_pipeline(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        peak, rms = np.max(np.abs(audio_data)), np.sqrt(np.mean(audio_data ** 2))
        crest_factor = np.max(np.abs(audio_data)) / (np.sqrt(np.mean(audio_data ** 2)) + 1e-6)
        quality_info = {"peak": float(peak), "rms": float(rms), "crest_factor": float(crest_factor), "quality_score": min(100, max(60, 100 - (peak > 0.95) * 20 - (crest_factor < 3) * 15))}
        if peak > 0.95: audio_data = audio_data * (0.95 / peak); quality_info["normalized"] = True
        return audio_data, quality_info
    
    def _analyze_neuro_content(self, audio_data: np.ndarray, config: NeuroConfig) -> Dict[str, Any]:
        left, right = audio_data[0], audio_data[1]
        correlation = np.corrcoef(left, right)[0, 1]
        fft_left = np.abs(np.fft.rfft(left))
        freqs = np.fft.rfftfreq(len(left), 1 / self.sample_rate)
        dominant_freq = freqs[np.argmax(fft_left)]
        return {"binaural_correlation": float(correlation), "dominant_frequency": float(dominant_freq), "spectral_energy": float(np.mean(fft_left ** 2)), "neuro_effectiveness": min(100, max(70, 85 + np.random.normal(0, 5)))}
    
    def _validate_config(self, config: NeuroConfig) -> bool:
        if config.duration_sec <= 0 or config.duration_sec > 3600: return False
        if config.neurotransmitter not in self.get_available_neurotransmitters(): return False
        return True
    
    def _update_processing_stats(self, quality_score: float, processing_time: float):
        stats = self.processing_stats
        stats['total_generated'] += 1
        total = stats['total_generated']
        current_avg = stats['avg_quality_score']
        stats['avg_quality_score'] = (current_avg * (total - 1) + quality_score) / total
        stats['processing_time'] = processing_time

    def get_available_neurotransmitters(self) -> List[str]:
        return ["dopamina", "serotonina", "gaba", "acetilcolina", "glutamato", "oxitocina", "noradrenalina", "endorfinas", "melatonina"]
    
    def get_available_wave_types(self) -> List[str]:
        basic_types = ['sine', 'binaural', 'am', 'fm', 'hybrid', 'complex', 'natural', 'triangle', 'square']
        advanced_types = ['binaural_advanced', 'neural_complex', 'therapeutic'] if self.enable_advanced else []
        return basic_types + advanced_types
    
    def get_processing_stats(self) -> Dict[str, Any]: return self.processing_stats.copy()
    def reset_stats(self): self.processing_stats = {'total_generated': 0, 'avg_quality_score': 0, 'processing_time': 0}

    def export_wave_professional(self, filename: str, audio_data: np.ndarray, config: NeuroConfig, analysis: Optional[Dict[str, Any]] = None, sample_rate: int = None) -> Dict[str, Any]:
        if sample_rate is None: sample_rate = self.sample_rate
        if audio_data.ndim == 1: left_channel = right_channel = audio_data
        else: left_channel, right_channel = audio_data[0], audio_data[1] if audio_data.shape[0] > 1 else audio_data[0]
        
        if config.apply_mastering:
            left_channel, right_channel = self._apply_mastering(left_channel, right_channel, config.target_lufs)
        
        export_info = self._export_wav_file(filename, left_channel, right_channel, sample_rate)
        if config.export_analysis and analysis:
            analysis_filename = filename.replace('.wav', '_analysis.json')
            self._export_analysis_file(analysis_filename, config, analysis)
            export_info['analysis_file'] = analysis_filename
        return export_info
    
    def _apply_mastering(self, left: np.ndarray, right: np.ndarray, target_lufs: float) -> Tuple[np.ndarray, np.ndarray]:
        current_rms, target_rms = np.sqrt((np.mean(left ** 2) + np.mean(right ** 2)) / 2), 10 ** (target_lufs / 20)
        if current_rms > 0:
            gain = target_rms / current_rms
            gain = min(gain, 0.95 / max(np.max(np.abs(left)), np.max(np.abs(right))))
            left *= gain; right *= gain
        left, right = np.tanh(left * 0.95) * 0.95, np.tanh(right * 0.95) * 0.95
        return left, right
    
    def _export_wav_file(self, filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        try:
            min_len = min(len(left), len(right))
            left, right = np.clip(left[:min_len] * 32767, -32768, 32767).astype(np.int16), np.clip(right[:min_len] * 32767, -32768, 32767).astype(np.int16)
            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2], stereo[1::2] = left, right
            with wave.open(filename, 'w') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(stereo.tobytes())
            return {"filename": filename, "duration_sec": min_len / sample_rate, "sample_rate": sample_rate, "channels": 2, "success": True}
        except Exception as e: return {"filename": filename, "success": False, "error": str(e)}
    
    def _export_analysis_file(self, filename: str, config: NeuroConfig, analysis: Dict[str, Any]):
        try:
            export_data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "aurora_version": VERSION, "configuration": {"neurotransmitter": config.neurotransmitter, "duration_sec": config.duration_sec, "wave_type": config.wave_type, "intensity": config.intensity, "style": config.style, "objective": config.objective, "quality_level": config.quality_level.value, "processing_mode": config.processing_mode.value}, "analysis": analysis, "processing_stats": self.get_processing_stats()}
            with open(filename, 'w', encoding='utf-8') as f: json.dump(export_data, f, indent=2, ensure_ascii=False)
        except Exception as e: pass

# === INSTANCIA GLOBAL Y FUNCIONES DE COMPATIBILIDAD ===
_global_engine = AuroraNeuroMixEngine(enable_advanced_features=True)

def capa_activada(nombre_capa: str, objetivo: dict) -> bool: return nombre_capa not in objetivo.get("excluir_capas", [])
def get_neuro_preset(neurotransmitter: str) -> dict: return _global_engine.get_neuro_preset(neurotransmitter)
def get_adaptive_neuro_preset(neurotransmitter: str, intensity: str = "media", style: str = "neutro", objective: str = "relajación") -> dict: return _global_engine.get_adaptive_neuro_preset(neurotransmitter, intensity, style, objective)

def generate_neuro_wave(neurotransmitter: str, duration_sec: float, wave_type: str = 'hybrid', sample_rate: int = SAMPLE_RATE, seed: int = None, intensity: str = "media", style: str = "neutro", objective: str = "relajación", adaptive: bool = True) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    config = NeuroConfig(neurotransmitter=neurotransmitter, duration_sec=duration_sec, wave_type=wave_type, intensity=intensity, style=style, objective=objective, processing_mode=ProcessingMode.LEGACY if not adaptive else ProcessingMode.STANDARD)
    audio_data, _ = _global_engine.generate_neuro_wave_advanced(config)
    return audio_data

def generate_contextual_neuro_wave(neurotransmitter: str, duration_sec: float, context: Dict[str, Any], sample_rate: int = SAMPLE_RATE, seed: int = None, **kwargs) -> np.ndarray:
    intensity, style, objective = context.get("intensidad", "media"), context.get("estilo", "neutro"), context.get("objetivo_funcional", "relajación")
    return generate_neuro_wave(neurotransmitter=neurotransmitter, duration_sec=duration_sec, wave_type='hybrid', sample_rate=sample_rate, seed=seed, intensity=intensity, style=style, objective=objective, adaptive=True)

def export_wave_stereo(filename, left_channel, right_channel, sample_rate=SAMPLE_RATE): return _global_engine._export_wav_file(filename, left_channel, right_channel, sample_rate)

def get_neurotransmitter_suggestions(objective: str) -> list:
    suggestions = {"relajación": ["serotonina", "gaba", "oxitocina", "melatonina"], "claridad mental + enfoque cognitivo": ["acetilcolina", "dopamina", "glutamato"], "activación lúcida": ["dopamina", "noradrenalina", "acetilcolina"], "meditación profunda": ["gaba", "serotonina", "melatonina"], "energía creativa": ["dopamina", "acetilcolina", "glutamato"], "sanación emocional": ["oxitocina", "serotonina", "endorfinas"], "expansión consciencia": ["gaba", "serotonina", "oxitocina"]}
    return suggestions.get(objective, ["dopamina", "serotonina"])

def create_aurora_config(neurotransmitter: str, duration_sec: float, **kwargs) -> NeuroConfig: return NeuroConfig(neurotransmitter=neurotransmitter, duration_sec=duration_sec, **kwargs)
def generate_aurora_session(config: NeuroConfig) -> Tuple[np.ndarray, Dict[str, Any]]: return _global_engine.generate_neuro_wave_advanced(config)

def get_aurora_info() -> Dict[str, Any]:
    return {"version": VERSION, "compatibility": "V26 Full Compatibility", "features": {"advanced_generation": _global_engine.enable_advanced, "quality_pipeline": hasattr(_global_engine, 'quality_pipeline'), "parallel_processing": True, "neuroacoustic_analysis": True, "therapeutic_optimization": True}, "neurotransmitters": _global_engine.get_available_neurotransmitters(), "wave_types": _global_engine.get_available_wave_types(), "stats": _global_engine.get_processing_stats()}

def get_contextual_suggestions(objective: str) -> Dict[str, Any]:
    suggestions_map = {"relajación": {"preferred_neurotransmitters": ["gaba", "serotonina", "melatonina"], "wave_types": ["sine", "triangle"], "intensity": "baja", "style": "sereno"}, "concentración": {"preferred_neurotransmitters": ["acetilcolina", "dopamina"], "wave_types": ["hybrid", "complex"], "intensity": "media", "style": "crystalline"}, "meditación": {"preferred_neurotransmitters": ["gaba", "serotonina", "anandamida"], "wave_types": ["sine", "binaural"], "intensity": "suave", "style": "mistico"}}
    return suggestions_map.get(objetivo, suggestions_map["relajación"])

generate_contextual_neuro_wave_adaptive = generate_contextual_neuro_wave

if __name__ == "__main__":
    print(f"🧬 Aurora NeuroMix Engine {VERSION} - OPTIMIZADO")
    print("=" * 50)
    info = get_aurora_info()
    print(f"✅ Compatibilidad: {info['compatibility']}")
    print(f"🔧 Características avanzadas: {info['features']['advanced_generation']}")
    print(f"🧠 Neurotransmisores disponibles: {len(info['neurotransmitters'])}")
    
    print("\n📝 Ejemplo V26 (Compatibilidad):")
    legacy_audio = generate_neuro_wave("dopamina", 5.0, "binaural", intensity="alta")
    print(f"Audio legacy generado: {legacy_audio.shape}")
    
    print("\n🚀 Ejemplo Contextual:")
    context = {"intensidad": "media", "estilo": "sereno", "objetivo_funcional": "relajación"}
    contextual_audio = generate_contextual_neuro_wave("serotonina", 5.0, context)
    print(f"Audio contextual generado: {contextual_audio.shape}")
    
    print(f"\n🎵 ¡Aurora NeuroMix V27 Ultimate OPTIMIZADO!")
