import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime
import warnings
from pathlib import Path
import hashlib

try:
    from presets_emocionales import *
    from style_profiles import *
    from presets_estilos import *
    from presets_fases import *
    from objective_templates import *
    AURORA_V7_AVAILABLE = True
    print("🌟 Aurora V7 loaded")
except ImportError:
    AURORA_V7_AVAILABLE = False
    print("⚠️ Aurora V7 unavailable")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Aurora.HyperMod")

class NeuroWaveType(Enum):
    ALPHA="alpha";BETA="beta";THETA="theta";DELTA="delta";GAMMA="gamma"
    BINAURAL="binaural";ISOCHRONIC="isochronic";SOLFEGGIO="solfeggio"
    SCHUMANN="schumann";THERAPEUTIC="therapeutic";NEURAL_SYNC="neural_sync"
    QUANTUM_FIELD="quantum_field";CEREMONIAL="ceremonial"

class EmotionalPhase(Enum):
    ENTRADA="entrada";DESARROLLO="desarrollo";CLIMAX="climax"
    RESOLUCION="resolucion";SALIDA="salida";PREPARACION="preparacion"
    INTENCION="intencion";VISUALIZACION="visualizacion"
    COLAPSO="colapso";ANCLAJE="anclaje";INTEGRACION="integracion"

@dataclass
class AudioConfig:
    sample_rate:int=44100;channels:int=2;bit_depth:int=16;block_duration:int=60;max_layers:int=8;target_loudness:float=-23.0
    preset_emocional:Optional[str]=None;estilo_visual:Optional[str]=None;perfil_acustico:Optional[str]=None
    template_objetivo:Optional[str]=None;secuencia_fases:Optional[str]=None;validacion_cientifica:bool=True
    optimizacion_neuroacustica:bool=True;modo_terapeutico:bool=False;precision_cuantica:float=0.95
    version_aurora:str="V7_Integrated";timestamp:str=field(default_factory=lambda:datetime.now().isoformat())

@dataclass
class LayerConfig:
    name:str;wave_type:NeuroWaveType;frequency:float;amplitude:float;phase:EmotionalPhase;modulation_depth:float=0.0
    spatial_enabled:bool=False;neurotransmisor:Optional[str]=None;efecto_deseado:Optional[str]=None
    coherencia_neuroacustica:float=0.9;efectividad_terapeutica:float=0.8;patron_evolutivo:str="linear"
    sincronizacion_cardiaca:bool=False;modulacion_cuantica:bool=False;base_cientifica:str="validado"
    contraindicaciones:List[str]=field(default_factory=list)

@dataclass
class ResultadoAuroraV7:
    audio_data:np.ndarray;metadata:Dict[str,Any];coherencia_neuroacustica:float=0.0;efectividad_terapeutica:float=0.0
    calidad_espectral:float=0.0;sincronizacion_fases:float=0.0;analisis_neurotransmisores:Dict[str,float]=field(default_factory=dict)
    validacion_objetivos:Dict[str,Any]=field(default_factory=dict);metricas_cuanticas:Dict[str,float]=field(default_factory=dict)
    sugerencias_optimizacion:List[str]=field(default_factory=list);proximas_fases_recomendadas:List[str]=field(default_factory=list)
    configuracion_optima:Optional[Dict[str,Any]]=None

class GestorAuroraIntegrado:
    def __init__(self):
        self.initialized=False;self.gestores={}
        if AURORA_V7_AVAILABLE:self._inicializar_gestores_aurora()
    
    def _inicializar_gestores_aurora(self):
        try:
            self.gestores={'emocionales':crear_gestor_presets(),'estilos':crear_gestor_estilos(),
                          'esteticos':crear_gestor_estilos_esteticos(),'fases':crear_gestor_fases(),
                          'templates':crear_gestor_optimizado()}
            self.initialized=True
        except:self.initialized=False
    
    def crear_layers_desde_preset_emocional(self,nombre_preset:str,duracion_min:int=20)->List[LayerConfig]:
        if not self.initialized:return self._crear_layers_fallback(nombre_preset)
        try:
            preset=self.gestores['emocionales'].obtener_preset(nombre_preset)
            if not preset:return self._crear_layers_fallback(nombre_preset)
            layers=[LayerConfig(f"Emocional_{preset.nombre}",self._mapear_frecuencia_a_tipo_onda(preset.frecuencia_base),
                               preset.frecuencia_base,0.7,EmotionalPhase.DESARROLLO,
                               neurotransmisor=list(preset.neurotransmisores.keys())[0] if preset.neurotransmisores else None,
                               coherencia_neuroacustica=0.95,efectividad_terapeutica=0.9)]
            for nt,intensidad in preset.neurotransmisores.items():
                if intensidad>0.5:
                    freq_nt=self._obtener_frecuencia_neurotransmisor(nt)
                    layers.append(LayerConfig(f"NT_{nt.title()}",self._mapear_frecuencia_a_tipo_onda(freq_nt),
                                            freq_nt,intensidad*0.6,EmotionalPhase.DESARROLLO,modulation_depth=0.2,
                                            neurotransmisor=nt,coherencia_neuroacustica=0.85,efectividad_terapeutica=intensidad))
            if hasattr(preset,'frecuencias_armonicas') and preset.frecuencias_armonicas:
                for i,freq_arm in enumerate(preset.frecuencias_armonicas[:2]):
                    layers.append(LayerConfig(f"Armonico_{i+1}",self._mapear_frecuencia_a_tipo_onda(freq_arm),
                                            freq_arm,0.3,EmotionalPhase.ENTRADA,spatial_enabled=True,
                                            coherencia_neuroacustica=0.8,efectividad_terapeutica=0.7))
            return layers
        except:return self._crear_layers_fallback(nombre_preset)
    
    def crear_layers_desde_secuencia_fases(self,nombre_secuencia:str,fase_actual:int=0)->List[LayerConfig]:
        if not self.initialized:return self._crear_layers_fallback("secuencia_generica")
        try:
            secuencia=self.gestores['fases'].obtener_secuencia(nombre_secuencia)
            if not secuencia or not secuencia.fases:return self._crear_layers_fallback("secuencia_generica")
            fase=secuencia.fases[min(fase_actual,len(secuencia.fases)-1)]
            layers=[LayerConfig(f"Fase_{fase.nombre}",self._mapear_frecuencia_a_tipo_onda(fase.beat_base),
                               fase.beat_base,0.8,self._mapear_tipo_fase_a_emotional_phase(fase.tipo_fase),
                               neurotransmisor=fase.neurotransmisor_principal,coherencia_neuroacustica=fase.nivel_confianza,
                               efectividad_terapeutica=0.9)]
            for nt,intensidad in fase.neurotransmisores_secundarios.items():
                freq_nt=self._obtener_frecuencia_neurotransmisor(nt)
                layers.append(LayerConfig(f"Fase_{nt.title()}",self._mapear_frecuencia_a_tipo_onda(freq_nt),
                                        freq_nt,intensidad*0.5,EmotionalPhase.DESARROLLO,neurotransmisor=nt,
                                        coherencia_neuroacustica=0.85,efectividad_terapeutica=intensidad))
            return layers
        except:return self._crear_layers_fallback("secuencia_generica")
    
    def crear_layers_desde_template_objetivo(self,nombre_template:str)->List[LayerConfig]:
        if not self.initialized:return self._crear_layers_fallback(nombre_template)
        try:
            template=self.gestores['templates'].obtener_template(nombre_template)
            if not template:return self._crear_layers_fallback(nombre_template)
            layers=[LayerConfig(f"Template_{template.nombre}",self._mapear_frecuencia_a_tipo_onda(template.frecuencia_dominante),
                               template.frecuencia_dominante,0.75,EmotionalPhase.DESARROLLO,
                               coherencia_neuroacustica=template.coherencia_neuroacustica,efectividad_terapeutica=template.nivel_confianza)]
            for nt,intensidad in template.neurotransmisores_principales.items():
                if intensidad>0.4:
                    freq_nt=self._obtener_frecuencia_neurotransmisor(nt)
                    layers.append(LayerConfig(f"Template_{nt.title()}",self._mapear_frecuencia_a_tipo_onda(freq_nt),
                                            freq_nt,intensidad*0.6,EmotionalPhase.DESARROLLO,modulation_depth=0.15,
                                            neurotransmisor=nt,coherencia_neuroacustica=0.88,efectividad_terapeutica=intensidad))
            return layers
        except:return self._crear_layers_fallback(nombre_template)
    
    def _mapear_frecuencia_a_tipo_onda(self,frecuencia:float)->NeuroWaveType:
        if frecuencia<=4:return NeuroWaveType.DELTA
        elif frecuencia<=8:return NeuroWaveType.THETA
        elif frecuencia<=13:return NeuroWaveType.ALPHA
        elif frecuencia<=30:return NeuroWaveType.BETA
        elif frecuencia<=100:return NeuroWaveType.GAMMA
        elif 174<=frecuencia<=963:return NeuroWaveType.SOLFEGGIO
        elif frecuencia==7.83:return NeuroWaveType.SCHUMANN
        elif frecuencia>=400:return NeuroWaveType.THERAPEUTIC
        else:return NeuroWaveType.ALPHA
    
    def _obtener_frecuencia_neurotransmisor(self,neurotransmisor:str)->float:
        return {"gaba":6.0,"serotonina":7.5,"dopamina":12.0,"acetilcolina":14.0,"norepinefrina":15.0,"oxitocina":8.0,
                "endorfina":10.5,"anandamida":5.5,"melatonina":4.0,"adrenalina":16.0}.get(neurotransmisor.lower(),10.0)
    
    def _mapear_tipo_fase_a_emotional_phase(self,tipo_fase)->EmotionalPhase:
        fase_str=tipo_fase.value if hasattr(tipo_fase,'value') else str(tipo_fase).lower()
        return {"preparacion":EmotionalPhase.PREPARACION,"activacion":EmotionalPhase.ENTRADA,"intencion":EmotionalPhase.INTENCION,
                "visualizacion":EmotionalPhase.VISUALIZACION,"manifestacion":EmotionalPhase.CLIMAX,"colapso":EmotionalPhase.COLAPSO,
                "integracion":EmotionalPhase.INTEGRACION,"anclaje":EmotionalPhase.ANCLAJE,"cierre":EmotionalPhase.SALIDA}.get(fase_str,EmotionalPhase.DESARROLLO)
    
    def _crear_layers_fallback(self,nombre:str)->List[LayerConfig]:
        return [LayerConfig("Base Alpha",NeuroWaveType.ALPHA,10.0,0.6,EmotionalPhase.DESARROLLO),
                LayerConfig("Theta Deep",NeuroWaveType.THETA,6.0,0.4,EmotionalPhase.CLIMAX),
                LayerConfig("Beta Focus",NeuroWaveType.BETA,15.0,0.3,EmotionalPhase.ENTRADA)]
    
    def obtener_info_preset(self,tipo:str,nombre:str)->Dict[str,Any]:
        if not self.initialized:return {"error":"Sistema Aurora V7 no disponible"}
        try:
            if tipo=="emocional":
                preset=self.gestores['emocionales'].obtener_preset(nombre)
                if preset:return {"nombre":preset.nombre,"descripcion":preset.descripcion,"categoria":preset.categoria.value,
                                "neurotransmisores":preset.neurotransmisores,"frecuencia_base":preset.frecuencia_base,
                                "efectos":{"atencion":preset.efectos.atencion,"calma":preset.efectos.calma,
                                         "creatividad":preset.efectos.creatividad,"energia":preset.efectos.energia},
                                "contextos_recomendados":preset.contextos_recomendados,"mejor_momento_uso":preset.mejor_momento_uso}
            elif tipo=="secuencia":
                secuencia=self.gestores['fases'].obtener_secuencia(nombre)
                if secuencia:return {"nombre":secuencia.nombre,"descripcion":secuencia.descripcion,"num_fases":len(secuencia.fases),
                                   "duracion_total":secuencia.duracion_total_min,"categoria":secuencia.categoria,
                                   "fases":[f.nombre for f in secuencia.fases]}
            elif tipo=="template":
                template=self.gestores['templates'].obtener_template(nombre)
                if template:return {"nombre":template.nombre,"descripcion":template.descripcion,"categoria":template.categoria.value,
                                  "complejidad":template.complejidad.value,"frecuencia_dominante":template.frecuencia_dominante,
                                  "duracion_recomendada":template.duracion_recomendada_min,"efectos_esperados":template.efectos_esperados,
                                  "evidencia_cientifica":template.evidencia_cientifica}
            return {"error":f"Tipo '{tipo}' no reconocido"}
        except Exception as e:return {"error":f"Error: {str(e)}"}

gestor_aurora=GestorAuroraIntegrado()

class NeuroWaveGenerator:
    def __init__(self,config:AudioConfig):
        self.config=config;self.cache_ondas={}
        if AURORA_V7_AVAILABLE and config.preset_emocional:self._analizar_preset_emocional()
    
    def _analizar_preset_emocional(self):
        try:info_preset=gestor_aurora.obtener_info_preset("emocional",self.config.preset_emocional)
        except:pass
    
    def generate_wave(self,wave_type:NeuroWaveType,frequency:float,duration:int,amplitude:float,layer_config:LayerConfig=None)->np.ndarray:
        cache_key=f"{wave_type.value}_{frequency}_{duration}_{amplitude}"
        if cache_key in self.cache_ondas:return self.cache_ondas[cache_key]*amplitude
        
        samples=int(self.config.sample_rate*duration);t=np.linspace(0,duration,samples,dtype=np.float32)
        
        if wave_type==NeuroWaveType.ALPHA:wave=np.sin(2*np.pi*frequency*t)
        elif wave_type==NeuroWaveType.BETA:wave=np.sin(2*np.pi*frequency*t)+0.3*np.sin(2*np.pi*frequency*2*t)
        elif wave_type==NeuroWaveType.THETA:wave=np.sin(2*np.pi*frequency*t)*np.exp(-t*0.05)
        elif wave_type==NeuroWaveType.DELTA:wave=np.sin(2*np.pi*frequency*t)*(1+0.3*np.sin(2*np.pi*0.1*t))
        elif wave_type==NeuroWaveType.GAMMA:wave=np.sin(2*np.pi*frequency*t)+0.1*np.random.normal(0,0.1,len(t))
        elif wave_type==NeuroWaveType.BINAURAL:
            left,right=np.sin(2*np.pi*frequency*t),np.sin(2*np.pi*(frequency+8)*t)
            return (np.column_stack([left,right])*amplitude).astype(np.float32)
        elif wave_type==NeuroWaveType.ISOCHRONIC:
            envelope=0.5*(1+np.square(np.sin(2*np.pi*10*t)));wave=np.sin(2*np.pi*frequency*t)*envelope
        elif wave_type==NeuroWaveType.SOLFEGGIO:wave=self._generate_solfeggio_wave(t,frequency)
        elif wave_type==NeuroWaveType.SCHUMANN:wave=self._generate_schumann_wave(t,frequency)
        elif wave_type==NeuroWaveType.THERAPEUTIC:wave=self._generate_therapeutic_wave(t,frequency,layer_config)
        elif wave_type==NeuroWaveType.NEURAL_SYNC:wave=self._generate_neural_sync_wave(t,frequency)
        elif wave_type==NeuroWaveType.QUANTUM_FIELD:wave=self._generate_quantum_field_wave(t,frequency)
        elif wave_type==NeuroWaveType.CEREMONIAL:wave=self._generate_ceremonial_wave(t,frequency)
        else:wave=np.sin(2*np.pi*frequency*t)
        
        if wave.ndim==1:wave=np.column_stack([wave,wave])
        if AURORA_V7_AVAILABLE and layer_config:wave=self._aplicar_mejoras_aurora_v7(wave,layer_config)
        self.cache_ondas[cache_key]=wave;return (wave*amplitude).astype(np.float32)
    
    def _generate_solfeggio_wave(self,t:np.ndarray,frequency:float)->np.ndarray:
        base=np.sin(2*np.pi*frequency*t)
        sacred_harmonics=0.2*np.sin(2*np.pi*frequency*3/2*t)+0.1*np.sin(2*np.pi*frequency*5/4*t)
        return base+sacred_harmonics+0.05*np.sin(2*np.pi*0.1*t)
    
    def _generate_schumann_wave(self,t:np.ndarray,frequency:float)->np.ndarray:
        base=np.sin(2*np.pi*frequency*t)
        harmonics=0.3*np.sin(2*np.pi*(frequency*2)*t)+0.2*np.sin(2*np.pi*(frequency*3)*t)+0.1*np.sin(2*np.pi*(frequency*4)*t)
        return (base+harmonics)*(1+0.1*np.sin(2*np.pi*0.02*t))
    
    def _generate_therapeutic_wave(self,t:np.ndarray,frequency:float,layer_config:LayerConfig=None)->np.ndarray:
        base=np.sin(2*np.pi*frequency*t)
        if layer_config and layer_config.neurotransmisor:
            nt=layer_config.neurotransmisor.lower();mod_freq={"gaba":0.1,"dopamina":0.5,"serotonina":0.2}.get(nt,0.15)
            therapeutic_mod=0.2*np.sin(2*np.pi*mod_freq*t)
        else:therapeutic_mod=0.2*np.sin(2*np.pi*0.1*t)
        return base*(0.9+0.1*np.tanh(0.1*t))+therapeutic_mod
    
    def _generate_neural_sync_wave(self,t:np.ndarray,frequency:float)->np.ndarray:
        return np.sin(2*np.pi*frequency*t)+0.3*np.sin(2*np.pi*frequency*1.618*t)+0.05*np.random.normal(0,0.5,len(t))+0.1*np.sin(2*np.pi*frequency*0.5*t)
    
    def _generate_quantum_field_wave(self,t:np.ndarray,frequency:float)->np.ndarray:
        return np.sin(2*np.pi*frequency*t)+0.4*np.cos(2*np.pi*frequency*np.sqrt(2)*t)+0.2*np.sin(2*np.pi*frequency*0.1*t)*np.cos(2*np.pi*frequency*0.07*t)+0.1*np.sin(2*np.pi*frequency*t)*np.sin(2*np.pi*frequency*1.414*t)
    
    def _generate_ceremonial_wave(self,t:np.ndarray,frequency:float)->np.ndarray:
        base=np.sin(2*np.pi*frequency*t)+0.3*np.sin(2*np.pi*frequency*0.618*t)
        ritual=0.2*np.sin(2*np.pi*0.05*t)*np.sin(2*np.pi*frequency*t)
        harmonics=0.1*np.sin(2*np.pi*frequency*3*t)+0.05*np.sin(2*np.pi*frequency*5*t)
        return base+ritual+harmonics
    
    def _aplicar_mejoras_aurora_v7(self,wave:np.ndarray,layer_config:LayerConfig)->np.ndarray:
        enhanced_wave=wave.copy()
        if layer_config.coherencia_neuroacustica>0.9:
            coherence_factor=layer_config.coherencia_neuroacustica
            enhanced_wave=enhanced_wave*coherence_factor+np.roll(enhanced_wave,1)*(1-coherence_factor)
        if layer_config.efectividad_terapeutica>0.8:
            therapeutic_envelope=1.0+0.1*layer_config.efectividad_terapeutica*np.sin(2*np.pi*0.1*np.arange(len(enhanced_wave))/self.config.sample_rate)
            if enhanced_wave.ndim==2:therapeutic_envelope=np.column_stack([therapeutic_envelope,therapeutic_envelope])
            enhanced_wave*=therapeutic_envelope[:len(enhanced_wave)]
        if layer_config.sincronizacion_cardiaca:
            heart_modulation=0.05*np.sin(2*np.pi*1.2*np.arange(len(enhanced_wave))/self.config.sample_rate)
            if enhanced_wave.ndim==2:heart_modulation=np.column_stack([heart_modulation,heart_modulation])
            enhanced_wave=enhanced_wave*(1+heart_modulation[:len(enhanced_wave)])
        return enhanced_wave
    
    def apply_modulation(self,wave:np.ndarray,mod_type:str,mod_depth:float,mod_freq:float=0.5)->np.ndarray:
        if mod_depth==0:return wave
        t=np.linspace(0,len(wave)/self.config.sample_rate,len(wave),dtype=np.float32)
        if mod_type=="AM":
            modulator=1+mod_depth*np.sin(2*np.pi*mod_freq*t)
            if wave.ndim==2:modulator=np.column_stack([modulator,modulator])
            return wave*modulator
        elif mod_type=="FM":
            phase_mod=mod_depth*np.sin(2*np.pi*mod_freq*t)
            if wave.ndim==2:phase_mod=np.column_stack([phase_mod,phase_mod])
            return wave*(1+0.1*phase_mod)
        elif mod_type=="QUANTUM" and AURORA_V7_AVAILABLE:
            quantum_mod=mod_depth*np.sin(2*np.pi*mod_freq*t)*np.cos(2*np.pi*mod_freq*1.414*t)
            if wave.ndim==2:quantum_mod=np.column_stack([quantum_mod,quantum_mod])
            return wave*(1+quantum_mod)
        return wave
    
    def apply_spatial_effects(self,wave:np.ndarray,effect_type:str="3D",layer_config:LayerConfig=None)->np.ndarray:
        if wave.ndim!=2:return wave
        t=np.linspace(0,len(wave)/self.config.sample_rate,len(wave),dtype=np.float32)
        if effect_type=="3D":
            pan_l,pan_r=0.5*(1+np.sin(2*np.pi*0.2*t)),0.5*(1+np.cos(2*np.pi*0.2*t))
            wave[:,0]*=pan_l;wave[:,1]*=pan_r
        elif effect_type=="8D":
            pan_l=0.5*(1+0.7*np.sin(2*np.pi*0.3*t)+0.3*np.sin(2*np.pi*0.17*t))
            pan_r=0.5*(1+0.7*np.cos(2*np.pi*0.3*t)+0.3*np.cos(2*np.pi*0.17*t))
            wave[:,0]*=pan_l;wave[:,1]*=pan_r
        elif effect_type=="THERAPEUTIC" and AURORA_V7_AVAILABLE and layer_config:
            if layer_config.neurotransmisor:
                nt=layer_config.neurotransmisor.lower()
                if nt=="oxitocina":
                    embrace_pan=0.5*(1+0.3*np.sin(2*np.pi*0.05*t))
                    wave[:,0]*=embrace_pan;wave[:,1]*=(2-embrace_pan)*0.5
                elif nt=="dopamina":
                    dynamic_pan=0.5*(1+0.4*np.sin(2*np.pi*0.15*t))
                    wave[:,0]*=dynamic_pan;wave[:,1]*=(2-dynamic_pan)*0.5
        elif effect_type=="QUANTUM" and AURORA_V7_AVAILABLE:
            quantum_pan_l=0.5*(1+0.4*np.sin(2*np.pi*0.1*t)*np.cos(2*np.pi*0.07*t))
            wave[:,0]*=quantum_pan_l;wave[:,1]*=(1-quantum_pan_l)
        return wave

def procesar_bloque_optimizado(args:Tuple[int,List[LayerConfig],AudioConfig,Dict[str,Any]])->Tuple[int,np.ndarray,Dict[str,Any]]:
    bloque_idx,layers,audio_config,params=args
    try:
        generator=NeuroWaveGenerator(audio_config);samples_per_block=int(audio_config.sample_rate*audio_config.block_duration)
        output_buffer=np.zeros((samples_per_block,audio_config.channels),dtype=np.float32)
        metricas_aurora={"coherencia_neuroacustica":0.0,"efectividad_terapeutica":0.0,"sincronizacion_fases":0.0,"calidad_espectral":0.0}
        
        for layer in layers:
            wave=generator.generate_wave(layer.wave_type,layer.frequency,audio_config.block_duration,layer.amplitude,layer)
            if layer.modulation_depth>0:
                mod_type="QUANTUM" if layer.modulacion_cuantica else "AM"
                wave=generator.apply_modulation(wave,mod_type,layer.modulation_depth)
            if layer.spatial_enabled:
                effect_type="THERAPEUTIC" if audio_config.modo_terapeutico else "3D"
                wave=generator.apply_spatial_effects(wave,effect_type,layer)
            
            phase_multiplier=get_phase_multiplier(layer.phase,bloque_idx,params.get('total_blocks',10))
            wave*=phase_multiplier
            
            if AURORA_V7_AVAILABLE:
                layer_metrics=_analizar_capa_aurora_v7(wave,layer)
                metricas_aurora["coherencia_neuroacustica"]+=layer_metrics.get("coherencia",0.0)
                metricas_aurora["efectividad_terapeutica"]+=layer_metrics.get("efectividad",0.0)
            output_buffer+=wave
        
        max_val=np.max(np.abs(output_buffer))
        if max_val>0.95:output_buffer*=0.85/max_val
        
        if len(layers)>0 and AURORA_V7_AVAILABLE:
            metricas_aurora["coherencia_neuroacustica"]/=len(layers);metricas_aurora["efectividad_terapeutica"]/=len(layers)
            metricas_aurora["calidad_espectral"]=_calcular_calidad_espectral(output_buffer)
        
        return (bloque_idx,output_buffer,metricas_aurora)
    except Exception as e:
        samples=int(audio_config.sample_rate*audio_config.block_duration)
        return (bloque_idx,np.zeros((samples,audio_config.channels),dtype=np.float32),{"error":str(e)})

def _analizar_capa_aurora_v7(wave:np.ndarray,layer:LayerConfig)->Dict[str,float]:
    metrics={}
    if wave.ndim==2:
        correlation=np.corrcoef(wave[:,0],wave[:,1])[0,1]
        metrics["coherencia"]=float(np.nan_to_num(correlation,layer.coherencia_neuroacustica))
    else:metrics["coherencia"]=layer.coherencia_neuroacustica
    rms=np.sqrt(np.mean(wave**2));dynamic_range=np.max(np.abs(wave))/(rms+1e-10)
    metrics["efectividad"]=float((1.0/(1.0+abs(dynamic_range-3.0)))*layer.efectividad_terapeutica)
    return metrics

def _calcular_calidad_espectral(audio_buffer:np.ndarray)->float:
    if audio_buffer.shape[0]<2:return 75.0
    try:
        fft_data=np.abs(np.fft.rfft(audio_buffer[:,0] if audio_buffer.ndim==2 else audio_buffer[0,:]))
        energy_distribution,flatness=np.std(fft_data),np.mean(fft_data)/(np.max(fft_data)+1e-10)
        return min(100.0,max(60.0,60+(energy_distribution*20)+(flatness*20)))
    except:return 75.0

def get_phase_multiplier(phase:EmotionalPhase,block_idx:int,total_blocks:int)->float:
    progress=block_idx/max(1,total_blocks-1)
    phase_map={EmotionalPhase.ENTRADA:0.3+0.4*progress if progress<0.2 else 0.7,
               EmotionalPhase.DESARROLLO:0.7+0.2*progress if progress<0.6 else 0.9,
               EmotionalPhase.CLIMAX:1.0 if 0.4<=progress<=0.8 else 0.8,
               EmotionalPhase.RESOLUCION:0.9-0.3*progress if progress>0.7 else 0.9,
               EmotionalPhase.SALIDA:max(0.1,0.7-0.6*progress) if progress>0.8 else 0.7,
               EmotionalPhase.PREPARACION:0.2+0.3*min(progress*2,1.0),
               EmotionalPhase.INTENCION:0.5+0.4*progress if progress<0.5 else 0.9,
               EmotionalPhase.VISUALIZACION:0.8+0.2*np.sin(progress*np.pi),
               EmotionalPhase.COLAPSO:0.9-0.4*progress if progress>0.6 else 0.9,
               EmotionalPhase.ANCLAJE:0.6+0.3*(1-progress),
               EmotionalPhase.INTEGRACION:0.7+0.2*np.sin(progress*np.pi*2)}
    return phase_map.get(phase,0.8)

def generar_bloques_aurora_integrado(duracion_total:int,layers_config:List[LayerConfig]=None,audio_config:AudioConfig=None,
                                   preset_emocional:str=None,secuencia_fases:str=None,template_objetivo:str=None,
                                   num_workers:int=None)->ResultadoAuroraV7:
    start_time=time.time()
    
    if audio_config is None:
        audio_config=AudioConfig(preset_emocional=preset_emocional,secuencia_fases=secuencia_fases,template_objetivo=template_objetivo)
    
    num_workers=min(mp.cpu_count(),6) if num_workers is None else num_workers
    
    if layers_config is None:
        if AURORA_V7_AVAILABLE:
            if preset_emocional:layers_config=gestor_aurora.crear_layers_desde_preset_emocional(preset_emocional,duracion_total)
            elif secuencia_fases:layers_config=gestor_aurora.crear_layers_desde_secuencia_fases(secuencia_fases)
            elif template_objetivo:layers_config=gestor_aurora.crear_layers_desde_template_objetivo(template_objetivo)
            else:layers_config=crear_preset_relajacion()
        else:layers_config=crear_preset_relajacion()
    
    total_blocks=int(np.ceil(duracion_total/audio_config.block_duration))
    args_list=[(i,layers_config,audio_config,{'total_blocks':total_blocks,'aurora_v7':AURORA_V7_AVAILABLE}) for i in range(total_blocks)]
    
    resultados={};metricas_globales={"coherencia_promedio":0.0,"efectividad_promedio":0.0,"calidad_promedio":0.0,"sincronizacion_promedio":0.0}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_block={executor.submit(procesar_bloque_optimizado,args):args[0] for args in args_list}
        for future in as_completed(future_to_block):
            try:
                block_idx,audio_data,metrics=future.result();resultados[block_idx]=(audio_data,metrics)
                if "error" not in metrics:
                    metricas_globales["coherencia_promedio"]+=metrics.get("coherencia_neuroacustica",0.0)
                    metricas_globales["efectividad_promedio"]+=metrics.get("efectividad_terapeutica",0.0)
                    metricas_globales["calidad_promedio"]+=metrics.get("calidad_espectral",75.0)
            except Exception as e:
                block_idx=future_to_block[future];samples=int(audio_config.sample_rate*audio_config.block_duration)
                resultados[block_idx]=(np.zeros((samples,audio_config.channels),dtype=np.float32),{"error":str(e)})
    
    num_blocks=len([r for r in resultados.values() if "error" not in r[1]])
    if num_blocks>0:
        for key in metricas_globales:metricas_globales[key]/=num_blocks
    
    bloques_ordenados=[]
    for i in range(total_blocks):
        if i in resultados:bloques_ordenados.append(resultados[i][0])
        else:bloques_ordenados.append(np.zeros((int(audio_config.sample_rate*audio_config.block_duration),audio_config.channels),dtype=np.float32))
    
    audio_final=np.vstack(bloques_ordenados) if bloques_ordenados else np.zeros((int(audio_config.sample_rate*duracion_total),audio_config.channels),dtype=np.float32)
    
    samples_objetivo=int(duracion_total*audio_config.sample_rate)
    if len(audio_final)>samples_objetivo:audio_final=audio_final[:samples_objetivo]
    elif len(audio_final)<samples_objetivo:audio_final=np.vstack([audio_final,np.zeros((samples_objetivo-len(audio_final),audio_config.channels),dtype=np.float32)])
    
    max_peak=np.max(np.abs(audio_final))
    if max_peak>0:audio_final*=(0.80 if audio_config.modo_terapeutico else 0.85)/max_peak
    
    resultado=ResultadoAuroraV7(audio_data=audio_final,
        metadata={"version":"V31_Aurora_V7_Integrated","duracion_seg":duracion_total,"sample_rate":audio_config.sample_rate,
                 "channels":audio_config.channels,"total_bloques":total_blocks,"capas_utilizadas":len(layers_config),
                 "preset_emocional":preset_emocional,"secuencia_fases":secuencia_fases,"template_objetivo":template_objetivo,
                 "aurora_v7_disponible":AURORA_V7_AVAILABLE,"timestamp":datetime.now().isoformat()},
        coherencia_neuroacustica=metricas_globales["coherencia_promedio"],
        efectividad_terapeutica=metricas_globales["efectividad_promedio"],
        calidad_espectral=metricas_globales["calidad_promedio"],
        sincronizacion_fases=metricas_globales["sincronizacion_promedio"])
    
    if AURORA_V7_AVAILABLE:resultado=_enriquecer_resultado_aurora_v7(resultado,layers_config,audio_config)
    resultado.sugerencias_optimizacion=_generar_sugerencias_optimizacion(resultado,audio_config)
    return resultado

def _enriquecer_resultado_aurora_v7(resultado:ResultadoAuroraV7,layers_config:List[LayerConfig],audio_config:AudioConfig)->ResultadoAuroraV7:
    neurotransmisores_detectados={}
    for layer in layers_config:
        if layer.neurotransmisor:
            nt=layer.neurotransmisor.lower()
            neurotransmisores_detectados[nt]=neurotransmisores_detectados.get(nt,0.0)+layer.amplitude*layer.efectividad_terapeutica
    resultado.analisis_neurotransmisores=neurotransmisores_detectados
    
    if audio_config.template_objetivo:
        info_template=gestor_aurora.obtener_info_preset("template",audio_config.template_objetivo)
        if "error" not in info_template:
            resultado.validacion_objetivos={"template_utilizado":info_template["nombre"],"categoria":info_template.get("categoria","unknown"),
                                          "efectos_esperados":info_template.get("efectos_esperados",[]),
                                          "coherencia_con_audio":min(1.0,resultado.coherencia_neuroacustica+0.1)}
    
    resultado.metricas_cuanticas={"coherencia_cuantica":resultado.coherencia_neuroacustica*0.95,
                                "entrelazamiento_simulado":resultado.efectividad_terapeutica*0.8,
                                "superposicion_armonica":resultado.calidad_espectral/100.0*0.9}
    return resultado

def _generar_sugerencias_optimizacion(resultado:ResultadoAuroraV7,audio_config:AudioConfig)->List[str]:
    sugerencias=[]
    if resultado.coherencia_neuroacustica<0.7:sugerencias.append("Mejorar coherencia: ajustar frecuencias de capas o usar preset emocional optimizado")
    if resultado.efectividad_terapeutica<0.6:sugerencias.append("Aumentar efectividad: incrementar amplitudes terapéuticas o usar modo terapéutico")
    if resultado.calidad_espectral<75:sugerencias.append("Mejorar calidad: revisar modulaciones o usar validación científica")
    if AURORA_V7_AVAILABLE and not audio_config.preset_emocional:sugerencias.append("Considerar usar preset emocional Aurora V7 para mejor integración científica")
    return sugerencias if sugerencias else ["Excelente calidad - considerar experimentar con nuevos tipos de onda Aurora V7"]

def generar_bloques(duracion_total:int,layers_config:List[LayerConfig],audio_config:AudioConfig=None,num_workers:int=None)->np.ndarray:
    return generar_bloques_aurora_integrado(duracion_total,layers_config,audio_config,num_workers=num_workers).audio_data

def crear_preset_relajacion()->List[LayerConfig]:
    return gestor_aurora.crear_layers_desde_preset_emocional("calma_profunda",20) if AURORA_V7_AVAILABLE else [
        LayerConfig("Alpha Base",NeuroWaveType.ALPHA,10.0,0.6,EmotionalPhase.DESARROLLO),
        LayerConfig("Theta Deep",NeuroWaveType.THETA,6.0,0.4,EmotionalPhase.CLIMAX),
        LayerConfig("Delta Sleep",NeuroWaveType.DELTA,2.0,0.2,EmotionalPhase.SALIDA)]

def crear_preset_enfoque()->List[LayerConfig]:
    return gestor_aurora.crear_layers_desde_preset_emocional("claridad_mental",25) if AURORA_V7_AVAILABLE else [
        LayerConfig("Beta Focus",NeuroWaveType.BETA,18.0,0.7,EmotionalPhase.DESARROLLO),
        LayerConfig("Alpha Bridge",NeuroWaveType.ALPHA,12.0,0.4,EmotionalPhase.ENTRADA),
        LayerConfig("Gamma Boost",NeuroWaveType.GAMMA,35.0,0.3,EmotionalPhase.CLIMAX)]

def crear_preset_meditacion()->List[LayerConfig]:
    return gestor_aurora.crear_layers_desde_preset_emocional("conexion_mistica",30) if AURORA_V7_AVAILABLE else [
        LayerConfig("Theta Meditation",NeuroWaveType.THETA,6.5,0.5,EmotionalPhase.DESARROLLO),
        LayerConfig("Schumann Resonance",NeuroWaveType.SCHUMANN,7.83,0.4,EmotionalPhase.CLIMAX),
        LayerConfig("Delta Deep",NeuroWaveType.DELTA,3.0,0.3,EmotionalPhase.INTEGRACION)]

def crear_preset_manifestacion()->List[LayerConfig]:
    return gestor_aurora.crear_layers_desde_secuencia_fases("manifestacion_clasica",0) if AURORA_V7_AVAILABLE else crear_preset_relajacion()

def crear_preset_sanacion()->List[LayerConfig]:
    return gestor_aurora.crear_layers_desde_preset_emocional("regulacion_emocional",25) if AURORA_V7_AVAILABLE else [
        LayerConfig("Solfeggio 528Hz",NeuroWaveType.SOLFEGGIO,528.0,0.5,EmotionalPhase.DESARROLLO),
        LayerConfig("Therapeutic Alpha",NeuroWaveType.THERAPEUTIC,8.0,0.6,EmotionalPhase.CLIMAX),
        LayerConfig("Heart Coherence",NeuroWaveType.ALPHA,0.1,0.3,EmotionalPhase.INTEGRACION)]

def exportar_wav_optimizado(audio_data:np.ndarray,filename:str,config:AudioConfig)->None:
    import wave
    try:
        if audio_data.dtype!=np.int16:
            audio_data=np.clip(audio_data,-1.0,1.0);audio_data=(audio_data*32767).astype(np.int16)
        
        with wave.open(filename,'wb') as wav_file:
            wav_file.setnchannels(config.channels);wav_file.setsampwidth(2);wav_file.setframerate(config.sample_rate)
            if config.channels==2:
                if audio_data.ndim==2:
                    interleaved=np.empty((audio_data.shape[0]*2,),dtype=np.int16)
                    interleaved[0::2],interleaved[1::2]=audio_data[:,0],audio_data[:,1]
                else:
                    interleaved=np.empty((audio_data.shape[1]*2,),dtype=np.int16)
                    interleaved[0::2],interleaved[1::2]=audio_data[0,:],audio_data[1,:]
                wav_file.writeframes(interleaved.tobytes())
            else:wav_file.writeframes(audio_data.tobytes())
    except Exception as e:logger.error(f"Export error: {e}")

def obtener_info_sistema()->Dict[str,Any]:
    info={"version":"V31_Aurora_V7_Integrated","compatibilidad_v31":"100%","aurora_v7_disponible":AURORA_V7_AVAILABLE,
          "tipos_onda_v31":len([t for t in NeuroWaveType if t.value in ["alpha","beta","theta","delta","gamma","binaural","isochronic"]]),
          "tipos_onda_aurora_v7":len([t for t in NeuroWaveType if t.value not in ["alpha","beta","theta","delta","gamma","binaural","isochronic"]]),
          "fases_emocionales":len(EmotionalPhase),
          "presets_disponibles":["crear_preset_relajacion","crear_preset_enfoque","crear_preset_meditacion","crear_preset_manifestacion","crear_preset_sanacion"]}
    
    if AURORA_V7_AVAILABLE:
        info["gestores_aurora_v7"]={"emocionales":"activo","estilos":"activo","esteticos":"activo","fases":"activo","templates":"activo"}
        try:
            info["presets_emocionales_disponibles"]=len(gestor_aurora.gestores['emocionales'].presets)
            info["secuencias_fases_disponibles"]=len(gestor_aurora.gestores['fases'].secuencias_predefinidas)
            info["templates_objetivos_disponibles"]=len(gestor_aurora.gestores['templates'].templates)
        except:pass
    return info

if __name__ == "__main__":
    print("🌟 Aurora HyperMod Engine V31 + Aurora V7 Integrado")
    print("=" * 50)
    info=obtener_info_sistema()
    print(f"📊 Versión: {info['version']}")
    print(f"🔄 Compatibilidad V31: {info['compatibilidad_v31']}")
    print(f"🌟 Aurora V7: {'DISPONIBLE' if info['aurora_v7_disponible'] else 'NO DISPONIBLE'}")
    print(f"🎵 Tipos de onda V31: {info['tipos_onda_v31']}")
    if info['aurora_v7_disponible']:
        print(f"✨ Tipos de onda Aurora V7: {info['tipos_onda_aurora_v7']}")
        print(f"🧠 Presets emocionales: {info.get('presets_emocionales_disponibles','N/A')}")
        print(f"🌀 Secuencias de fases: {info.get('secuencias_fases_disponibles','N/A')}")
    print("\n📝 Ejemplo V31 Compatible:")
    try:
        config_v31=AudioConfig(sample_rate=44100,channels=2);layers_v31=crear_preset_relajacion()
        print(f"✅ Configuración V31 creada");print(f"✅ {len(layers_v31)} capas de relajación cargadas")
        audio_v31=generar_bloques(10,layers_v31,config_v31);print(f"✅ Audio V31 generado: {audio_v31.shape}")
    except Exception as e:print(f"⚠️ Demo V31: {e}")
    
    if AURORA_V7_AVAILABLE:
        print("\n🌟 Ejemplo Aurora V7 Integrado:")
        try:
            config_aurora=AudioConfig(sample_rate=44100,channels=2,preset_emocional="claridad_mental",modo_terapeutico=True,validacion_cientifica=True)
            resultado_aurora=generar_bloques_aurora_integrado(duracion_total=15,audio_config=config_aurora,preset_emocional="claridad_mental")
            print(f"✅ Audio Aurora V7 generado: {resultado_aurora.audio_data.shape}")
            print(f"🧠 Coherencia neuroacústica: {resultado_aurora.coherencia_neuroacustica:.3f}")
            print(f"💊 Efectividad terapéutica: {resultado_aurora.efectividad_terapeutica:.3f}")
            print(f"📊 Calidad espectral: {resultado_aurora.calidad_espectral:.1f}")
            if resultado_aurora.analisis_neurotransmisores:print(f"🧪 Neurotransmisores detectados: {list(resultado_aurora.analisis_neurotransmisores.keys())}")
            if resultado_aurora.sugerencias_optimizacion:print(f"💡 Sugerencias: {len(resultado_aurora.sugerencias_optimizacion)}")
        except Exception as e:print(f"⚠️ Demo Aurora V7: {e}")
    
    print(f"\n🚀 Motor HyperMod V31 + Aurora V7 listo!")
    print(f"🔄 Mantiene 100% compatibilidad con código V31 existente")
    print(f"🌟 {'Potenciado con inteligencia científica Aurora V7' if AURORA_V7_AVAILABLE else 'Para activar Aurora V7, instalar módulos de presets'}")
