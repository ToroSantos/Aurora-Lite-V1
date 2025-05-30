import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import json
import warnings
from datetime import datetime

class CategoriaObjetivo(Enum):
    COGNITIVO="cognitivo";EMOCIONAL="emocional";ESPIRITUAL="espiritual";TERAPEUTICO="terapeutico";CREATIVO="creativo";FISICO="fisico";SOCIAL="social";EXPERIMENTAL="experimental"

class NivelComplejidad(Enum):
    PRINCIPIANTE="principiante";INTERMEDIO="intermedio";AVANZADO="avanzado";EXPERTO="experto"

class ModoActivacion(Enum):
    SUAVE="suave";PROGRESIVO="progresivo";INMEDIATO="inmediato";ADAPTATIVO="adaptativo"

@dataclass
class ConfiguracionCapaV7:
    enabled:bool=True;carrier:Optional[float]=None;mod_type:str="am";freq_l:Optional[float]=None;freq_r:Optional[float]=None;style:Optional[str]=None
    neurotransmisores:Dict[str,float]=field(default_factory=dict);frecuencias_personalizadas:List[float]=field(default_factory=list)
    modulacion_profundidad:float=0.5;modulacion_velocidad:float=0.1;fade_in_ms:int=500;fade_out_ms:int=1000;delay_inicio_ms:int=0
    duracion_relativa:float=1.0;pan_position:float=0.0;width_estereo:float=1.0;movimiento_espacial:bool=False;patron_movimiento:str="static"
    evolucion_temporal:bool=False;parametro_evolutivo:str="amplitude";curva_evolucion:str="linear"
    sincronizacion:List[str]=field(default_factory=list);modulacion_cruzada:Dict[str,float]=field(default_factory=dict)
    prioridad:int=1;calidad_renderizado:float=0.8;uso_cpu:str="medio"

@dataclass
class TemplateObjetivoV7:
    nombre:str;descripcion:str;categoria:CategoriaObjetivo;complejidad:NivelComplejidad=NivelComplejidad.INTERMEDIO
    emotional_preset:str;style:str;layers:Dict[str,ConfiguracionCapaV7]=field(default_factory=dict)
    neurotransmisores_principales:Dict[str,float]=field(default_factory=dict);ondas_cerebrales_objetivo:List[str]=field(default_factory=list)
    frecuencia_dominante:float=10.0;coherencia_neuroacustica:float=0.85;duracion_recomendada_min:int=20;duracion_minima_min:int=10
    duracion_maxima_min:int=60;fases_temporales:List[str]=field(default_factory=list);modo_activacion:ModoActivacion=ModoActivacion.PROGRESIVO
    efectos_esperados:List[str]=field(default_factory=list);efectos_secundarios:List[str]=field(default_factory=list)
    contraindicaciones:List[str]=field(default_factory=list);tiempo_efecto_min:int=5
    mejor_momento_uso:List[str]=field(default_factory=list);peor_momento_uso:List[str]=field(default_factory=list)
    ambiente_recomendado:str="tranquilo";postura_recomendada:str="comoda";nivel_atencion_requerido:str="medio"
    parametros_ajustables:List[str]=field(default_factory=list);variaciones_disponibles:List[str]=field(default_factory=list)
    nivel_personalizacion:str="medio";estudios_referencia:List[str]=field(default_factory=list);evidencia_cientifica:str="experimental"
    nivel_confianza:float=0.8;fecha_validacion:Optional[str]=None;investigadores:List[str]=field(default_factory=list)
    version:str="v7.1";autor:str="Aurora_V7_Optimized";fecha_creacion:str=field(default_factory=lambda:datetime.now().isoformat())
    ultima_actualizacion:str=field(default_factory=lambda:datetime.now().isoformat());compatibilidad_v6:bool=True
    tiempo_renderizado_estimado:int=30;recursos_cpu:str="medio";recursos_memoria:str="medio";calidad_audio:float=0.85
    veces_usado:int=0;rating_promedio:float=0.0;feedback_usuarios:List[str]=field(default_factory=list)
    
    def __post_init__(self):
        if not self.nombre:raise ValueError("Template debe tener nombre")
        if not self.emotional_preset:warnings.warn("Template sin preset emocional")
        if not 0<=self.nivel_confianza<=1:warnings.warn(f"Nivel confianza fuera de rango:{self.nivel_confianza}")
        if self.duracion_minima_min>=self.duracion_maxima_min:warnings.warn("Duración mínima >= máxima")
        for c in["neuro_wave","binaural","wave_pad","textured_noise","heartbeat"]:
            if c not in self.layers:self.layers[c]=ConfiguracionCapaV7(enabled=False)
        if self.neurotransmisores_principales and self.ondas_cerebrales_objetivo:self.coherencia_neuroacustica=self._calc_coherencia()
        ca=sum(1 for l in self.layers.values()if l.enabled);self.tiempo_renderizado_estimado=max(15,ca*8)
        ct=self._calc_complejidad();self.recursos_cpu="alto"if ct>0.8 else"medio"if ct>0.5 else"bajo";self.recursos_memoria=self.recursos_cpu
    
    def _calc_coherencia(self)->float:
        m={"gaba":["delta","theta"],"serotonina":["alpha","theta"],"dopamina":["beta"],"acetilcolina":["beta","gamma"],"norepinefrina":["beta","gamma"],"anandamida":["theta","delta"],"oxitocina":["alpha"],"melatonina":["delta"],"endorfina":["alpha","beta"]}
        st,pt=0.0,0.0
        for nt,i in self.neurotransmisores_principales.items():
            if nt.lower()in m:oe=m[nt.lower()];c=len(set(oe)&set([o.lower()for o in self.ondas_cerebrales_objetivo]));cn=c/len(oe)if oe else 0;st+=cn*i;pt+=i
        return st/pt if pt>0 else 0.5
    
    def _calc_complejidad(self)->float:
        c=sum(1 for l in self.layers.values()if l.enabled)*0.1
        for l in self.layers.values():
            if l.enabled:c+=0.15 if l.evolucion_temporal else 0;c+=0.1 if l.movimiento_espacial else 0;c+=0.1 if l.modulacion_cruzada else 0;c+=0.05 if l.neurotransmisores else 0
        c+={NivelComplejidad.EXPERTO:0.3,NivelComplejidad.AVANZADO:0.2,NivelComplejidad.INTERMEDIO:0.1}.get(self.complejidad,0)
        return min(c,1.0)

class GestorTemplatesOptimizado:
    def __init__(self):
        self.templates={};self.categorias_disponibles={};self.cache_busquedas={};self.estadisticas_uso={}
        self._migrar_v6();self._init_v7();self._org_cat()
    
    def _migrar_v6(self):
        tv6={
            "claridad_mental":{"v6":{"emotional_preset":"claridad_metal","style":"minimalista","layers":{"neuro_wave":{"enabled":True,"carrier":210,"mod_type":"am"},"binaural":{"enabled":True,"freq_l":100,"freq_r":111},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"soft_grain"},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.COGNITIVO,"complejidad":NivelComplejidad.INTERMEDIO,"descripcion":"Optimización cognitiva para concentración sostenida","neurotransmisores_principales":{"acetilcolina":0.9,"dopamina":0.6,"norepinefrina":0.4},"ondas_cerebrales_objetivo":["beta","alpha"],"frecuencia_dominante":11.0,"efectos_esperados":["Mejora concentración","Claridad mental","Reducción distracciones"],"mejor_momento_uso":["mañana","tarde"],"duracion_recomendada_min":25,"evidencia_cientifica":"validado","nivel_confianza":0.92}},
            "enfoque_total":{"v6":{"emotional_preset":"estado_flujo","style":"futurista","layers":{"neuro_wave":{"enabled":True,"carrier":420,"mod_type":"fm"},"binaural":{"enabled":True,"freq_l":110,"freq_r":122},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"crystalline"},"heartbeat":{"enabled":True}}},"v7":{"categoria":CategoriaObjetivo.COGNITIVO,"complejidad":NivelComplejidad.AVANZADO,"descripcion":"Estado de flujo profundo para rendimiento máximo","neurotransmisores_principales":{"dopamina":0.9,"norepinefrina":0.7,"endorfina":0.5},"ondas_cerebrales_objetivo":["beta","gamma"],"frecuencia_dominante":12.0,"efectos_esperados":["Estado de flujo","Rendimiento máximo","Inmersión total"],"mejor_momento_uso":["mañana","tarde"],"duracion_recomendada_min":35,"contraindicaciones":["hipertension","ansiedad_severa"],"evidencia_cientifica":"validado","nivel_confianza":0.94}},
            "relajacion_profunda":{"v6":{"emotional_preset":"calma_profunda","style":"etereo","layers":{"neuro_wave":{"enabled":True,"carrier":80,"mod_type":"am"},"binaural":{"enabled":True,"freq_l":80,"freq_r":88},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"breathy"},"heartbeat":{"enabled":True}}},"v7":{"categoria":CategoriaObjetivo.TERAPEUTICO,"complejidad":NivelComplejidad.PRINCIPIANTE,"descripcion":"Relajación terapéutica profunda","neurotransmisores_principales":{"gaba":0.9,"serotonina":0.8,"melatonina":0.6},"ondas_cerebrales_objetivo":["theta","alpha"],"frecuencia_dominante":8.0,"efectos_esperados":["Relajación profunda","Reducción estrés","Calma mental"],"mejor_momento_uso":["tarde","noche"],"duracion_recomendada_min":30,"evidencia_cientifica":"clinico","nivel_confianza":0.96}},
            "bienestar_tribal":{"v6":{"emotional_preset":"fuerza_tribal","style":"tribal","layers":{"neuro_wave":{"enabled":True,"carrier":300,"mod_type":"am"},"binaural":{"enabled":True,"freq_l":90,"freq_r":103},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"dense"},"heartbeat":{"enabled":True}}},"v7":{"categoria":CategoriaObjetivo.SOCIAL,"complejidad":NivelComplejidad.INTERMEDIO,"descripcion":"Conexión grupal y fuerza colectiva","neurotransmisores_principales":{"dopamina":0.7,"endorfina":0.8,"oxitocina":0.6},"ondas_cerebrales_objetivo":["alpha","theta"],"frecuencia_dominante":13.0,"efectos_esperados":["Conexión grupal","Fuerza colectiva","Energia tribal"],"mejor_momento_uso":["tarde","noche"],"duracion_recomendada_min":25,"evidencia_cientifica":"validado","nivel_confianza":0.85}},
            "integracion_emocional":{"v6":{"emotional_preset":"regulacion_emocional","style":"organico","layers":{"neuro_wave":{"enabled":True,"carrier":130,"mod_type":"hybrid"},"binaural":{"enabled":False},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"warm_dust"},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.EMOCIONAL,"complejidad":NivelComplejidad.INTERMEDIO,"descripcion":"Procesamiento y equilibrio emocional","neurotransmisores_principales":{"serotonina":0.8,"gaba":0.6,"oxitocina":0.5},"ondas_cerebrales_objetivo":["alpha","theta"],"frecuencia_dominante":9.0,"efectos_esperados":["Equilibrio emocional","Procesamiento traumas","Autorregulación"],"mejor_momento_uso":["tarde","noche"],"duracion_recomendada_min":20,"evidencia_cientifica":"validado","nivel_confianza":0.88}},
            "meditacion_chamanica":{"v6":{"emotional_preset":"introspeccion_suave","style":"tribal","layers":{"neuro_wave":{"enabled":False},"binaural":{"enabled":True,"freq_l":85,"freq_r":90},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"dense"},"heartbeat":{"enabled":True}}},"v7":{"categoria":CategoriaObjetivo.ESPIRITUAL,"complejidad":NivelComplejidad.AVANZADO,"descripcion":"Viaje introspectivo profundo","neurotransmisores_principales":{"serotonina":0.7,"anandamida":0.6,"gaba":0.5},"ondas_cerebrales_objetivo":["theta","delta"],"frecuencia_dominante":5.0,"efectos_esperados":["Introspección profunda","Conexión ancestral","Visiones internas"],"mejor_momento_uso":["noche"],"duracion_recomendada_min":40,"contraindicaciones":["primera_experiencia"],"evidencia_cientifica":"experimental","nivel_confianza":0.78}},
            "conexion_astral":{"v6":{"emotional_preset":"conexion_mistica","style":"alienigena","layers":{"neuro_wave":{"enabled":True,"carrier":333,"mod_type":"hybrid"},"binaural":{"enabled":True,"freq_l":95,"freq_r":102},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"crystalline"},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.EXPERIMENTAL,"complejidad":NivelComplejidad.EXPERTO,"descripcion":"Experiencia multidimensional","neurotransmisores_principales":{"anandamida":0.8,"serotonina":0.6,"oxitocina":0.4},"ondas_cerebrales_objetivo":["theta","delta"],"frecuencia_dominante":7.0,"efectos_esperados":["Proyección astral","Experiencia extracorpórea","Expansión dimensional"],"mejor_momento_uso":["noche"],"duracion_recomendada_min":60,"contraindicaciones":["epilepsia","trastornos_psiquiatricos","primera_experiencia"],"evidencia_cientifica":"experimental","nivel_confianza":0.65}},
            "detox_emocional":{"v6":{"emotional_preset":"regulacion_emocional","style":"organico","layers":{"neuro_wave":{"enabled":True,"carrier":150,"mod_type":"am"},"binaural":{"enabled":False},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"warm_dust"},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.TERAPEUTICO,"complejidad":NivelComplejidad.INTERMEDIO,"descripcion":"Limpieza y purificación emocional","neurotransmisores_principales":{"gaba":0.8,"serotonina":0.7,"oxitocina":0.5},"ondas_cerebrales_objetivo":["alpha","theta"],"frecuencia_dominante":7.83,"efectos_esperados":["Limpieza emocional","Liberación traumas","Renovación energética"],"mejor_momento_uso":["tarde"],"duracion_recomendada_min":25,"evidencia_cientifica":"validado","nivel_confianza":0.88}},
            "activacion_cuanticadelser":{"v6":{"emotional_preset":"expansion_creativa","style":"mistico","layers":{"neuro_wave":{"enabled":True,"carrier":240,"mod_type":"hybrid"},"binaural":{"enabled":True,"freq_l":105,"freq_r":117},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"shimmer"},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.EXPERIMENTAL,"complejidad":NivelComplejidad.EXPERTO,"descripcion":"Activación de potencial creativo","neurotransmisores_principales":{"dopamina":0.8,"anandamida":0.7,"acetilcolina":0.6},"ondas_cerebrales_objetivo":["gamma","theta"],"frecuencia_dominante":12.0,"efectos_esperados":["Activación cuántica","Creatividad exponencial","Visión multidimensional"],"mejor_momento_uso":["noche"],"duracion_recomendada_min":60,"contraindicaciones":["epilepsia","trastornos_psiquiatricos","primera_experiencia"],"evidencia_cientifica":"experimental","nivel_confianza":0.68}},
            "sueno_consciente":{"v6":{"emotional_preset":"calma_profunda","style":"etereo","layers":{"neuro_wave":{"enabled":False},"binaural":{"enabled":True,"freq_l":80,"freq_r":85},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"soft_grain"},"heartbeat":{"enabled":True}}},"v7":{"categoria":CategoriaObjetivo.EXPERIMENTAL,"complejidad":NivelComplejidad.AVANZADO,"descripcion":"Inducción de sueños lúcidos","neurotransmisores_principales":{"gaba":0.9,"melatonina":0.8,"serotonina":0.6},"ondas_cerebrales_objetivo":["delta","theta"],"frecuencia_dominante":5.0,"efectos_esperados":["Sueños lúcidos","Control onírico","Exploración subconsciente"],"mejor_momento_uso":["noche"],"duracion_recomendada_min":90,"contraindicaciones":["insomnio_cronico"],"evidencia_cientifica":"experimental","nivel_confianza":0.72}},
            "presencia_total":{"v6":{"emotional_preset":"seguridad_interior","style":"minimalista","layers":{"neuro_wave":{"enabled":True,"carrier":120,"mod_type":"am"},"binaural":{"enabled":False},"wave_pad":{"enabled":True},"textured_noise":{"enabled":True,"style":"centered"},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.ESPIRITUAL,"complejidad":NivelComplejidad.INTERMEDIO,"descripcion":"Mindfulness profundo y presencia consciente","neurotransmisores_principales":{"gaba":0.7,"serotonina":0.8,"acetilcolina":0.5},"ondas_cerebrales_objetivo":["alpha","theta"],"frecuencia_dominante":10.0,"efectos_esperados":["Presencia total","Mindfulness","Consciencia expandida"],"mejor_momento_uso":["mañana","tarde"],"duracion_recomendada_min":20,"evidencia_cientifica":"validado","nivel_confianza":0.91}},
            "agradecimiento_activo":{"v6":{"emotional_preset":"apertura_corazon","style":"sutil","layers":{"neuro_wave":{"enabled":False},"binaural":{"enabled":True,"freq_l":88,"freq_r":96},"wave_pad":{"enabled":True},"textured_noise":{"enabled":False},"heartbeat":{"enabled":False}}},"v7":{"categoria":CategoriaObjetivo.EMOCIONAL,"complejidad":NivelComplejidad.PRINCIPIANTE,"descripcion":"Cultivo de gratitud profunda","neurotransmisores_principales":{"oxitocina":0.9,"serotonina":0.7,"endorfina":0.6},"ondas_cerebrales_objetivo":["alpha"],"frecuencia_dominante":8.0,"efectos_esperados":["Gratitud profunda","Apertura corazón","Conexión amorosa"],"mejor_momento_uso":["mañana","tarde"],"duracion_recomendada_min":15,"evidencia_cientifica":"validado","nivel_confianza":0.89}}
        }
        for n,d in tv6.items():
            v6,v7=d["v6"],d["v7"];lv7={}
            for ln,lc in v6["layers"].items():
                lv7[ln]=ConfiguracionCapaV7(enabled=lc.get("enabled",False),carrier=lc.get("carrier"),mod_type=lc.get("mod_type","am"),freq_l=lc.get("freq_l"),freq_r=lc.get("freq_r"),style=lc.get("style"),modulacion_profundidad=0.5+(len(ln)%3)*0.1,fade_in_ms=1000 if lc.get("enabled")else 0,fade_out_ms=1500 if lc.get("enabled")else 0,prioridad=3 if ln in["neuro_wave","binaural"]else 2)
            self.templates[n]=TemplateObjetivoV7(nombre=n.replace("_"," ").title(),emotional_preset=v6["emotional_preset"],style=v6["style"],layers=lv7,**v7)
    
    def _init_v7(self):
        self.templates["hiperfocus_cuantico"]=TemplateObjetivoV7(nombre="Hiperfocus Cuántico",descripcion="Estado de concentración extrema",categoria=CategoriaObjetivo.COGNITIVO,complejidad=NivelComplejidad.EXPERTO,emotional_preset="estado_cuantico_cognitivo",style="cuantico_cristalino",layers={"neuro_wave":ConfiguracionCapaV7(enabled=True,carrier=432.0,mod_type="quantum_hybrid",neurotransmisores={"acetilcolina":0.95,"dopamina":0.8,"norepinefrina":0.7},modulacion_profundidad=0.7,modulacion_velocidad=0.618,evolucion_temporal=True,parametro_evolutivo="quantum_coherence",curva_evolucion="fibonacci",prioridad=5,calidad_renderizado=1.0),"binaural":ConfiguracionCapaV7(enabled=True,freq_l=110.0,freq_r=152.0,movimiento_espacial=True,patron_movimiento="quantum_spiral",modulacion_cruzada={"neuro_wave":0.618},prioridad=4),"wave_pad":ConfiguracionCapaV7(enabled=True,style="quantum_harmonics",frecuencias_personalizadas=[111.0,222.0,333.0,444.0,555.0],width_estereo=2.0,evolucion_temporal=True,parametro_evolutivo="dimensional_shift",prioridad=3),"textured_noise":ConfiguracionCapaV7(enabled=True,style="quantum_static",modulacion_profundidad=0.3,sincronizacion=["neuro_wave"],prioridad=2)},neurotransmisores_principales={"acetilcolina":0.95,"dopamina":0.8,"norepinefrina":0.7,"gaba":0.3},ondas_cerebrales_objetivo=["gamma","beta"],frecuencia_dominante=42.0,duracion_recomendada_min=45,duracion_minima_min=30,duracion_maxima_min=90,efectos_esperados=["Concentración extrema","Procesamiento cuántico","Creatividad + lógica"],contraindicaciones=["epilepsia","trastornos_atencion","hipertension_severa"],mejor_momento_uso=["mañana"],peor_momento_uso=["noche"],nivel_atencion_requerido="alto",evidencia_cientifica="experimental",nivel_confianza=0.75,tiempo_efecto_min=10)
        self.templates["sanacion_multidimensional"]=TemplateObjetivoV7(nombre="Sanación Multidimensional",descripcion="Proceso de sanación integral",categoria=CategoriaObjetivo.TERAPEUTICO,complejidad=NivelComplejidad.AVANZADO,emotional_preset="sanacion_integral",style="medicina_sagrada",layers={"neuro_wave":ConfiguracionCapaV7(enabled=True,carrier=528.0,mod_type="healing_wave",neurotransmisores={"serotonina":0.9,"oxitocina":0.8,"endorfina":0.7,"gaba":0.6},modulacion_profundidad=0.4,evolucion_temporal=True,parametro_evolutivo="healing_intensity",curva_evolucion="healing_spiral",fade_in_ms=5000,fade_out_ms=3000,prioridad=5),"binaural":ConfiguracionCapaV7(enabled=True,freq_l=85.0,freq_r=93.0,movimiento_espacial=True,patron_movimiento="healing_embrace",fade_in_ms=3000,prioridad=4),"wave_pad":ConfiguracionCapaV7(enabled=True,style="chakra_harmonics",frecuencias_personalizadas=[174.0,285.0,396.0,417.0,528.0,639.0,741.0,852.0,963.0],width_estereo=1.8,evolucion_temporal=True,parametro_evolutivo="chakra_activation",prioridad=4),"textured_noise":ConfiguracionCapaV7(enabled=True,style="healing_light",modulacion_profundidad=0.2,delay_inicio_ms=8000,prioridad=2),"heartbeat":ConfiguracionCapaV7(enabled=True,style="universal_heart",modulacion_profundidad=0.3,sincronizacion=["neuro_wave"],prioridad=3)},neurotransmisores_principales={"serotonina":0.9,"oxitocina":0.8,"endorfina":0.7,"gaba":0.6},ondas_cerebrales_objetivo=["alpha","theta"],frecuencia_dominante=8.0,duracion_recomendada_min=50,duracion_minima_min=30,duracion_maxima_min=90,fases_temporales=["preparacion","activacion_chakras","sanacion_profunda","integracion","cierre"],efectos_esperados=["Sanación emocional","Equilibrio energético","Liberación traumas"],mejor_momento_uso=["tarde","noche"],ambiente_recomendado="sagrado_silencioso",postura_recomendada="recostada",evidencia_cientifica="clinico",nivel_confianza=0.87)
        self.templates["creatividad_exponencial"]=TemplateObjetivoV7(nombre="Creatividad Exponencial",descripcion="Desbloqueo masivo del potencial creativo",categoria=CategoriaObjetivo.CREATIVO,complejidad=NivelComplejidad.AVANZADO,emotional_preset="genio_creativo",style="vanguardia_artistica",layers={"neuro_wave":ConfiguracionCapaV7(enabled=True,carrier=256.0,mod_type="creative_chaos",neurotransmisores={"dopamina":0.9,"anandamida":0.8,"acetilcolina":0.7,"serotonina":0.6},modulacion_profundidad=0.6,modulacion_velocidad=0.15,evolucion_temporal=True,parametro_evolutivo="creative_explosion",curva_evolucion="exponential",prioridad=5),"binaural":ConfiguracionCapaV7(enabled=True,freq_l=105.0,freq_r=115.0,movimiento_espacial=True,patron_movimiento="creative_dance",modulacion_cruzada={"wave_pad":0.4},prioridad=4),"wave_pad":ConfiguracionCapaV7(enabled=True,style="artistic_chaos",frecuencias_personalizadas=[256.0,341.3,426.7,512.0,682.7],width_estereo=2.2,evolucion_temporal=True,parametro_evolutivo="artistic_flow",movimiento_espacial=True,patron_movimiento="inspiration_spiral",prioridad=4),"textured_noise":ConfiguracionCapaV7(enabled=True,style="creative_sparks",modulacion_profundidad=0.4,modulacion_velocidad=0.2,prioridad=3)},neurotransmisores_principales={"dopamina":0.9,"anandamida":0.8,"acetilcolina":0.7,"serotonina":0.6},ondas_cerebrales_objetivo=["alpha","theta","gamma"],frecuencia_dominante=10.0,duracion_recomendada_min=35,efectos_esperados=["Explosión creativa","Ideas revolucionarias","Conexión artística"],mejor_momento_uso=["mañana","tarde"],ambiente_recomendado="espacio_creativo",nivel_atencion_requerido="medio",evidencia_cientifica="validado",nivel_confianza=0.84)
    
    def _org_cat(self):
        for t in self.templates.values():
            c=t.categoria.value;self.categorias_disponibles.setdefault(c,[]).append(t.nombre.lower().replace(" ","_"))
    
    @lru_cache(maxsize=128)
    def obtener_template(self,nombre:str)->Optional[TemplateObjetivoV7]:
        nn=nombre.lower().replace(" ","_");t=self.templates.get(nn)
        if t:self._upd_stats(nn)
        return t
    
    def _upd_stats(self,nombre:str):
        if nombre not in self.estadisticas_uso:self.estadisticas_uso[nombre]={"veces_usado":0,"ultima_vez":None,"rating_promedio":0.0,"feedback_total":0}
        self.estadisticas_uso[nombre]["veces_usado"]+=1;self.estadisticas_uso[nombre]["ultima_vez"]=datetime.now().isoformat()
        if nombre in self.templates:self.templates[nombre].veces_usado=self.estadisticas_uso[nombre]["veces_usado"]
    
    def buscar_templates_inteligente(self,criterios:Dict[str,Any],limite:int=10)->List[TemplateObjetivoV7]:
        ck=json.dumps(criterios,sort_keys=True)+f"_limit_{limite}"
        if ck in self.cache_busquedas:return self.cache_busquedas[ck]
        candidatos=list(self.templates.values());puntuaciones=[(t,self._calc_relevancia(t,criterios))for t in candidatos]
        puntuaciones.sort(key=lambda x:x[1],reverse=True);resultados=[t for t,p in puntuaciones[:limite]if p>0.1]
        self.cache_busquedas[ck]=resultados;return resultados
    
    def _calc_relevancia(self,template:TemplateObjetivoV7,criterios:Dict[str,Any])->float:
        p=0.0
        if"categoria"in criterios and template.categoria.value==criterios["categoria"]:p+=0.3
        if"efectos"in criterios:eb=[e.lower()for e in criterios["efectos"]];et=[e.lower()for e in template.efectos_esperados];p+=len(set(eb)&set(et))*0.2
        if"neurotransmisores"in criterios:nb=set(criterios["neurotransmisores"]);nt=set(template.neurotransmisores_principales.keys());p+=len(nb&nt)*0.15
        if"complejidad"in criterios and template.complejidad.value==criterios["complejidad"]:p+=0.1
        if"duracion_min"in criterios:db=criterios["duracion_min"];p+=0.1 if template.duracion_minima_min<=db<=template.duracion_maxima_min else 0
        if"momento"in criterios and criterios["momento"]in template.mejor_momento_uso:p+=0.05
        p*=template.nivel_confianza;p+=min(template.veces_usado/100,1.0)*0.05 if template.veces_usado>0 else 0
        return p
    
    def recomendar_templates(self,perfil_usuario:Dict[str,Any],limite:int=5)->List[Tuple[TemplateObjetivoV7,float,str]]:
        recomendaciones=[(t,self._calc_compatibilidad(t,perfil_usuario),self._gen_razon(t,perfil_usuario))for t in self.templates.values()]
        recomendaciones=[(t,c,r)for t,c,r in recomendaciones if c>0.3];recomendaciones.sort(key=lambda x:x[1],reverse=True)
        return recomendaciones[:limite]
    
    def _calc_compatibilidad(self,template:TemplateObjetivoV7,perfil:Dict[str,Any])->float:
        c=0.0;exp=perfil.get("experiencia","principiante")
        if exp=="principiante"and template.complejidad in[NivelComplejidad.PRINCIPIANTE,NivelComplejidad.INTERMEDIO]:c+=0.3
        elif exp=="intermedio"and template.complejidad!=NivelComplejidad.EXPERTO:c+=0.3
        elif exp=="avanzado":c+=0.3
        for obj in perfil.get("objetivos",[]):
            if obj.lower()in[e.lower()for e in template.efectos_esperados]:c+=0.2
        mp=perfil.get("momento_preferido")
        if mp and mp in template.mejor_momento_uso:c+=0.1
        for cond in perfil.get("condiciones_medicas",[]):
            if cond in template.contraindicaciones:c-=0.5
        dp=perfil.get("duracion_preferida",20)
        if template.duracion_minima_min<=dp<=template.duracion_maxima_min:c+=0.1
        return max(0.0,c)
    
    def _gen_razon(self,template:TemplateObjetivoV7,perfil:Dict[str,Any])->str:
        razones=[];objs=perfil.get("objetivos",[])
        ec=[e for e in template.efectos_esperados if any(obj.lower()in e.lower()for obj in objs)]
        if ec:razones.append(f"Ayuda con: {', '.join(ec[:2])}")
        exp=perfil.get("experiencia","principiante")
        if template.complejidad.value==exp:razones.append(f"Perfecto para nivel {exp}")
        if template.evidencia_cientifica in["validado","clinico"]:razones.append("Respaldado científicamente")
        if template.nivel_confianza>0.85:razones.append("Alta efectividad")
        return" • ".join(razones)if razones else"Compatible con tu perfil"
    
    def generar_template_personalizado(self,objetivo_personalizado:str,parametros:Dict[str,Any])->TemplateObjetivoV7:
        cat=self._infer_categoria(objetivo_personalizado);comp=NivelComplejidad(parametros.get("complejidad","intermedio"))
        nt=self._infer_nt(objetivo_personalizado,cat);oc=self._infer_ondas(objetivo_personalizado,cat)
        tp=TemplateObjetivoV7(nombre=f"Personalizado: {objetivo_personalizado.title()}",descripcion=f"Template personalizado: {objetivo_personalizado}",categoria=cat,complejidad=comp,emotional_preset=parametros.get("emotional_preset","personalizado"),style=parametros.get("style","adaptativo"),neurotransmisores_principales=nt,ondas_cerebrales_objetivo=oc,duracion_recomendada_min=parametros.get("duracion",20),efectos_esperados=[objetivo_personalizado],evidencia_cientifica="personalizado",nivel_confianza=0.7,autor="Usuario_Personalizado")
        tp.layers=self._gen_capas_auto(cat,nt,oc,parametros);return tp
    
    def _infer_categoria(self,objetivo:str)->CategoriaObjetivo:
        ol=objetivo.lower()
        if any(w in ol for w in["concentrar","enfocar","estudiar","trabajar","claridad"]):return CategoriaObjetivo.COGNITIVO
        elif any(w in ol for w in["relajar","dormir","ansiedad","estres","calmar"]):return CategoriaObjetivo.TERAPEUTICO
        elif any(w in ol for w in["crear","inspirar","arte","musica","innovar"]):return CategoriaObjetivo.CREATIVO
        elif any(w in ol for w in["meditar","espiritual","conexion","consciencia"]):return CategoriaObjetivo.ESPIRITUAL
        elif any(w in ol for w in["amor","gratitud","compasion","corazon"]):return CategoriaObjetivo.EMOCIONAL
        elif any(w in ol for w in["ejercicio","deporte","fisico","energia"]):return CategoriaObjetivo.FISICO
        elif any(w in ol for w in["social","grupo","comunidad","compartir"]):return CategoriaObjetivo.SOCIAL
        else:return CategoriaObjetivo.EXPERIMENTAL
    
    def _infer_nt(self,objetivo:str,categoria:CategoriaObjetivo)->Dict[str,float]:
        ol=objetivo.lower();nt={}
        if any(w in ol for w in["concentrar","enfocar","atencion"]):nt.update({"acetilcolina":0.8,"dopamina":0.6})
        if any(w in ol for w in["relajar","calmar","tranquil"]):nt.update({"gaba":0.8,"serotonina":0.6})
        if any(w in ol for w in["crear","inspirar","innovar"]):nt.update({"dopamina":0.7,"anandamida":0.5})
        if any(w in ol for w in["amor","conexion","compasion"]):nt.update({"oxitocina":0.8,"serotonina":0.6})
        if not nt:
            defaults={CategoriaObjetivo.COGNITIVO:{"acetilcolina":0.7,"dopamina":0.5},CategoriaObjetivo.TERAPEUTICO:{"gaba":0.7,"serotonina":0.6},CategoriaObjetivo.CREATIVO:{"dopamina":0.6,"anandamida":0.4},CategoriaObjetivo.ESPIRITUAL:{"serotonina":0.6,"anandamida":0.4},CategoriaObjetivo.EMOCIONAL:{"oxitocina":0.7,"serotonina":0.5},CategoriaObjetivo.FISICO:{"endorfina":0.6,"dopamina":0.4},CategoriaObjetivo.SOCIAL:{"oxitocina":0.6,"dopamina":0.4}}
            nt=defaults.get(categoria,{"serotonina":0.5})
        return nt
    
    def _infer_ondas(self,objetivo:str,categoria:CategoriaObjetivo)->List[str]:
        ol=objetivo.lower()
        if any(w in ol for w in["concentrar","enfocar","trabajar"]):return["beta"]
        elif any(w in ol for w in["relajar","meditar","calmar"]):return["alpha","theta"]
        elif any(w in ol for w in["dormir","descansar"]):return["delta","theta"]
        elif any(w in ol for w in["crear","inspirar"]):return["alpha","theta"]
        else:
            defaults={CategoriaObjetivo.COGNITIVO:["beta","alpha"],CategoriaObjetivo.TERAPEUTICO:["alpha","theta"],CategoriaObjetivo.CREATIVO:["alpha","theta"],CategoriaObjetivo.ESPIRITUAL:["theta","delta"],CategoriaObjetivo.EMOCIONAL:["alpha"],CategoriaObjetivo.FISICO:["beta","alpha"],CategoriaObjetivo.SOCIAL:["alpha"]}
            return defaults.get(categoria,["alpha"])
    
    def _gen_capas_auto(self,categoria:CategoriaObjetivo,neurotransmisores:Dict[str,float],ondas_objetivo:List[str],parametros:Dict[str,Any])->Dict[str,ConfiguracionCapaV7]:
        capas={}
        if neurotransmisores:capas["neuro_wave"]=ConfiguracionCapaV7(enabled=True,carrier=parametros.get("carrier_freq",150.0),mod_type=parametros.get("mod_type","am"),neurotransmisores=neurotransmisores,modulacion_profundidad=parametros.get("intensidad",0.5),prioridad=4)
        if ondas_objetivo:
            fb=100.0;bf={"delta":2.0,"theta":6.0,"alpha":10.0,"beta":15.0,"gamma":30.0}.get(ondas_objetivo[0],10.0)
            capas["binaural"]=ConfiguracionCapaV7(enabled=True,freq_l=fb,freq_r=fb+bf,prioridad=3)
        if parametros.get("armonicos",True):capas["wave_pad"]=ConfiguracionCapaV7(enabled=True,style="adaptive_harmonics",prioridad=2)
        if parametros.get("textura",True):capas["textured_noise"]=ConfiguracionCapaV7(enabled=True,style="adaptive_texture",modulacion_profundidad=0.2,prioridad=1)
        return capas
    
    def exportar_estadisticas(self)->Dict[str,Any]:
        return{"version":"v7.1_optimized","total_templates":len(self.templates),"templates_por_categoria":{cat.value:len([t for t in self.templates.values()if t.categoria==cat])for cat in CategoriaObjetivo},"templates_por_complejidad":{comp.value:len([t for t in self.templates.values()if t.complejidad==comp])for comp in NivelComplejidad},"confianza_promedio":sum(t.nivel_confianza for t in self.templates.values())/len(self.templates),"templates_mas_usados":sorted([(n,s["veces_usado"])for n,s in self.estadisticas_uso.items()],key=lambda x:x[1],reverse=True)[:10],"cache_efectividad":{"busquedas_cacheadas":len(self.cache_busquedas),"templates_cacheados":self.obtener_template.cache_info()._asdict()}}
    
    def limpiar_cache(self):self.obtener_template.cache_clear();self.cache_busquedas.clear()

OBJECTIVE_TEMPLATES={}

def _generar_compatibilidad_v6():
    global OBJECTIVE_TEMPLATES;gestor=GestorTemplatesOptimizado()
    for nombre,template in gestor.templates.items():
        lv6={nc:{k:v for k,v in{"enabled":cc.enabled,"carrier":cc.carrier,"mod_type":cc.mod_type,"freq_l":cc.freq_l,"freq_r":cc.freq_r,"style":cc.style}.items()if v is not None}for nc,cc in template.layers.items()}
        OBJECTIVE_TEMPLATES[nombre]={"emotional_preset":template.emotional_preset,"style":template.style,"layers":lv6}

_generar_compatibilidad_v6()

def crear_gestor_optimizado()->GestorTemplatesOptimizado:return GestorTemplatesOptimizado()
def buscar_templates_por_objetivo(objetivo:str,limite:int=5)->List[TemplateObjetivoV7]:return crear_gestor_optimizado().buscar_templates_inteligente({"efectos":[objetivo]},limite)
def recomendar_para_perfil(perfil_usuario:Dict[str,Any])->List[Tuple[str,float,str]]:return[(t.nombre,c,r)for t,c,r in crear_gestor_optimizado().recomendar_templates(perfil_usuario)]
def validar_template_avanzado(template:TemplateObjetivoV7)->Dict[str,Any]:
    v={"valido":True,"puntuacion_calidad":0.0,"advertencias":[],"errores":[],"sugerencias":[],"metricas":{}};c=template.coherencia_neuroacustica;v["metricas"]["coherencia_neuroacustica"]=c
    if c<0.5:v["advertencias"].append("Baja coherencia neuroacústica");v["puntuacion_calidad"]-=0.2
    elif c>0.8:v["puntuacion_calidad"]+=0.3
    ca=[c for c in template.layers.values()if c.enabled];v["metricas"]["capas_activas"]=len(ca)
    if not ca:v["errores"].append("No hay capas activas");v["valido"]=False
    elif len(ca)>6:v["advertencias"].append("Muchas capas activas pueden afectar rendimiento")
    if template.duracion_recomendada_min<5:v["advertencias"].append("Duración muy corta")
    elif template.duracion_recomendada_min>90:v["advertencias"].append("Duración muy larga")
    else:v["puntuacion_calidad"]+=0.1
    if template.evidencia_cientifica in["validado","clinico"]:v["puntuacion_calidad"]+=0.2
    elif template.evidencia_cientifica=="experimental":v["puntuacion_calidad"]+=0.1
    v["puntuacion_calidad"]=max(0.0,min(1.0,v["puntuacion_calidad"]+0.5));return v

if __name__=="__main__":
    print("🧠 Aurora V7 - Templates Optimizados");print("="*40);g=crear_gestor_optimizado();s=g.exportar_estadisticas()
    print(f"📊 {s['total_templates']} templates disponibles");print(f"🎯 Confianza promedio: {s['confianza_promedio']:.2%}")
    print("\n📋 Por categoría:");[print(f"  • {c.title()}: {cnt}")for c,cnt in s["templates_por_categoria"].items()if cnt>0]
    print("\n🔍 Búsqueda 'concentración':");[print(f"  • {t.nombre} ({t.nivel_confianza:.0%})")for t in buscar_templates_por_objetivo("concentración",3)]
    print("\n💡 Recomendación principiante:");p={"experiencia":"principiante","objetivos":["relajación","concentración"],"duracion_preferida":20}
    [print(f"  • {n} ({c:.0%}) - {r}")for n,c,r in recomendar_para_perfil(p)[:2]]
    print(f"\n✅ Sistema V7.1 listo - {len(OBJECTIVE_TEMPLATES)} templates V6")
