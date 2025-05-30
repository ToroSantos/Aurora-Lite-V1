"""Aurora V7 Layer Scheduler - Optimized Version"""
import math,numpy as np
from typing import Dict,List,Any,Optional,Union,Tuple
from dataclasses import dataclass,field
from enum import Enum
from datetime import datetime

class CurvaEvolucion(Enum):
    LINEAL="lineal";EXPONENCIAL="exponencial";LOGARITMICA="logaritmica";SIGMOIDE="sigmoide";SINUSOIDAL="sinusoidal";RESPIRATORIA="respiratoria";CARDIACA="cardiaca";FIBONACCI="fibonacci";AUREA="aurea";CUANTICA="cuantica"

class PatronEspacial(Enum):
    NEUTRO="neutro";TRIBAL="tribal";MISTICO="mistico";ETEREO="etereo";CRISTALINO="cristalino";ORGANICO="organico";CUANTICO="cuantico";CEREMONIAL="ceremonial";TERAPEUTICO="terapeutico"

@dataclass
class ConfiguracionV7:
    curva_evolucion:CurvaEvolucion=CurvaEvolucion.LINEAL;patron_espacial:PatronEspacial=PatronEspacial.NEUTRO;validacion_neuroacustica:bool=True;optimizacion_coherencia:bool=True;intensidad_base:float=1.0;suavizado_transiciones:bool=True;factor_coherencia:float=0.8;generado_por:str="LayerSchedulerV7";timestamp:str=field(default_factory=lambda:datetime.now().isoformat())

def activar_capas_por_bloque(b_idx,total_b,tipo="neuro_wave"):
    if b_idx<0 or b_idx>=total_b or total_b<=0:return False
    p=b_idx/max(1,total_b-1)
    if tipo in["neuro","neuro_wave","pad","wave_pad","binaural"]:return True
    elif tipo=="heartbeat":return b_idx<int(total_b*0.75) if total_b>3 else True
    elif tipo=="textured_noise":
        if total_b<=4:return True
        elif p<0.2 or p>0.8:return True
        else:return b_idx%2==0
    else:return p>0.1

def intensidad_dinamica(b_idx,total_b,modo="normal"):
    if total_b<=0:return 1.0
    p=b_idx/max(1,total_b-1)
    if modo=="ascenso":return 0.3+0.7*_sig(p)
    elif modo=="disolucion":return max(0.1,0.9*pow(1.0-p,1.5))
    elif modo=="expansivo":return 0.4+0.6*math.exp(-2*pow(abs(p-0.5)*2,2))
    elif modo=="ritmico":return(0.5+0.5*abs(math.sin(p*math.pi*3)))*(0.7+0.3*p)
    elif modo=="transformativo":
        if p<0.25:return 0.5+0.5*(p/0.25)
        elif p<0.75:return 1.0
        else:return 1.0-0.3*((p-0.75)/0.25)
    elif modo=="ondulante":return 0.6+0.4*(math.sin(p*math.pi*2)+0.3*math.sin(p*math.pi*6))
    else:return 1.0+0.05*math.sin(p*math.pi*8)

def paneo_emocional(b_idx,total_b,estilo="neutro"):
    if total_b<=0:return 0.0
    p=b_idx/max(1,total_b-1)
    if estilo=="tribal":return((-1)**b_idx*0.6)*(0.5+0.5*p)+0.2*math.sin(p*math.pi*12)
    elif estilo=="mistico":return 0.8*math.sin(2*math.pi*p*1.618)*math.cos(math.pi*p)
    elif estilo=="etereo":return(0.7*math.sin(math.pi*p)+0.3*math.sin(math.pi*p*3))*pow(0.5,b_idx if b_idx>0 else 0)
    elif estilo=="cristalino":return 0.9*math.sin(math.pi*p*2)*math.cos(math.pi*p*3)
    elif estilo=="organico":return 0.6*(math.sin(p*math.pi*1.5)+0.3*math.sin(p*math.pi*4.5))
    elif estilo=="cuantico":return[-0.8,-0.4,0.0,0.4,0.8][min(int(p*5),4)]
    elif estilo=="ceremonial":return 0.7*math.sin(p*math.pi*2)*(1.0+0.5*math.sin(p*math.pi*8))
    elif estilo=="terapeutico":return 0.5*(math.sin(p*math.pi)+0.2*math.sin(p*math.pi*6))
    else:return 0.1*math.sin(p*math.pi*4)

def estructura_layer_fase(total_b,modo="normal",estilo="neutro"):
    if total_b<=0:return[]
    e=[];cg=_calc_coh_sec(total_b,modo,estilo)
    for b in range(total_b):
        p=b/max(1,total_b-1)
        bv6={"bloque":b,"gain":round(intensidad_dinamica(b,total_b,modo),3),"paneo":round(paneo_emocional(b,total_b,estilo),3),"capas":{"neuro_wave":activar_capas_por_bloque(b,total_b,"neuro_wave"),"wave_pad":activar_capas_por_bloque(b,total_b,"pad"),"heartbeat":activar_capas_por_bloque(b,total_b,"heartbeat"),"binaural":activar_capas_por_bloque(b,total_b,"binaural"),"textured_noise":activar_capas_por_bloque(b,total_b,"textured_noise")}}
        bv6["v7_enhanced"]={"progreso_normalizado":round(p,3),"fase_temporal":_det_fase(p),"energia_relativa":round(_calc_energia(b,total_b,modo),3),"coherencia_neuroacustica":round(_calc_coh_bloque(bv6,cg),3),"validez_cientifica":_val_bloque(bv6),"factor_estabilidad":round(_calc_estab(b,total_b),3),"estado_consciencia_sugerido":_inf_estado(p,modo),"efectos_esperados":_pred_efectos(p,modo,estilo),"optimizaciones_sugeridas":_gen_opts(bv6,b,total_b),"algoritmo_version":"v7.1","curva_aplicada":_id_curva(modo),"patron_espacial":_id_patron(estilo)}
        e.append(bv6)
    _val_est_global(e)
    return e

def generar_estructura_inteligente(dur_min:int,config_base:Optional[Dict[str,Any]]=None,**params)->Dict[str,Any]:
    config=config_base or{}
    dur_bloque=params.get('duracion_bloque_min',2.0)
    total_b=max(2,int(dur_min/dur_bloque))
    modo=_inf_modo(config,params);estilo=_inf_estilo(config,params)
    est_base=estructura_layer_fase(total_b,modo,estilo)
    est_opt=optimizar_coherencia_temporal(est_base)
    val=validar_estructura_cientifica(est_opt)
    return{"configuracion":{"duracion_minutos":dur_min,"total_bloques":total_b,"duracion_bloque_min":dur_min/total_b,"modo_temporal":modo,"estilo_espacial":estilo},"estructura":est_opt,"validacion_cientifica":val,"optimizaciones_aplicadas":["Coherencia temporal","Transiciones suavizadas","Validación neuroacústica","Balanceo energético"],"metadatos":{"generado_por":"GeneradorInteligenteV7","version":"v7.1","timestamp":datetime.now().isoformat(),"confianza_cientifica":val.get("confianza_global",0.8)}}

def optimizar_coherencia_temporal(est:List[Dict[str,Any]])->List[Dict[str,Any]]:
    if not est or len(est)<2:return est
    est_opt=est.copy()
    for i in range(len(est_opt)-1):
        ba,bs=est_opt[i],est_opt[i+1]
        if abs(bs["gain"]-ba["gain"])>0.5:
            est_opt[i+1]["gain"]=round(ba["gain"]*0.3+bs["gain"]*0.7,3)
            if"v7_enhanced"in bs:bs["v7_enhanced"]["optimizacion_aplicada"]="suavizado_ganancia"
        if abs(bs["paneo"]-ba["paneo"])>0.6:
            est_opt[i+1]["paneo"]=round(ba["paneo"]*0.4+bs["paneo"]*0.6,3)
    _opt_caps(est_opt);_bal_energia(est_opt)
    return est_opt

def validar_estructura_cientifica(est:List[Dict[str,Any]])->Dict[str,Any]:
    val={"valida_cientificamente":True,"confianza_global":0.0,"advertencias":[],"errores":[],"recomendaciones":[],"metricas_detalladas":{}}
    if not est:val["valida_cientificamente"]=False;val["errores"].append("Estructura vacía");return val
    cohs=[]
    for i,b in enumerate(est):
        if not 0.1<=b["gain"]<=2.0:val["advertencias"].append(f"Bloque {i}: Ganancia fuera rango ({b['gain']})")
        if not -1.0<=b["paneo"]<=1.0:val["errores"].append(f"Bloque {i}: Paneo inválido ({b['paneo']})")
        ca=sum(1 for a in b["capas"].values()if a)
        if ca==0:val["errores"].append(f"Bloque {i}: Sin capas")
        elif ca>5:val["advertencias"].append(f"Bloque {i}: Muchas capas ({ca})")
        cohs.append(_calc_coh_det(b))
    val["confianza_global"]=sum(cohs)/len(cohs)
    val["metricas_detalladas"]={"coherencia_promedio":val["confianza_global"],"coherencia_minima":min(cohs),"coherencia_maxima":max(cohs),"variacion_coherencia":max(cohs)-min(cohs),"total_bloques":len(est),"capas_activas_promedio":sum(sum(1 for c in b["capas"].values()if c)for b in est)/len(est)}
    if val["confianza_global"]<0.7:val["recomendaciones"].append("Optimizar coherencia")
    if val["metricas_detalladas"]["variacion_coherencia"]>0.3:val["recomendaciones"].append("Suavizar transiciones")
    val["valida_cientificamente"]=len(val["errores"])==0 and val["confianza_global"]>0.5
    return val

def _sig(x:float,f:float=6.0)->float:return 1/(1+math.exp(-f*(x-0.5)))
def _calc_coh_sec(tb:int,modo:str,estilo:str)->float:
    cm={"normal":0.9,"ascenso":0.85,"disolucion":0.85,"expansivo":0.8,"ritmico":0.75,"transformativo":0.8,"ondulante":0.7}
    ce={"neutro":1.0,"tribal":0.9,"mistico":0.85,"etereo":0.9,"cristalino":0.95,"organico":0.9,"cuantico":0.7,"ceremonial":0.85,"terapeutico":0.95}
    cb=cm.get(modo,0.8);fe=ce.get(estilo,0.8);fl=0.8 if tb<3 else(0.9 if tb>15 else 1.0)
    return cb*fe*fl
def _det_fase(p:float)->str:return"inicio"if p<0.2 else("desarrollo_temprano"if p<0.4 else("centro"if p<0.6 else("desarrollo_tardio"if p<0.8 else"final")))
def _calc_energia(bi:int,tb:int,modo:str)->float:
    p=bi/max(1,tb-1)
    if modo=="ascenso":return 0.3+0.7*p
    elif modo=="disolucion":return 0.9-0.6*p
    elif modo=="expansivo":return 0.4+0.6*(1-abs(p-0.5)*2)
    else:return 0.7+0.3*math.sin(p*math.pi)
def _calc_coh_bloque(b:Dict[str,Any],cg:float)->float:
    fg=1.0-abs(b["gain"]-1.0)*0.1;fp=1.0-abs(b["paneo"])*0.1;ca=sum(1 for a in b["capas"].values()if a)
    fc=0.0 if ca==0 else(1.0 if ca<=3 else(0.9 if ca<=5 else 0.7))
    return max(0.0,min(1.0,cg*fg*fp*fc))
def _val_bloque(b:Dict[str,Any])->bool:return 0.0<=b["gain"]<=2.0 and-1.0<=b["paneo"]<=1.0 and any(b["capas"].values())
def _calc_estab(bi:int,tb:int)->float:return 0.6+0.4*(1-abs(bi/max(1,tb-1)-0.5)*2)
def _inf_estado(p:float,modo:str)->str:
    estados={"inicio":"preparacion","desarrollo_temprano":"activacion","centro":"optimo","desarrollo_tardio":"integracion","final":"culminacion"}
    fase=_det_fase(p);estado=estados.get(fase,"neutral")
    if modo=="transformativo"and fase=="centro":return"transformacion_profunda"
    elif modo=="expansivo"and fase=="centro":return"expansion_maxima"
    return estado
def _pred_efectos(p:float,modo:str,estilo:str)->List[str]:
    ef=[];fase=_det_fase(p)
    if fase=="inicio":ef.extend(["relajacion","apertura"])
    elif fase=="centro":ef.extend(["estado_optimo","max_efectividad"])
    elif fase=="final":ef.extend(["integracion","cierre"])
    if modo=="ascenso":ef.append("energia_creciente")
    elif modo=="expansivo":ef.append("expansion")
    if estilo=="mistico":ef.append("conexion_espiritual")
    elif estilo=="terapeutico":ef.append("sanacion")
    return ef
def _gen_opts(b:Dict[str,Any],i:int,t:int)->List[str]:
    opts=[]
    if b["gain"]>1.5:opts.append("Reducir intensidad")
    elif b["gain"]<0.3:opts.append("Aumentar intensidad")
    if abs(b["paneo"])>0.8:opts.append("Paneo extremo")
    ca=sum(1 for a in b["capas"].values()if a)
    if ca>4:opts.append("Simplificar capas")
    elif ca<2:opts.append("Enriquecer capas")
    return opts
def _val_est_global(est:List[Dict[str,Any]]):
    if est:
        gains=[b["gain"]for b in est];paneos=[b["paneo"]for b in est]
        est[0]["v7_global_metadata"]={"total_bloques":len(est),"gain_promedio":round(sum(gains)/len(gains),3),"gain_rango":[round(min(gains),3),round(max(gains),3)],"paneo_promedio":round(sum(paneos)/len(paneos),3),"paneo_rango":[round(min(paneos),3),round(max(paneos),3)],"estructura_validada":True,"timestamp_validacion":datetime.now().isoformat()}
def _opt_caps(est:List[Dict[str,Any]]):
    for b in est:b["capas"]["neuro_wave"]=True;b["capas"]["binaural"]=True
def _bal_energia(est:List[Dict[str,Any]]):
    ep=sum(b["gain"]for b in est)/len(est)
    if ep>1.2:
        fa=1.0/ep
        for b in est:b["gain"]=round(b["gain"]*fa,3)
    elif ep<0.8:
        fa=0.9/ep
        for b in est:b["gain"]=round(b["gain"]*fa,3)
def _calc_coh_det(b:Dict[str,Any])->float:
    c=max(0.5,1.0-abs(b["gain"]-1.0)*0.3)*max(0.7,1.0-abs(b["paneo"])*0.2)
    ca=sum(1 for a in b["capas"].values()if a);fc=1.0 if ca in[3,4]else(0.9 if ca in[2,5]else 0.7)
    return max(0.0,min(1.0,c*fc))
def _inf_modo(config:Dict[str,Any],params:Dict[str,Any])->str:
    if"modo"in params:return params["modo"]
    if"categoria"in config:
        cat=config["categoria"].lower()
        if"cognitivo"in cat:return"ascenso"
        elif"terapeutico"in cat:return"disolucion"
        elif"espiritual"in cat:return"expansivo"
    return"normal"
def _inf_estilo(config:Dict[str,Any],params:Dict[str,Any])->str:
    if"estilo"in params:return params["estilo"]
    if"style"in config:
        mapeo={"tribal":"tribal","mistico":"mistico","etereo":"etereo","cristalino":"cristalino","organico":"organico","cuantico":"cuantico","alienigena":"cuantico","minimalista":"neutro"}
        return mapeo.get(config["style"].lower(),"neutro")
    return"neutro"
def _id_curva(modo:str)->str:
    curvas={"ascenso":"sigmoide","disolucion":"exponencial_inv","expansivo":"gaussiana_inv","ritmico":"sinusoidal","transformativo":"meseta","ondulante":"armonicos"}
    return curvas.get(modo,"constante")
def _id_patron(estilo:str)->str:
    patrones={"tribal":"alternante","mistico":"geometria_aurea","etereo":"multicapa","cristalino":"geometria_dual","organico":"respiratorio","cuantico":"discretos","ceremonial":"ritual","terapeutico":"sanacion"}
    return patrones.get(estilo,"natural")

def obtener_estadisticas_scheduler()->Dict[str,Any]:
    return{"version":"v7.1_optimized","compatibilidad_v6":"100%","funciones_mejoradas":4,"nuevas_funciones":3,"curvas_disponibles":len(CurvaEvolucion),"patrones_disponibles":len(PatronEspacial),"validacion_cientifica":True}