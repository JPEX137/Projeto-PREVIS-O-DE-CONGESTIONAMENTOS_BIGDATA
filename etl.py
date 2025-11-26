import pandas as pd
import geopandas as gpd
import json
import os
from pyproj import CRS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuração de avisos para limpar o terminal
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("INICIANDO PROCESSAMENTO DE DADOS (ETL) - FORTALEZA")
print("="*60)

# ===============================================
# 1. CARREGAMENTO DOS DADOS
# ===============================================
try:
    # CSVs e Excel
    csv_demandas = pd.read_csv("demandas-marco-abril-2022.csv", sep=",", encoding="utf-8")
    csv_ocorrencias = pd.read_csv("ocorrencias_por_regional.csv", sep=",", encoding="utf-8")
    csv_idh = pd.read_excel("indicededesenvolvimentohumano.xlsx")
    
    # GeoJSONs
    regionais_novas = gpd.read_file("static/data/Secretarias_Executivas_Regionais.geojson")
    bairros = gpd.read_file("static/data/Bairros_de_Fortaleza.geojson")
    
    # Camadas de Infraestrutura
    canais = gpd.read_file("static/data/Canais.geojson")
    valas_drenos = gpd.read_file("static/data/Valas_e_Drenos.geojson")
    
    # Camadas de Risco (Verifique se estes arquivos existem na pasta)
    # Se algum não existir, o código vai avisar, mas tentará seguir
    risco_geo = gpd.read_file("static/data/risco_geologico.json")
    inundacao_geo = gpd.read_file("static/data/inundacao_raster_corrected.geojson") 
    
    print("[OK] Arquivos carregados com sucesso.")
    
except FileNotFoundError as e:
    print(f"ERRO CRÍTICO: Arquivo não encontrado: {e.filename}")
    exit()
except Exception as e:
    print(f"ERRO CRÍTICO: {e}")
    exit()

# Definir CRS Métrico para cálculos de área (SIRGAS 2000 / UTM zone 24S)
crs_metrico = CRS.from_epsg(31984)

# ===============================================
# 2. PROCESSAMENTO: LIXO (Fator Agravante)
# ===============================================
print(">>> Processando Demandas de Lixo...")
csv_demandas.columns = csv_demandas.columns.str.strip().str.upper()
# Filtrar Fortaleza e Regionais válidas
csv_fortaleza = csv_demandas[
    (csv_demandas['CIDADE'].str.upper().str.strip() == 'FORTALEZA') & 
    (csv_demandas['ZONA'].str.contains('SER', na=False))
].copy()

LIXO_KEYWORDS = [
    "COLETA DE ENTULHO", "COLETA DE PODA", "COLETA MECANIZADA", 
    "PONTO DE LIXO", "VOLUMOSO", "LIXO"
]
csv_fortaleza['IS_LIXO'] = csv_fortaleza['TIPO DA DEMANDA'].str.upper().str.contains('|'.join(LIXO_KEYWORDS), na=False)

agg_lixo = csv_fortaleza[csv_fortaleza['IS_LIXO']].groupby('ZONA').size().reset_index(name='lixo_count')
agg_lixo.rename(columns={'ZONA': 'regiao_adm'}, inplace=True)

# ===============================================
# 3. PROCESSAMENTO: ALAGAMENTO (Target/Histórico)
# ===============================================
print(">>> Processando Histórico de Alagamentos...")
ALAGAMENTO_KEYWORDS = ["Inundação", "Alagamento"]
df_alagamento = csv_ocorrencias[
    csv_ocorrencias['Tipologia de Ocorrência'].isin(ALAGAMENTO_KEYWORDS)
].copy()

# Limpeza do nome da regional para bater com o GeoJSON (Ex: "SR 1" -> "SER 1")
df_alagamento['Regional'] = df_alagamento['Regional'].str.replace('SR ', 'SER ', regex=False)
df_alagamento = df_alagamento[df_alagamento['Regional'] != 'TODAS']

agg_alagamento = df_alagamento.groupby('Regional')['Ocorrências'].sum().reset_index()
agg_alagamento.rename(columns={'Regional': 'regiao_adm', 'Ocorrências': 'alagamento_count'}, inplace=True)

# ===============================================
# 4. PROCESSAMENTO: IDH (Vulnerabilidade Social)
# ===============================================
print(">>> Processando IDH por Regional...")
# Limpeza e Conversão
df_idh = csv_idh[['Bairros', 'IDH']].copy()
df_idh['IDH'] = pd.to_numeric(df_idh['IDH'].astype(str).str.replace(',', '.'), errors='coerce')
df_idh.dropna(subset=['IDH'], inplace=True)

# Spatial Join: Bairro -> Regional
bairros = bairros.to_crs(regionais_novas.crs)
# Usamos o centróide do bairro para evitar duplicidade de fronteira
bairros['centroide'] = bairros.geometry.centroid
bairros_geo = bairros.set_geometry('centroide')

join_bairro_regional = gpd.sjoin(bairros_geo, regionais_novas[['regiao_adm', 'geometry']], how='inner', predicate='within')
mapa_bairro = join_bairro_regional[['Nome', 'regiao_adm']].drop_duplicates()

idh_regional = df_idh.merge(mapa_bairro, left_on='Bairros', right_on='Nome', how='inner')
agg_idh = idh_regional.groupby('regiao_adm')['IDH'].mean().reset_index()
agg_idh.rename(columns={'IDH': 'idh_medio'}, inplace=True)

# ===============================================
# 5. PROCESSAMENTO AVANÇADO: DENSIDADE DE RISCO
# ===============================================
print(">>> Calculando Densidade de Risco Geológico e Inundação...")

# Converter tudo para métrico para calcular áreas
regionais_proj = regionais_novas.to_crs(crs_metrico)
regionais_proj['area_total_km2'] = regionais_proj.geometry.area / 1_000_000

# Função para calcular % de área coberta por uma camada de risco
def calcular_densidade_risco(gdf_regionais, gdf_risco, nome_coluna_saida):
    if gdf_risco is None or gdf_risco.empty:
        return pd.DataFrame({'regiao_adm': gdf_regionais['regiao_adm'], nome_coluna_saida: 0.0})
    
    gdf_risco = gdf_risco.to_crs(crs_metrico)
    
    # A mágica acontece aqui: Overlay (Interseção geométrica real)
    # Isso corta o mapa de risco exatamente no formato da regional
    overlay = gpd.overlay(gdf_regionais, gdf_risco, how='intersection')
    
    # Calcular a área dos pedaços de risco dentro da regional
    overlay['area_risco'] = overlay.geometry.area
    
    # Somar área de risco por regional
    soma_risco = overlay.groupby('regiao_adm')['area_risco'].sum().reset_index()
    
    # Juntar com a área total da regional para pegar a %
    resultado = gdf_regionais[['regiao_adm', 'geometry']].merge(soma_risco, on='regiao_adm', how='left').fillna(0)
    resultado[nome_coluna_saida] = (resultado['area_risco'] / resultado.geometry.area) * 100
    
    return resultado[['regiao_adm', nome_coluna_saida]]

# Calcular Risco Geológico
df_densidade_geo = calcular_densidade_risco(regionais_proj, risco_geo, 'perc_risco_geo')

# Calcular Risco Inundação
df_densidade_inund = calcular_densidade_risco(regionais_proj, inundacao_geo, 'perc_risco_inundacao')

# ===============================================
# 6. PROCESSAMENTO: INFRAESTRUTURA (Drenagem)
# ===============================================
print(">>> Calculando Densidade de Drenagem...")
canais_proj = canais.to_crs(crs_metrico)
valas_proj = valas_drenos.to_crs(crs_metrico)

# Clipar linhas dentro das regionais
canais_clip = gpd.overlay(canais_proj, regionais_proj, how='intersection')
valas_clip = gpd.overlay(valas_proj, regionais_proj, how='intersection')

# Calcular comprimentos
canais_clip['comprimento'] = canais_clip.geometry.length
valas_clip['comprimento'] = valas_clip.geometry.length

infra_total = pd.concat([
    canais_clip.groupby('regiao_adm')['comprimento'].sum(),
    valas_clip.groupby('regiao_adm')['comprimento'].sum()
]).groupby('regiao_adm').sum().reset_index()

# Calcular densidade (Km de drenagem por Km² de área) -> Mais justo que valor absoluto
agg_infra = regionais_proj[['regiao_adm', 'area_total_km2']].merge(infra_total, on='regiao_adm', how='left').fillna(0)
agg_infra['densidade_drenagem'] = (agg_infra['comprimento'] / 1000) / agg_infra['area_total_km2']

# ===============================================
# 7. CONSOLIDAÇÃO E CÁLCULO DO ÍNDICE
# ===============================================
print(">>> Consolidando Master Dataframe...")

df_master = regionais_novas[['regiao_adm']].copy()
df_master = df_master.merge(agg_lixo, on='regiao_adm', how='left').fillna(0)
df_master = df_master.merge(agg_alagamento, on='regiao_adm', how='left').fillna(0)
df_master = df_master.merge(agg_idh, on='regiao_adm', how='left')
# Preencher IDH faltante com a média (para não quebrar o cálculo)
df_master['idh_medio'] = df_master['idh_medio'].fillna(df_master['idh_medio'].mean())

df_master = df_master.merge(df_densidade_geo, on='regiao_adm', how='left').fillna(0)
df_master = df_master.merge(df_densidade_inund, on='regiao_adm', how='left').fillna(0)
df_master = df_master.merge(agg_infra[['regiao_adm', 'densidade_drenagem']], on='regiao_adm', how='left').fillna(0)

# --- NORMALIZAÇÃO (MinMax) ---
# É crucial normalizar para que "1000 demandas de lixo" não engulam "0.5 de IDH"
scaler = MinMaxScaler()
cols_to_norm = ['lixo_count', 'idh_medio', 'perc_risco_geo', 'perc_risco_inundacao', 'densidade_drenagem']
df_norm = pd.DataFrame(scaler.fit_transform(df_master[cols_to_norm]), columns=[c+'_norm' for c in cols_to_norm])
df_final = pd.concat([df_master, df_norm], axis=1)

# --- CÁLCULO DO SCORE ---
# Fórmula: 
# (+) Risco Geo (30%) 
# (+) Risco Inundação (30%) 
# (+) Lixo (10%) 
# (-) IDH (15%) -> Invertido (1 - IDH) pois maior IDH reduz risco
# (-) Drenagem (15%) -> Invertido pois mais drenagem reduz risco

print(">>> Calculando Score Ponderado...")
df_final['score_risco'] = (
    (df_final['perc_risco_geo_norm'] * 0.30) +
    (df_final['perc_risco_inundacao_norm'] * 0.30) +
    (df_final['lixo_count_norm'] * 0.10) +
    ((1 - df_final['idh_medio_norm']) * 0.15) +
    ((1 - df_final['densidade_drenagem_norm']) * 0.15)
)

# Classificação por Quantis
df_final['cluster_risco'] = pd.qcut(df_final['score_risco'], 3, labels=['Baixo', 'Médio', 'Alto'])

# ===============================================
# 8. GERAÇÃO DE PREDIÇÕES (Predict Final)
# ===============================================
def gerar_predicao(row):
    """Gera um texto preditivo baseado nos dados da linha"""
    nivel = row['cluster_risco']
    risco_geo = row['perc_risco_geo']
    lixo = row['lixo_count']
    drenagem = row['densidade_drenagem']
    
    if nivel == 'Alto':
        if risco_geo > 10:
            return "ALERTA CRÍTICO: Alta probabilidade de recorrência devido à topografia desfavorável. Intervenções estruturais (obras) são urgentes."
        elif lixo > 500 and drenagem < 2:
            return "ALERTA OPERACIONAL: Risco elevado por obstrução. O sistema de drenagem é insuficiente e o acúmulo de lixo agrava o cenário. Priorizar limpeza."
        else:
            return "ALERTA GERAL: Combinação de múltiplos fatores indica vulnerabilidade severa a eventos extremos."
    elif nivel == 'Médio':
        if drenagem < 1:
            return "ATENÇÃO: Déficit de infraestrutura detectado. Se as chuvas intensificarem, a capacidade de escoamento será superada rapidamente."
        else:
            return "MONITORAMENTO: Áreas pontuais de risco. Manter limpeza preventiva para evitar agravamento."
    else:
        return "ESTÁVEL: Baixa probabilidade de grandes desastres, mas requer manutenção da infraestrutura existente."

df_final['predicao_texto'] = df_final.apply(gerar_predicao, axis=1)

# ===============================================
# 9. EXPORTAÇÃO
# ===============================================

# Preparar JSON para o Front-end
export_cols = {
    'regiao_adm': 'regiao',
    'lixo_count': 'lixo',
    'alagamento_count': 'alagamento',
    'idh_medio': 'idh',
    'perc_risco_geo': 'risco_geo',
    'perc_risco_inundacao': 'risco_inundacao',
    'densidade_drenagem': 'densidade_drenagem',
    'cluster_risco': 'cluster',
    'predicao_texto': 'predict'
}

df_export = df_final.rename(columns=export_cols)[export_cols.values()]
# Arredondar valores para ficar bonito no HTML
df_export['idh'] = df_export['idh'].round(3)
df_export['risco_geo'] = df_export['risco_geo'].round(2)
df_export['risco_inundacao'] = df_export['risco_inundacao'].round(2)
df_export['densidade_drenagem'] = df_export['densidade_drenagem'].round(2)

# Salvar JSON
if not os.path.exists('static/data'):
    os.makedirs('static/data')
    
df_export.to_json('static/data/result.json', orient='records', force_ascii=False, indent=2)
print(f"[OK] JSON gerado: static/data/result.json")

# Atualizar GeoJSON das Regionais com os dados calculados (para o mapa choropleth)
regionais_com_dados = regionais_novas.merge(df_export, left_on='regiao_adm', right_on='regiao', how='left')
regionais_com_dados.to_file("static/data/Secretarias_Executivas_Regionais.geojson", driver='GeoJSON')
print(f"[OK] GeoJSON atualizado com métricas.")

# ===============================================
# 10. GERAÇÃO DE GRÁFICOS (Matplotlib)
# ===============================================
print(">>> Gerando Gráficos...")
output_path = 'static/images/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Ordenar para gráficos
df_g = df_export.set_index('regiao')

# Gráfico 1: Cluster (Resultado Final)
plt.figure(figsize=(10, 6))
colors = df_g['cluster'].map({'Alto': '#d73027', 'Médio': '#fee08b', 'Baixo': '#4575b4'})
df_g['alagamento'].sort_values().plot(kind='barh', color=colors)
plt.title('Validação: Ocorrências Reais vs. Modelo de Risco (Cor)')
plt.xlabel('Ocorrências de Alagamento (2022)')
plt.tight_layout()
plt.savefig(output_path + 'grafico_resultado_cluster.png')
plt.close()

# Gráfico 2: Correlação Risco Geo vs Alagamento
plt.figure(figsize=(8, 6))
plt.scatter(df_g['risco_geo'], df_g['alagamento'], color='purple', s=100, alpha=0.7)
# Linha de tendência
z = np.polyfit(df_g['risco_geo'], df_g['alagamento'], 1)
p = np.poly1d(z)
plt.plot(df_g['risco_geo'], p(df_g['risco_geo']), "r--")
plt.title('Correlação: % Área de Risco Geológico vs. Alagamentos')
plt.xlabel('% da Regional em Área de Risco Geológico')
plt.ylabel('Ocorrências de Alagamento')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path + 'grafico_correlacao_risco_geo.png')
plt.close()

# Demais gráficos básicos (aproveitando lógica anterior)
# IDH
plt.figure(figsize=(8, 6))
plt.scatter(df_g['idh'], df_g['alagamento'], color='green', s=100, alpha=0.7)
z = np.polyfit(df_g['idh'], df_g['alagamento'], 1)
p = np.poly1d(z)
plt.plot(df_g['idh'], p(df_g['idh']), "r--")
plt.title('Correlação: IDH vs. Alagamentos')
plt.xlabel('IDH')
plt.ylabel('Ocorrências')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path + 'grafico_correlacao_idh.png')
plt.close()

# Drenagem
plt.figure(figsize=(8, 6))
plt.scatter(df_g['densidade_drenagem'], df_g['alagamento'], color='blue', s=100, alpha=0.7)
z = np.polyfit(df_g['densidade_drenagem'], df_g['alagamento'], 1)
p = np.poly1d(z)
plt.plot(df_g['densidade_drenagem'], p(df_g['densidade_drenagem']), "r--")
plt.title('Correlação: Densidade de Drenagem vs. Alagamentos')
plt.xlabel('Km de Rede por Km²')
plt.ylabel('Ocorrências')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path + 'grafico_correlacao_drenagem.png')
plt.close()

print("PROCESSAMENTO CONCLUÍDO COM SUCESSO.")