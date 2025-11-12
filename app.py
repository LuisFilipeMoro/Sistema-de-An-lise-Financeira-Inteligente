import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página

st.set_page_config(page_title="Sistema de Análise Financeira Inteligente", page_icon="💹", layout="wide")

st.title("💹 Sistema de Análise Financeira Inteligente")
st.markdown("### – Gestão e Inovação Digital ")
st.write("Envie um arquivo CSV com dados financeiros ou preços de ativos para realizar a análise automática.")

# Upload do arquivo

arquivo = st.file_uploader("📁 Envie seu arquivo CSV", type=["csv"])

if arquivo is not None:
    # Leitura dos dados
    df = pd.read_csv(arquivo)
    st.subheader("📊 Visualização dos Dados")
    st.write(df.head())

    # Verifica se há pelo menos 2 colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(colunas_numericas) < 2:
        st.error("O arquivo deve conter pelo menos duas colunas numéricas (ex: preços ou valores financeiros).")
    else:
        # Cálculo dos retornos logarítmicos

        precos = df[colunas_numericas].values
        retornos = np.log(precos[1:] / precos[:-1])
        nomes = colunas_numericas

        # Retorno e volatilidade
        retorno_medio_diario = np.mean(retornos, axis=0)
        volatilidade_diaria = np.std(retornos, axis=0)
        retorno_anual = retorno_medio_diario * 252
        volatilidade_anual = volatilidade_diaria * np.sqrt(252)

        # Exibir métricas
        st.subheader("📈 Métricas Financeiras (Anualizadas)")
        resultados = pd.DataFrame({
            "Ativo": nomes,
            "Retorno Anual (%)": retorno_anual * 100,
            "Volatilidade Anual (%)": volatilidade_anual * 100
        })
        st.dataframe(resultados.style.format({"Retorno Anual (%)": "{:.2f}", "Volatilidade Anual (%)": "{:.2f}"}))

        # Matriz de Covariância e Correlação
        matriz_cov = np.cov(retornos, rowvar=False)
        matriz_corr = np.corrcoef(retornos, rowvar=False)

        st.subheader("🧮 Matriz de Covariância")
        st.dataframe(pd.DataFrame(matriz_cov, index=nomes, columns=nomes).style.format("{:.6f}"))

        st.subheader("🔗 Matriz de Correlação")
        st.dataframe(pd.DataFrame(matriz_corr, index=nomes, columns=nomes).style.format("{:.3f}"))

       
        # Simulação de Monte Carlo (simplificada)
        st.subheader("🎲 Simulação de Monte Carlo (Portfólio Equilibrado)")
        pesos = np.array([1 / len(nomes)] * len(nomes))
        num_simulacoes = 5000

        retorno_medio = np.mean(retornos, axis=0)
        matriz_cholesky = np.linalg.cholesky(matriz_cov)

        simulacoes = []
        for _ in range(num_simulacoes):
            retornos_aleatorios = np.random.normal(0, 1, size=(252, len(nomes)))
            retornos_correlacionados = retorno_medio + retornos_aleatorios @ matriz_cholesky.T
            retorno_port = np.sum((retornos_correlacionados @ pesos))
            simulacoes.append(retorno_port)

        simulacoes = np.array(simulacoes)
        var_95 = np.percentile(simulacoes, 5)

        st.metric("Value at Risk (95%)", f"{var_95 * 100:.2f}%")
        st.metric("Média dos Retornos Simulados", f"{np.mean(simulacoes) * 100:.2f}%")

        # Gráfico de distribuição dos retornos simulados
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(simulacoes, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'VaR 95%: {var_95*100:.2f}%')
        ax.set_title("Distribuição dos Retornos do Portfólio")
        ax.legend()
        st.pyplot(fig)

        # Recomendações simples
        st.subheader("💡 Recomendações Automáticas")
        melhor_ativo = nomes[np.argmax(retorno_anual)]
        mais_arriscado = nomes[np.argmax(volatilidade_anual)]

        st.success(f"O ativo com **melhor desempenho** é **{melhor_ativo}** com retorno anual de {retorno_anual[np.argmax(retorno_anual)]*100:.2f}%.")
        st.warning(f"O ativo **mais arriscado** é **{mais_arriscado}** com volatilidade anual de {volatilidade_anual[np.argmax(volatilidade_anual)]*100:.2f}%.")

else:
    st.info("Envie um arquivo CSV para iniciar a análise.")
