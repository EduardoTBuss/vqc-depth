# vqc-depth — profundidade de circuitos quânticos variacionais vs. MLP clássica

Infraestrutura experimental do estudo submetido à **ERAMIA-RS 2026**
(II Escola Regional de Aprendizado de Máquina e Inteligência Artificial do RS).

> **Uma única variável independente: a profundidade do circuito.**
> Tudo o mais — dados, split, encoding, loss, otimizador, épocas, batch,
> inicialização — é mantido idêntico entre os modelos.

Este repositório é **privado** e contém apenas código: nenhum dado baixado,
nenhum resultado e nenhum texto do artigo são versionados aqui.

## Perguntas de pesquisa

| RQ | Pergunta |
|---|---|
| **RQ1** | Como a profundidade do circuito afeta o desempenho preditivo de VQCs? |
| **RQ2** | Como a profundidade afeta estabilidade de treino e convergência entre seeds? |
| **RQ3** | Como VQCs de diferentes profundidades se comparam a uma MLP clássica sob a mesma representação de entrada de baixa dimensão? |

**Hipótese, registrada antes de qualquer execução:** aumentar a profundidade
pode ampliar a capacidade representacional do VQC, mas os ganhos tendem a
**saturar ou regredir** conforme a otimização fica mais difícil.

Isto **não** é uma alegação de vantagem quântica, não é técnica nova e não é
tuning de MLP.

## O grid

3 datasets × 4 modelos × 10 seeds = **120 execuções**.

| Eixo | Valores |
|---|---|
| Datasets | `banknote` · `blood` · `mammographic` |
| Modelos | `vqc1` · `vqc2` · `vqc4` · `mlp` |
| Seeds | `range(10)` |
| Qubits | 4 |
| Encoding | angle, `x_i → RY(x_i)`, aplicado uma vez |
| Ansatz | `RY(θ) → RZ(θ) → bloco CNOT em anel`, repetido 1/2/4× |
| Medição | `⟨Z₀⟩ → p(y=1) = (1 + ⟨Z⟩)/2` |
| Loss / otimizador | BCE / Adam — **idênticos para VQC e MLP** |
| Épocas / batch | 100 / 16 |
| Backend | `default.qubit`, statevector analítico, sem ruído, sem shots |

**Métrica primária: F1.**

## Datasets — 4 features nativas, zero PCA

Nenhuma redução de dimensionalidade: os três datasets já têm (ou são reduzidos
por descarte justificado a) exatamente 4 features, uma por qubit. Isso remove o
PCA como fator de confusão.

| Nome | UCI id | n | Features | Papel no estudo |
|---|---:|---:|---|---|
| `banknote` | 267 | 1372 | variance, skewness, curtosis, entropy | Fácil e limpo. Os modelos provavelmente empatam perto de 99% — serve de piso de sanidade. |
| `blood` | 176 | 748 | Recency, Frequency, Monetary, Time | Difícil e desbalanceado (23,8% positivos). |
| `mammographic` | 161 | 961 | Age, Shape, Margin, Density | Clínico, dificuldade média, com dados faltantes. |

Dois pontos que **precisam ser declarados no artigo**:

- **`blood` tem 4 features nominais mas apenas 3 informativas.** Verificamos
  numericamente que `Monetary = 250 × Frequency` exatamente (igualdade bit a
  bit, `max|M − 250F| = 0`), e que o **posto da matriz de features é 3**. O VQC
  de 4 qubits gasta um qubit codificando uma cópia reescalada de outra feature.
- **`mammographic` descarta o BI-RADS**, que é uma *avaliação* subjetiva do
  radiologista, não uma medida. Mantê-lo seria vazar o rótulo por procuração.
  Tem **160 valores faltantes** (Age 5, Shape 31, Margin 48, Density 76) em 130
  linhas; são imputados pela **mediana do treino**, e a contagem por partição é
  registrada em cada execução.

## ⚠️ Nota crítica sobre o ansatz: parâmetros efetivos < parâmetros nominais

Descoberto e verificado em 25/08/2026, **antes de rodar o grid definitivo**.

No ansatz `RY(θ) → RZ(θ) → anel de CNOT`, a propagação de Heisenberg do
observável pelo anel final dá:

```
U_anel† Z₀ U_anel = Z₁Z₂Z₃
```

Consequência: na **última camada**, os 4 `RZ` comutam com esse observável
efetivo (que é diagonal) e o `RY` do qubit 0 também não o afeta (o observável
age como identidade no fio 0). **Cinco parâmetros da última camada são
inertes**, em qualquer profundidade:

| modelo | nominais | **efetivos** |
|---|---:|---:|
| `vqc1` | 8 | **3** |
| `vqc2` | 16 | **11** |
| `vqc4` | 32 | **27** |

Verificado numericamente por duas vias independentes: perturbar cada parâmetro
em 0,9 rad muda a saída em menos de 1e-12 para exatamente esses cinco, e o
gradiente neles é da ordem de 1e-17.

**Isto não é um bug** — é propriedade do ansatz como especificado. Mas tem duas
consequências que precisam ser tratadas:

1. a tabela de parâmetros **superestima a capacidade** dos modelos;
2. o `vqc1`, com apenas **3 graus de liberdade efetivos**, pode confundir
   "efeito da profundidade" com "camada final degenerada" — o que atinge a RQ1
   diretamente.

Alternativas medidas (parâmetros efetivos, `L=1 / 2 / 4`):

| ansatz | efetivos |
|---|---|
| `RY → RZ → anel` (atual) | 3 / 11 / 27 |
| `RZ → RY → anel` | **6 / 14 / 30** |
| `anel → RY → RZ` | 1 / 7 / 23 |

Há um **teste-sentinela** na suíte que fixa esses números: se o ansatz for
alterado, ele quebra e força uma decisão consciente, em vez de mudar todos os
resultados em silêncio.

## Protocolo de comparação justa

Para cada `(dataset, seed)`:

1. um único split estratificado **70/15/15**;
2. imputador e scaler ajustados **somente no treino**, aplicados a val/test;
3. **os mesmos índices reutilizados pelos 4 modelos**.

O split é função apenas de `(dataset, seed)` — nunca do modelo. Cada execução
grava o **hash SHA-256** dos índices de treino/validação/teste, e há um teste
que falha se dois modelos do mesmo `(dataset, seed)` divergirem, inclusive
entre processos e sessões diferentes.

Isso não é preciosismo: é o que torna o **Wilcoxon pareado por seed**
estatisticamente legítimo.

**A seleção da melhor época é pela validação. O conjunto de teste é tocado uma
única vez, no fim, com os pesos escolhidos pela validação.**

### Por que MinMax para [0, π] e não StandardScaler

Com `RY(x)`, um valor além de ±π **alia**: dois pontos distintos viram o mesmo
estado quântico. `StandardScaler` produz caudas bem além disso. Usamos
`MinMaxScaler` para `[0, π]`, ajustado só no treino.

Val e test podem sair ligeiramente fora da faixa — e **não são clipados**,
porque clipar fingiria que o treino viu um extremo que não viu. Medido em todas
as seeds, o extravasamento é pequeno (pior caso 11,4% de π; 0,04–0,18% dos
valores), muito longe do aliasing, que só começa em 2π.

## Instalação

```bash
git clone https://github.com/EduardoTBuss/vqc-depth.git
cd vqc-depth
pip install -r requirements.txt
pip install -e .
pytest -q
```

Requer Python ≥ 3.10. **CPU basta** — são 4 qubits em `default.qubit`, GPU não
traz ganho nenhum.

## Uso

```bash
python -m vqcdepth --part train|rebuild|plot|tables|all|status --outdir DIR
    [--datasets ...] [--models ...] [--seeds ...] [--epochs N] [--no-checkpoint]
```

A raiz de saída **sempre** vem de `--outdir`: nenhum caminho é fixado no
código, o que é o que permite escrever direto no Google Drive a partir do Colab.

```bash
# o que falta rodar (não treina nada)
python -m vqcdepth --part status --outdir ./out

# piloto rápido
python -m vqcdepth --part train --outdir ./out \
    --datasets banknote --models vqc1 --seeds 0 1 --epochs 10

# grid completo (retomável: reexecute à vontade)
python -m vqcdepth --part train --outdir ./out

# CSVs, tabelas e figuras — funcionam com grid PARCIAL
python -m vqcdepth --part rebuild --outdir ./out
python -m vqcdepth --part tables  --outdir ./out
python -m vqcdepth --part plot    --outdir ./out
```

Particionável por dataset/modelo/seed para dividir o grid entre runtimes.

Para rodar no Colab com persistência no Drive, veja **[COLAB.md](COLAB.md)** e
o launcher em `notebooks/ERAMIA_launcher.ipynb`.

### Token do GitHub para o Colab

O repositório é privado, então o Colab precisa de um token para cloná-lo. Crie
um **fine-grained PAT** em *GitHub → Settings → Developer settings → Personal
access tokens → Fine-grained tokens*:

- **Repository access:** *Only select repositories* → apenas `vqc-depth`
- **Permissions:** *Repository permissions → Contents → **Read-only***
- nada além disso

Guarde-o nos **Colab Secrets** com o nome `GH_PAT` (ícone de chave na barra
lateral) e libere *Notebook access*. **Nunca** digite o token numa célula,
nunca o imprima, nunca o commite.

## Persistência e retomada

- **Um arquivo por execução**, id `dataset__model__seedNN`
  (ex.: `banknote__vqc4__seed07`), escrito atomicamente (`tmp` + `os.replace`).
- Reabrir uma sessão = varrer os arquivos existentes e rodar só o que falta.
- **Nunca um CSV compartilhado sendo apendado**: Drive + dois runtimes = corrupção.
- Os CSVs são **reconstruídos** dos arquivos por execução (`--part rebuild`),
  e o rebuild funciona com grid parcial.
- Retomada **por execução, não por época**. Os checkpoints a cada 5 épocas são
  **artefato do experimento**, não mecanismo de recuperação.

Cada execução salva: métricas finais (schema completo em
`tables.RAW_COLUMNS`), histórico de **todas** as épocas, predições
(`sample_id, y_true, y_pred, y_probability`), TN/FP/FN/TP como números (a
figura de matriz de confusão se recria), contagem de parâmetros, tempo de
treino e de inferência, hashes do split e `env.json` do ambiente.

Como bônus exploratório, `gradient_norm` e `parameter_norm` por época são
registrados. Servem para *observar* se circuitos profundos otimizam pior
(barren plateau) — **não é objetivo do artigo**.

## Estrutura

```
src/vqcdepth/
├── config.py     toda constante do experimento; nenhum outro módulo define hiperparâmetro
├── data.py       carga dos 3 datasets UCI (ids verificados em tempo de carga)
├── splits.py     split 70/15/15 determinístico + imputação/escala sem vazamento + hashes
├── circuits.py   os VQCs em PennyLane; avaliação vetorizada por parameter broadcasting
├── models.py     a MLP baseline em PyTorch
├── train.py      o laço de treino, idêntico em estrutura para VQC e MLP
├── storage.py    escrita atômica, um arquivo por execução, varredura para retomada
├── runner.py     orquestração do grid: o que falta, executa, persiste
├── tables.py     rebuild dos CSVs, agregação, Wilcoxon pareado + correção de Holm
├── figures.py    todas as figuras, em PDF e PNG
└── cli.py        a interface de linha de comando
```

## Custo computacional

**Medido em piloto local (25/08/2026):** 0,370 s/época para `vqc1` e 1,208
s/época para `vqc4` com `n_train=961` e batch 16 (61 passos por época). O grid
completo de 120 execuções leva **~82 min** sequenciais localmente, e algo como
**2 a 2,7 h** no Colab, que é mais lento. Mais de metade do custo é do `vqc4`.

A célula de piloto do notebook remede isso na máquina onde o grid vai rodar.

⚠️ Neste pacote, **uma época é uma varredura completa do conjunto de treino em
minibatches de 16**. No repositório `qnn-benchmark`, "época" significava um
único passo de minibatch. Para o Banknote isso são ~60 passos por época contra
1 — diferença de ordens de grandeza. **Não extrapole tempos entre os dois
projetos.**

## Reprodutibilidade

- Toda aleatoriedade passa por geradores **explícitos**, semeados por
  `(seed, dataset, modelo)`. Não há uso de RNG global — nem `np.random.seed`,
  nem `torch.manual_seed`, nem `random.seed`.
- Não há uso do `hash()` embutido do Python para derivar sementes: ele é
  randomizado por processo (`PYTHONHASHSEED`) e produziria splits diferentes
  entre sessões. Tudo passa por `hashlib`.
- A estabilidade dos splits entre processos é verificada por teste, com
  `PYTHONHASHSEED` diferente em cada um.
- `env.json` grava versões de pacotes e hardware de cada rodada. **É esse
  arquivo, e não o `requirements.txt`, que vai para a seção de reprodutibilidade
  do artigo.**

## Limitações declaradas

Simulador ideal (sem ruído, sem shots), 3 datasets, circuito pequeno (4 qubits),
uma única família de ansatz, MLP baseline sem tuning extensivo, e **nenhuma
alegação de vantagem quântica**.

## Licença

MIT.
