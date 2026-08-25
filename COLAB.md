# Rodando o grid no Google Colab (tudo persistido no Drive)

Tudo é escrito no Google Drive, sob `MyDrive/ERAMIA-VQC/`:

```
MyDrive/ERAMIA-VQC/
├── runs/          UM JSON por execução + o histórico de épocas dela (o checkpoint real)
├── results/       os CSVs reconstruídos (raw_results, training_history, summary)
├── predictions/   um parquet por execução: sample_id, y_true, y_pred, y_probability
├── checkpoints/   pesos a cada 5 épocas + best (artefato do experimento)
├── figures/       PDF (para o LaTeX) e PNG (para olhar)
├── tables/        CSV + Markdown + LaTeX
├── logs/          stdout de cada execução
└── env.json       versões e hardware da rodada
```

## Como a retomada funciona

Cada execução escreve **o seu próprio arquivo**, de forma atômica
(`tmp` + `os.replace`). Uma execução que já tem arquivo é pulada. Se a sessão
morrer, reconecte e rode a mesma célula: ela continua exatamente de onde parou,
perdendo no máximo a execução em voo.

**Nunca existe um CSV compartilhado sendo apendado.** Drive + dois runtimes
escrevendo no mesmo CSV é receita de corrupção silenciosa. Os CSVs são
*reconstruídos* a partir dos arquivos por execução, com `--part rebuild`, e o
rebuild funciona com grid parcial.

A retomada é **por execução, não por época**. As execuções são curtas o
bastante para que resume intra-execução seja over-engineering. Os checkpoints
por época existem como **artefato do experimento** (curvas de convergência,
observação de norma de gradiente), não como mecanismo de recuperação.

## Custo

**Medido em piloto local (25/08/2026)**, não estimado:

| modelo | s/época (n_train=961) | 1 run de 100 épocas | 30 runs |
|---|---:|---:|---:|
| `vqc1` | 0,370 | 0,6 min | 13,8 min |
| `vqc2` | 0,579 | 1,0 min | 21,6 min |
| `vqc4` | 1,208 | 2,0 min | 45,2 min |
| `mlp`  | 0,034 | 0,06 min | 1,3 min |

**Grid completo (120 execuções, sequencial): ~82 min** numa máquina local
razoável. O Colab costuma ser 1,5–2× mais lento, então espere **~2 a 2,7 h**.
O custo é dominado pelo `vqc4` (55% do total).

`vqc1` e `vqc4` foram medidos por mim no piloto; `vqc2` e `mlp` vêm da
instrumentação do desenvolvimento e **não foram reconfirmados no piloto** —
rode a célula 5 com `--models vqc2 mlp` se quiser fechar a conta.

⚠️ A célula 5 do notebook (piloto) **mede** os segundos por época e extrapola.
Rode-a antes de qualquer planejamento. Não confie em nenhuma estimativa que não
tenha saído dessa célula, e em particular **não extrapole do repositório
`qnn-benchmark`**: lá, "época" era um único passo de minibatch; aqui, época é a
varredura completa do treino em minibatches de 16. Para o Banknote isso são
~60 passos por época contra 1 — diferença de ordens de grandeza.

**Não peça runtime com GPU.** São 4 qubits em `default.qubit`; a GPU não seria
tocada. Runtime de CPU é a escolha certa.

## O notebook

[`notebooks/ERAMIA_launcher.ipynb`](notebooks/ERAMIA_launcher.ipynb) é um
launcher **fino**: toda a lógica vive no pacote `vqcdepth`, versionado aqui. O
notebook só monta o Drive, clona, instala, roda o gate de testes, mede o piloto
e dispara o grid.

Mantenha uma cópia dele no seu Drive e abra por ali (*File → Open notebook →
Google Drive*). Isso evita ter que autorizar o Colab a acessar o GitHub a cada
sessão.

---

## Célula 1 — Drive

Monta o Drive e cria a árvore de saída.

## Célula 2 — repositório privado, via Colab Secrets

O repo é **privado**, então o clone precisa de um token.

**Antes de rodar**, crie o segredo no Colab (ícone de chave na barra lateral):

- Nome: **`GH_PAT`**
- Valor: um **fine-grained PAT** do GitHub, **restrito a este repositório**,
  com **apenas `Contents: Read-only`**. Nada além disso — esse token não
  precisa poder escrever, nem ver outros repos.
- Ative *Notebook access* para este notebook.

O token **nunca** é digitado numa célula, **nunca** é impresso e **nunca** é
commitado. A célula ainda faz duas coisas para não vazá-lo:

1. captura a saída do `git clone` e **não imprime a mensagem de erro crua** (o
   git ecoa a URL, que contém o token, quando a autenticação falha);
2. reescreve o `remote` logo depois do clone, para o token não ficar gravado
   em `.git/config` dentro da máquina do Colab.

Se o clone falhar, a mensagem é genérica de propósito: confira o PAT, o escopo
`Contents: Read` e o nome do repositório.

## Célula 3 — dependências

`pip install -r requirements.txt` e `pip install -e .`.

## Célula 4 — gate: a suíte de testes

```python
!VQCDEPTH_ALLOW_DOWNLOAD=1 python -m pytest tests/ -q
```

A variável `VQCDEPTH_ALLOW_DOWNLOAD=1` **importa**: sem ela, os 18 testes que
dependem dos datasets da UCI são *pulados* num clone novo (que não tem cache
local), e o gate ficaria mais fraco do que parece — justamente a carga dos três
datasets e os hashes golden dos splits reais. Com a variável, a suíte roda
completa: **177 testes, ~55 s** (verificado num clone limpo).

**Se isto falhar, não rode o grid.** É aqui que se prova que a versão
vetorizada do circuito dá o mesmo valor *e o mesmo gradiente* do laço por
amostra, que os 4 modelos do mesmo `(dataset, seed)` recebem literalmente os
mesmos índices, e que o conjunto de teste não vazou para o pré-processamento.

## Célula 5 — piloto (**a célula que mede o tempo**)

1 dataset × 1 modelo × 2 seeds × 10 épocas.

Confirma que os arquivos aparecem no Drive **e mede segundos por época**. A
célula seguinte lê os arquivos recém-gravados e extrapola — o número sai do que
foi medido, não de um chute.

Rode também com `--models vqc4` para pegar o pior caso: o custo cresce com a
profundidade.

O piloto usa `--epochs 10`, então **essas execuções não são grau-de-paper**.
A terceira célula do piloto as apaga (execuções, predições e checkpoints) antes
do grid real — se ficarem, o resume as considera prontas e o grid final teria
duas células treinadas com 10 épocas em vez de 100.

## Célula 6 — grid completo

```python
!python -m vqcdepth --part train --outdir "{DRIVE}" 2>&1 | tee "{LOG}"
```

Retomável: pode reexecutar à vontade.

Para dividir entre dois runtimes (cada execução é um arquivo, então não há
corrida):

```python
# runtime A
!python -m vqcdepth --part train --outdir "{DRIVE}" --datasets banknote
# runtime B
!python -m vqcdepth --part train --outdir "{DRIVE}" --datasets blood mammographic
```

Só vale a pena se o piloto mostrar que o grid é longo. O Colab gratuito
raramente dá dois backends ao mesmo tempo.

## Célula 7 — CSVs, tabelas e figuras

```python
!python -m vqcdepth --part rebuild --outdir "{DRIVE}"
!python -m vqcdepth --part tables  --outdir "{DRIVE}"
!python -m vqcdepth --part plot    --outdir "{DRIVE}"
```

Funciona com **grid parcial**. Não espere as 120 execuções para começar a
olhar os dados.

`tables/wilcoxon_depth.*` é o que responde à RQ1 (vqc1 vs vqc2 vs vqc4);
`tables/wilcoxon_vs_baseline.*` compara cada modelo com a MLP. Em ambos, a
coluna que vai para o paper é **`p_holm`**, não `p_value` — `p_holm` corrige as
múltiplas comparações dentro de cada dataset.

## Verificar o andamento a qualquer momento

```python
!python -m vqcdepth --part status --outdir "{DRIVE}"
```

Diz quantas execuções já estão no disco, quantas faltam e quais são as
próximas. Não treina nada.
