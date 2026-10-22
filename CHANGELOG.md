# Changelog

## 0.8.1

Correcoes da auditoria externa (achados A-D):

- A (vazamento na selecao): `closed_loop_discovery` escolhe candidatos so com dados de desenvolvimento (validacao cruzada); o holdout externo (20% das amostras, minimo 4) fica selado e so pontua o baseline e o candidato final (2 avaliacoes, independente do numero de tentativas); `initial`/`final` reportam o MAE desse holdout. `StructureInducer.fit` restringe selecao e poda de termos ao treino de cada dobra
- B (orcamento de avaliacoes): `CompositionalFrontierSearch` garante que `score_fn` e chamado no maximo `budget` vezes (`explored` e o numero exato de chamadas) e pontua cada candidato uma unica vez
- C (validacao temporal): `StructureInducer` aceita `cv_strategy="time_series"` (walk-forward: cada bloco de teste e estritamente posterior ao treino, inclusive na poda interna); em `closed_loop_discovery` o holdout externo passa a ser o bloco final
- D (semente): `seed` propagada de ponta a ponta (`StructureInducer.fit`, `fit_open` com semente derivada por dobra externa, `ClosedLoopDiscovery.run`, `ExperimentSpec.seed`); mesma semente + mesmos dados = mesmo resultado

Tools:

- `induce_structure` e `closed_loop_discovery` aceitam `seed` (inteiro >= 0, padrao 0) e `cv_strategy` (`"iid"`, padrao, ou `"time_series"`) via MCP, HTTP e `call_tool`; valores invalidos devolvem `supported: false`; a resposta ecoa `seed` e `cv_strategy`. Com os padroes o resultado numerico e identico ao da 0.8.0
- Docstring de `closed_loop_discovery` corrigida: `initial`/`final` trazem o MAE do holdout externo (nao o da validacao cruzada)

Limpeza e testes:

- Removidas todas as mencoes a CAIROS (docstrings e `ResearchLoop.report`, cujo `schema` agora e `motor_tools.research_loop.v1`)
- Testes: 277 (antes 251); novos testes de regressao do achado A (particao disjunta do holdout, holdout nunca usado em `fit`, no maximo 2 avaliacoes, selecao invariante aos rotulos do holdout) e dos novos parametros (Python, HTTP e MCP)
- Badge de testes do README atualizada

## 0.8.0

- Empacotamento limpo: `motor/` e `domains/` agora sao subpacotes (`motor_tools.motor`, `motor_tools.domains`); nada vaza para o namespace do usuario
- `pyproject.toml`: `packages.find` restrito a `motor_tools*`
- Nova tool set (25 no total): induce_structure, closed_loop_discovery, summarize_evidence, propose_hypotheses, run_research_experiment
- `call_tool` fail-soft: nunca levanta excecao, devolve `{"supported": false, "error": ...}`
- Removido codigo sem uso: research_copilot/autopilot/archive/memory/adaptive, active_discovery, experiment_planner, hypothesis_discriminator, evidence_state e os dominios billiard, point2d, world3, finance
- Corrigido import dinamico em agi_active_recovery
- Testes: 249 (antes 33); novos testes de fail-soft, HTTP e MCP
- README reescrito

## 0.7.0

- Reintegrado domains/ (AlienPhysics e ambientes de dominio)
- Novas tools AGI: agi_robustness_battery, agi_active_recovery_battery, agi_generalization_battery, regime_transfer_battery
- Nova tool observability_check
- Modulos research/AGI restaurados como suporte das baterias

## 0.6.0

- Pacote minimo so com tools estatisticas/monitoramento

## 0.5.0

- Gates, atribuicao, incerteza, canonicalize

## 0.4.0 - 0.1.0

- Ver historico anterior
