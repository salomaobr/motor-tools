# Changelog

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
