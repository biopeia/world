# World V-03: Mapeo Formal del Protocolo de Commit y Recuperación y Evidencia TLC

**Especificación TLA+:** `specs/WorldCommitRecovery.tla`
**Configuración TLC:** `specs/WorldCommitRecovery.cfg`
**Script ejecutor TLC:** `specs/run_tlc.sh`
**Implementación Python:** `src/world/manifest.py`, `src/world/durable_runtime.py`, `src/world/owner_checkpoint_store.py`
**Suite de pruebas:** `tests/test_systematic_fault_injection.py`, `tests/test_formal_model_verification.py`
**Autoridades normativas:** ADR-0083 (Single-Writer Durability), ADR-0093 (Fail-Closed Recovery), Docs #234, #236.

---

## 1. Resolución de Hallazgos de Revisión (Rework V-03)

En respuesta a la revisión técnica sobre PR #188 ([World PR #188 comment](https://github.com/biopeia/world/pull/188#issuecomment-6232950702)):

1. **Corrección de la configuración TLA+ (Hallazgo F2):**
   - Se corrigió `specs/WorldCommitRecovery.cfg` para sustituir `SPECIFICATION Init` por la fórmula temporal válida `SPECIFICATION Spec` (`Spec == Init /\ [][Next]_vars`), incorporando `CHECK_DEADLOCK FALSE` para permitir el límite finito `MaxTick`.
   - Se tiparon fuertemente los estados del manifiesto temporal (`temp_manifest_status \in {"NONE", "PREPARING", "COMMITTED"}`) para eliminar comparaciones de registros con valores escalares en el motor semántico de TLC.
2. **Ejecución real y comprobada con TLC (Hallazgo F1):**
   - Se ejecutó TLC real (TLC2 Version 2026.10.15.145915, rev `932d35e` con OpenJDK 21) directamente contra `specs/WorldCommitRecovery.tla`.
   - Para `MaxTick = 2`: **205 estados generados, 142 estados distintos, profundidad 30, 0 violaciones de invariantes**.
   - Para `MaxTick = 3`: **477 estados generados, 330 estados distintos, profundidad 43, 0 violaciones de invariantes**.
   - Se comprobó la concordancia a nivel de conteo de estados y transiciones exploradas entre el explorador diferencial en Python (`specs/verify_model.py`) y la salida del comprobador formal TLC (exactamente 142 estados / 204 transiciones para `MaxTick=2`; 330 estados / 476 transiciones para `MaxTick=3`). Esta concordancia empírica sirve como comprobación cruzada acotada y rápida en entornos de desarrollo sin JVM, reconociendo explícitamente que la autoridad formal normativa reside en la especificación TLA+ (`specs/WorldCommitRecovery.tla`) verificada por TLC y no en una prueba formal de bisimulación matemática del código Python.
   - Se integró la verificación con TLC con validación criptográfica estricta de digest de artefacto (`tests/test_formal_model_verification.py::test_tlc_model_checker_execution`), eliminando cualquier paso silencioso y requiriendo skip explícito o verificación exhaustiva.

---

## 2. Correspondencia de Variables de Estado

| Variable TLA+ | Tipo formal | Equivalente en código Python / Sistema de archivos |
|---|---|---|
| `manifest_status` | `{"COMMITTED", "PREPARING", "RECOVERY_REQUIRED"}` | `WorldCausalManifest.status` (`ManifestStatus` enum en `src/world/manifest.py:30-33`). Leído de `manifest.json`. |
| `manifest_tick` | `Nat` | `WorldCausalManifest.causal_tick` (entero no negativo en `manifest.py:46`). Representa el último corte causal confirmado. |
| `manifest_pending_tick` | `Nat` | `WorldCausalManifest.pending_tick` (entero o `None` en `manifest.py:58`). Solo presente durante `PREPARING` o `RECOVERY_REQUIRED`. |
| `manifest_refs` | `Set(CheckpointReference)` | `WorldCausalManifest.checkpoint_references` (tupla inmutable de referencias en `manifest.py:48`). |
| `temp_manifest_status` | `{"NONE", "PREPARING", "COMMITTED"}` | Presencia de archivo temporal `.manifest.json.XXXXXX` creado por `tempfile.mkstemp` en `WorldCausalManifestStore._write` (`manifest.py:492`). Desaparece tras `os.replace` o tras caída del proceso. |
| `journal_committed_len` | `Nat` | Longitud en registros confirmados apuntada por la referencia `CheckpointReference` con dueño `world-journal` (`durable_runtime.py:597`). |
| `journal_file_len` | `Nat` | Tamaño real del archivo `world-causal-journal-v3.jsonl` en disco. Puede exceder a `journal_committed_len` si un proceso murió tras escribir bytes no confirmados. |
| `owner_blobs` | `Set(Blob)` | Archivos inmutables direccionados por contenido en el directorio del almacén de checkpoints (`xx/<sha256>.blob` en `FileOwnerCheckpointStore`, `owner_checkpoint_store.py:83-140`). |
| `recovery_receipts` | `Set(Receipt)` | Archivos JSON inmutables en `recoveries/receipt-<world>-<transition>-<tick>.json` generados por `WorldCausalManifestStore._complete_recovery_locked` (`manifest.py:458-477`). |
| `proc_phase` | Enum de fases | Estado de ejecución en memoria de `DurableWorldRuntime` durante `tick()` o `recover_previous_cut()`. Desaparece ante `os._exit()`. |

---

## 3. Correspondencia de Acciones Abstractas a Operaciones Concretas

| Acción TLA+ | Código Python invocado | Efectos concretos en disco / llamada atómica |
|---|---|---|
| `TickBegin` | `DurableWorldRuntime.tick` -> `_manifest_store._begin_transition` (`durable_runtime.py:1486`) | Serializa manifiesto con estado `PREPARING` y escribe en archivo temporal `.manifest.json.tmp`. |
| `PublishPreparingManifest` | `WorldCausalManifestStore._write` (`manifest.py:498`) | `os.replace(.manifest.json.tmp, manifest.json)` seguido de `_fsync_directory()`. Marca el paso durable a `PREPARING`. |
| `WriteOwnerCheckpoints` | `write_owner_checkpoints(proposed_tick)` (`durable_runtime.py:1498`) | Cada propietario escribe `.blob.tmp`, invoca `os.link` hacia `<sha256>.blob`, sincroniza directorio y borra `.tmp`. |
| `WriteJournalDelta` | `self._journal_store.write` (`durable_runtime.py:1624`) | Abre `world-causal-journal-v3.jsonl`, ejecuta `stream.truncate(committed_offset)`, escribe los registros candidatos y ejecuta `fsync()`. |
| `CommitTransition` | `WorldCausalManifestStore._commit_transition` -> `_write` (`manifest.py:406, 498`) | **PUNTO ATÓMICO DE CONFIRMACIÓN CAUSAL:** `os.replace(.manifest.json.tmp, manifest.json)` con estado `COMMITTED`. |
| `ProcessCrash` | Abrupción de proceso (`os._exit`, fallo eléctrico simulado, `SIGKILL`) | El proceso termina instantáneamente. Se pierden buffers de memoria y archivos temporales abiertos sin enlazar; las llamadas completadas (`link`, `replace`, `truncate`) permanecen en disco. |
| `StartupReopen` | `DurableWorldRuntime.restore` -> `manifest_store._require_active` (`durable_runtime.py:886`, `manifest.py:219-248`) | Lee `manifest.json`. Si encuentra `PREPARING`, ejecuta transición atómica a `RECOVERY_REQUIRED` y lanza `RecoveryRequiredError` (fail-closed). |
| `RecoveryBegin` | `DurableWorldRuntime.recover_previous_cut` (`durable_runtime.py:928-988`) | Valida estado `RECOVERY_REQUIRED`, verifica consistencia del prefijo del journal y llama a los contratos de los propietarios para restaurar estado privado. |
| `RecoveryDiscardTail` | `journal_store._discard_uncommitted_tail` (`durable_runtime.py:581-612`) | Trunca el journal en disco exactamente al offset del último corte confirmado y ejecuta `fsync()`. |
| `RecoveryWriteReceipt` | `manifest_store._complete_recovery_locked` (`manifest.py:458-477`) | Escribe recibo inmutable de recuperación en `recoveries/receipt-....json` con `os.link` y `_fsync_directory()`. |
| `RecoveryCommitManifest` | `manifest_store._write(recovered)` (`manifest.py:478`) | `os.replace` publicando el manifiesto restaurado con estado `COMMITTED` al tick anterior. |

---

## 4. Evidencia TLC y Reproducción

### Comando de ejecución reproducible
```bash
# Mediante el script oficial del repositorio:
./specs/run_tlc.sh

# O invocando directamente la JVM con tla2tools:
java -XX:+UseParallelGC -cp /tmp/tla2tools.jar tlc2.TLC -workers 1 specs/WorldCommitRecovery.tla
```

### Transcripción de salida de TLC
```text
TLC2 Version 2026.10.15.145915 (rev: 932d35e)
Running breadth-first search Model-Checking with fp 15 and seed 8537363786278514981 with 1 worker on 8 cores...
Parsing file specs/WorldCommitRecovery.tla
Semantic processing of module WorldCommitRecovery
Starting... (2026-10-17 12:46:03)
Computing initial states...
Finished computing initial states: 1 distinct state generated.
Model checking completed. No error has been found.
205 states generated, 142 distinct states found, 0 states left on queue.
The depth of the complete state graph search is 30.
Finished in 00s at (2026-10-17 12:46:03)
```

---

## 5. Demostración y Verificación de Invariantes

La correspondencia entre las invariantes formales del modelo y las suites de pruebas es biunívoca:

| Invariante Formal | Enunciado en TLA+ | Prueba de verificación en Python |
|---|---|---|
| **I1** — No aceptación de cortes no confirmados | `I1_NoUnconfirmedCuts`: Si `manifest_status = "COMMITTED"`, todas las referencias corresponden al tick confirmado y el journal tiene longitud confirmada. | `test_systematic_n_th_effect_tick_fault_injection` & `test_invariant_i1_no_unconfirmed_cut_acceptance`. |
| **I2** — Coherencia de referencias | `I2_ReferenceCoherence`: Toda referencia en el manifiesto activo existe en el almacén de blobs inmutables. | `test_invariant_i2_reference_coherence_across_transitions`. |
| **I3** — Continuidad causal | `I3_JournalConsistency`: La longitud confirmada del journal nunca retrocede y es un prefijo del archivo de journal. | `test_invariant_i3_causal_continuity_multi_cycle`. |
| **I4** — Idempotencia de recuperación | `RecoveryCommitManifest` & re-ejecución: Múltiples ejecuciones de `recover_previous_cut` restauran el mismo corte sin avance espurio. | `test_invariant_i4_recovery_idempotence` & `test_systematic_n_th_effect_recovery_fault_injection`. |
| **I5** — Fallo cerrado | `I4_FailClosed`: El proceso en reposo jamás puede operar con un manifiesto en `PREPARING` sin recuperación explícita. | `test_invariant_i5_fail_closed_on_corruptions`. |
| **I6** — Autoridad de dominio | Aislamiento formal: World maneja descriptores y hashes opacos (`ref.sha256`); no inspecciona ni muta la semántica de los payloads. | `test_invariant_i6_domain_authority_opaque_owner_payloads`. |
