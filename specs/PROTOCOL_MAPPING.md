# World V-03: Mapeo Formal del Protocolo de Commit y Recuperación

**Especificación TLA+:** `specs/WorldCommitRecovery.tla`  
**Configuración:** `specs/WorldCommitRecovery.cfg`  
**Implementación Python:** `src/world/manifest.py`, `src/world/durable_runtime.py`, `src/world/owner_checkpoint_store.py`  
**Suite de pruebas:** `tests/test_systematic_fault_injection.py`  
**Autoridades normativas:** ADR-0083 (Single-Writer Durability), ADR-0093 (Fail-Closed Recovery), Docs #234, #236.

---

## 1. Correspondencia de Variables de Estado

| Variable TLA+ | Tipo formal | Equivalente en código Python / Sistema de archivos |
|---|---|---|
| `manifest_status` | `{"COMMITTED", "PREPARING", "RECOVERY_REQUIRED"}` | `WorldCausalManifest.status` (`ManifestStatus` enum en `src/world/manifest.py:30-33`). Leído de `manifest.json`. |
| `manifest_tick` | `Nat` | `WorldCausalManifest.causal_tick` (entero no negativo en `manifest.py:46`). Representa el último corte causal confirmado. |
| `manifest_pending_tick` | `Nat` | `WorldCausalManifest.pending_tick` (entero o `None` en `manifest.py:58`). Solo presente durante `PREPARING` o `RECOVERY_REQUIRED`. |
| `manifest_refs` | `Set(CheckpointReference)` | `WorldCausalManifest.checkpoint_references` (tupla inmutable de referencias en `manifest.py:48`). |
| `temp_manifest` | `Record \/ NULL` | Archivo temporal `.manifest.json.XXXXXX` creado por `tempfile.mkstemp` en `WorldCausalManifestStore._write` (`manifest.py:492`). Desaparece tras `os.replace` o tras caída del proceso. |
| `journal_committed_len` | `Nat` | Longitud en bytes / registros confirmados apuntada por la referencia `CheckpointReference` con dueño `world-journal` (`durable_runtime.py:597`). |
| `journal_file_len` | `Nat` | Tamaño real del archivo `world-causal-journal-v3.jsonl` en disco. Puede exceder a `journal_committed_len` si un proceso murió tras escribir bytes no confirmados. |
| `owner_blobs` | `Set(Blob)` | Archivos inmutables direccionados por contenido en el directorio del almacén de checkpoints (`xx/<sha256>.blob` en `FileOwnerCheckpointStore`, `owner_checkpoint_store.py:83-140`). |
| `recovery_receipts` | `Set(Receipt)` | Archivos JSON inmutables en `recoveries/receipt-<world>-<transition>-<tick>.json` generados por `WorldCausalManifestStore._complete_recovery_locked` (`manifest.py:458-477`). |
| `proc_phase` | Enum de fases | Estado de ejecución en memoria de `DurableWorldRuntime` durante `tick()` o `recover_previous_cut()`. Desaparece ante `os._exit()`. |

---

## 2. Correspondencia de Acciones Abstractas a Operaciones Concretas

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

## 3. Demostración y Verificación de Invariantes

La correspondencia entre las invariantes formales del modelo y las suites de pruebas es biunívoca:

| Invariante Formal | Enunciado en TLA+ | Prueba de verificación en Python |
|---|---|---|
| **I1** — No aceptación de cortes no confirmados | `I1_NoUnconfirmedCuts`: Si `manifest_status = "COMMITTED"`, todas las referencias corresponden al tick confirmado y el journal tiene longitud confirmada. | `test_systematic_n_th_effect_tick_fault_injection` & `test_invariant_i1_no_unconfirmed_cut_acceptance`. |
| **I2** — Coherencia de referencias | `I2_ReferenceCoherence`: Toda referencia en el manifiesto activo existe en el almacén de blobs inmutables. | `test_invariant_i2_reference_coherence_across_transitions`. |
| **I3** — Continuidad causal | `I3_JournalConsistency`: La longitud confirmada del journal nunca retrocede y es un prefijo del archivo de journal. | `test_invariant_i3_causal_continuity_multi_cycle`. |
| **I4** — Idempotencia de recuperación | `RecoveryCommitManifest` & re-ejecución: Múltiples ejecuciones de `recover_previous_cut` restauran el mismo corte sin avance espurio. | `test_invariant_i4_recovery_idempotence` & `test_systematic_n_th_effect_recovery_fault_injection`. |
| **I5** — Fallo cerrado | `I4_FailClosed`: El proceso en reposo jamás puede operar con un manifiesto en `PREPARING` sin recuperación explícita. | `test_invariant_i5_fail_closed_on_corruptions`. |
| **I6** — Autoridad de dominio | Aislamiento formal: World maneja descriptores y hashes opacos (`ref.sha256`); no inspecciona ni muta la semántica de los payloads. | `test_invariant_i6_domain_authority_opaque_owner_payloads`. |
