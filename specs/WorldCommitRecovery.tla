----------------------- MODULE WorldCommitRecovery -----------------------
(***************************************************************************)
(* Formal specification of World's atomic commit and recovery protocol.   *)
(* Models ADR-0083 (single-writer durability), ADR-0093 (explicit          *)
(* recovery after material failure), and Docs #234 / #236 (crash points).   *)
(*                                                                         *)
(* Focus: POSIX process termination safety (completed syscalls retained;   *)
(* in-flight uncommitted memory/buffers lost).                              *)
(***************************************************************************)

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    MaxTick,            \* Maximum tick bound for model checking (e.g. 2 or 3)
    OwnerIdentities     \* Set of registered semantic owners (e.g. {"organism"})

VARIABLES
    \* Durable filesystem state:
    manifest_status,    \* "COMMITTED" \/ "PREPARING" \/ "RECOVERY_REQUIRED"
    manifest_tick,      \* Confirmed causal tick in active manifest
    manifest_pending_tick, \* Pending tick during PREPARING / RECOVERY_REQUIRED
    manifest_refs,      \* Set of durable checkpoint references in active manifest

    temp_manifest_status, \* In-flight temp manifest status: "NONE" \/ "PREPARING" \/ "COMMITTED"
    temp_manifest_tick,   \* In-flight temp manifest tick
    temp_manifest_pending_tick, \* In-flight temp manifest pending tick
    temp_manifest_refs,   \* In-flight temp manifest references

    journal_committed_len, \* Committed journal length in records
    journal_file_len,      \* Current journal file length on disk (may include uncommitted tail)

    owner_blobs,        \* Set of content-addressed blobs published in owner stores
    recovery_receipts,  \* Set of recovery receipts on disk

    \* Runtime process execution state:
    proc_phase,         \* Process phase: "IDLE", "PREPARING_MANIFEST", "WRITING_OWNERS",
                        \* "WRITING_JOURNAL", "COMMITTING_MANIFEST", "RECOVERING_TAIL",
                        \* "RECOVERING_RECEIPT", "RECOVERING_MANIFEST", "CRASHED"
    candidate_tick,     \* Next proposed tick in memory
    candidate_refs      \* Proposed checkpoint references in memory

vars == <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
          temp_manifest_status, temp_manifest_tick, temp_manifest_pending_tick, temp_manifest_refs,
          journal_committed_len, journal_file_len, owner_blobs,
          recovery_receipts, proc_phase, candidate_tick, candidate_refs>>

-----------------------------------------------------------------------------
(* Initial state: World genesis cut confirmed at tick 0 *)

Init ==
    /\ manifest_status = "COMMITTED"
    /\ manifest_tick = 0
    /\ manifest_pending_tick = 0
    /\ manifest_refs = {[owner |-> o, tick |-> 0] : o \in OwnerIdentities}
    /\ temp_manifest_status = "NONE"
    /\ temp_manifest_tick = 0
    /\ temp_manifest_pending_tick = 0
    /\ temp_manifest_refs = {}
    /\ journal_committed_len = 1
    /\ journal_file_len = 1
    /\ owner_blobs = {[owner |-> o, tick |-> 0] : o \in OwnerIdentities}
    /\ recovery_receipts = {}
    /\ proc_phase = "IDLE"
    /\ candidate_tick = 0
    /\ candidate_refs = {}

-----------------------------------------------------------------------------
(* Tick step 1: Begin transition -> write PREPARING temporary manifest *)

TickBegin ==
    /\ proc_phase = "IDLE"
    /\ manifest_status = "COMMITTED"
    /\ manifest_tick < MaxTick
    /\ proc_phase' = "PREPARING_MANIFEST"
    /\ candidate_tick' = manifest_tick + 1
    /\ candidate_refs' = {}
    /\ temp_manifest_status' = "PREPARING"
    /\ temp_manifest_tick' = manifest_tick
    /\ temp_manifest_pending_tick' = manifest_tick + 1
    /\ temp_manifest_refs' = manifest_refs
    /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                  journal_committed_len, journal_file_len, owner_blobs, recovery_receipts>>

(* Atomic replacement of PREPARING manifest on disk *)
PublishPreparingManifest ==
    /\ proc_phase = "PREPARING_MANIFEST"
    /\ temp_manifest_status = "PREPARING"
    /\ manifest_status' = "PREPARING"
    /\ manifest_pending_tick' = candidate_tick
    /\ temp_manifest_status' = "NONE"
    /\ temp_manifest_tick' = 0
    /\ temp_manifest_pending_tick' = 0
    /\ temp_manifest_refs' = {}
    /\ proc_phase' = "WRITING_OWNERS"
    /\ UNCHANGED <<manifest_tick, manifest_refs, journal_committed_len, journal_file_len,
                  owner_blobs, recovery_receipts, candidate_tick, candidate_refs>>

(* Tick step 2: Owners write opaque content-addressed checkpoints *)
WriteOwnerCheckpoints ==
    /\ proc_phase = "WRITING_OWNERS"
    /\ LET new_refs == {[owner |-> o, tick |-> candidate_tick] : o \in OwnerIdentities}
       IN
          /\ owner_blobs' = owner_blobs \cup new_refs
          /\ candidate_refs' = new_refs
          /\ proc_phase' = "WRITING_JOURNAL"
          /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                        temp_manifest_status, temp_manifest_tick, temp_manifest_pending_tick,
                        temp_manifest_refs, journal_committed_len, journal_file_len,
                        recovery_receipts, candidate_tick>>

(* Tick step 3: Journal write (truncate to committed offset, append delta) *)
WriteJournalDelta ==
    /\ proc_phase = "WRITING_JOURNAL"
    /\ journal_file_len' = journal_committed_len + 1
    /\ proc_phase' = "COMMITTING_MANIFEST"
    /\ temp_manifest_status' = "COMMITTED"
    /\ temp_manifest_tick' = candidate_tick
    /\ temp_manifest_pending_tick' = 0
    /\ temp_manifest_refs' = candidate_refs
    /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                  journal_committed_len, owner_blobs, recovery_receipts, candidate_tick,
                  candidate_refs>>

(* Tick step 4: Atomic replacement of COMMITTED manifest -> ATOMIC COMMIT POINT *)
CommitTransition ==
    /\ proc_phase = "COMMITTING_MANIFEST"
    /\ temp_manifest_status = "COMMITTED"
    /\ manifest_status' = "COMMITTED"
    /\ manifest_tick' = candidate_tick
    /\ manifest_pending_tick' = 0
    /\ manifest_refs' = candidate_refs
    /\ journal_committed_len' = journal_file_len
    /\ temp_manifest_status' = "NONE"
    /\ temp_manifest_tick' = 0
    /\ temp_manifest_pending_tick' = 0
    /\ temp_manifest_refs' = {}
    /\ proc_phase' = "IDLE"
    /\ candidate_tick' = 0
    /\ candidate_refs' = {}
    /\ UNCHANGED <<journal_file_len, owner_blobs, recovery_receipts>>

-----------------------------------------------------------------------------
(* Process crash & restart at any execution point *)

ProcessCrash ==
    /\ proc_phase \notin {"IDLE", "CRASHED"}
    /\ proc_phase' = "CRASHED"
    /\ temp_manifest_status' = "NONE"
    /\ temp_manifest_tick' = 0
    /\ temp_manifest_pending_tick' = 0
    /\ temp_manifest_refs' = {}
    /\ candidate_tick' = 0
    /\ candidate_refs' = {}
    /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                  journal_committed_len, journal_file_len, owner_blobs, recovery_receipts>>

(* Startup reopen from crash *)
StartupReopen ==
    /\ proc_phase = "CRASHED"
    /\ IF manifest_status = "PREPARING"
       THEN \* Fail closed: promote interrupted PREPARING to RECOVERY_REQUIRED
            /\ manifest_status' = "RECOVERY_REQUIRED"
            /\ proc_phase' = "IDLE"
            /\ UNCHANGED <<manifest_tick, manifest_pending_tick, manifest_refs,
                          temp_manifest_status, temp_manifest_tick, temp_manifest_pending_tick,
                          temp_manifest_refs, journal_committed_len, journal_file_len,
                          owner_blobs, recovery_receipts, candidate_tick, candidate_refs>>
       ELSE \* COMMITTED or already RECOVERY_REQUIRED
            /\ proc_phase' = "IDLE"
            /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                          temp_manifest_status, temp_manifest_tick, temp_manifest_pending_tick,
                          temp_manifest_refs, journal_committed_len, journal_file_len,
                          owner_blobs, recovery_receipts, candidate_tick, candidate_refs>>

-----------------------------------------------------------------------------
(* Recovery protocol: recover_previous_cut *)

RecoveryBegin ==
    /\ proc_phase = "IDLE"
    /\ manifest_status = "RECOVERY_REQUIRED"
    /\ proc_phase' = "RECOVERING_TAIL"
    /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                  temp_manifest_status, temp_manifest_tick, temp_manifest_pending_tick,
                  temp_manifest_refs, journal_committed_len, journal_file_len,
                  owner_blobs, recovery_receipts, candidate_tick, candidate_refs>>

(* Recovery step 1: discard uncommitted journal tail *)
RecoveryDiscardTail ==
    /\ proc_phase = "RECOVERING_TAIL"
    /\ journal_file_len' = journal_committed_len
    /\ proc_phase' = "RECOVERING_RECEIPT"
    /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                  temp_manifest_status, temp_manifest_tick, temp_manifest_pending_tick,
                  temp_manifest_refs, journal_committed_len, owner_blobs,
                  recovery_receipts, candidate_tick, candidate_refs>>

(* Recovery step 2: write immutable recovery receipt *)
RecoveryWriteReceipt ==
    /\ proc_phase = "RECOVERING_RECEIPT"
    /\ recovery_receipts' = recovery_receipts \cup {[transition |-> manifest_pending_tick,
                                                     restored_tick |-> manifest_tick]}
    /\ proc_phase' = "RECOVERING_MANIFEST"
    /\ temp_manifest_status' = "COMMITTED"
    /\ temp_manifest_tick' = manifest_tick
    /\ temp_manifest_pending_tick' = 0
    /\ temp_manifest_refs' = manifest_refs
    /\ UNCHANGED <<manifest_status, manifest_tick, manifest_pending_tick, manifest_refs,
                  journal_committed_len, journal_file_len, owner_blobs, candidate_tick,
                  candidate_refs>>

(* Recovery step 3: atomic replace of recovered COMMITTED manifest *)
RecoveryCommitManifest ==
    /\ proc_phase = "RECOVERING_MANIFEST"
    /\ temp_manifest_status = "COMMITTED"
    /\ manifest_status' = "COMMITTED"
    /\ manifest_pending_tick' = 0
    /\ temp_manifest_status' = "NONE"
    /\ temp_manifest_tick' = 0
    /\ temp_manifest_pending_tick' = 0
    /\ temp_manifest_refs' = {}
    /\ proc_phase' = "IDLE"
    /\ UNCHANGED <<manifest_tick, manifest_refs, journal_committed_len, journal_file_len,
                  owner_blobs, recovery_receipts, candidate_tick, candidate_refs>>

-----------------------------------------------------------------------------
(* Transition relation *)

Next ==
    \/ TickBegin
    \/ PublishPreparingManifest
    \/ WriteOwnerCheckpoints
    \/ WriteJournalDelta
    \/ CommitTransition
    \/ ProcessCrash
    \/ StartupReopen
    \/ RecoveryBegin
    \/ RecoveryDiscardTail
    \/ RecoveryWriteReceipt
    \/ RecoveryCommitManifest

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Safety Invariants *)

\* I1: No unconfirmed cuts accepted as committed
I1_NoUnconfirmedCuts ==
    (manifest_status = "COMMITTED") =>
        /\ \A ref \in manifest_refs: ref.tick = manifest_tick
        /\ journal_committed_len >= 1

\* I2: Checkpoint reference coherence: all references in active manifest exist in store
I2_ReferenceCoherence ==
    \A ref \in manifest_refs: ref \in owner_blobs

\* I3: Journal prefix matches committed length
I3_JournalConsistency ==
    /\ journal_committed_len <= journal_file_len
    /\ (manifest_status = "COMMITTED" => journal_committed_len = manifest_tick + 1)

\* I4: Fail-closed on interrupted preparation
I4_FailClosed ==
    (proc_phase = "IDLE" /\ manifest_status = "PREPARING") => FALSE

\* I5: Recovery receipt points to a valid restored tick
I5_ReceiptIntegrity ==
    \A receipt \in recovery_receipts:
        receipt.restored_tick <= manifest_tick

=============================================================================
