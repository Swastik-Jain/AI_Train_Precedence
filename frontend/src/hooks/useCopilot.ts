import { useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCopilotStore, type AISuggestion } from '../store/useCopilotStore';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type ActionResult =
  | { ok: true; timestamp: string }
  | { ok: false; safetyConflict: boolean; message: string };

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------
export function useCopilot() {
  const navigate = useNavigate();
  const {
    activeSuggestions,
    previewState,
    globalSchedule,
    toasts,
    isConnected,
    connectCopilotWS,
    setPreviewState,
    commitSuggestion,
    rejectSuggestion,
    addToast,
    dismissToast,
  } = useCopilotStore();

  // Connect to the copilot WebSocket once on mount
  useEffect(() => {
    connectCopilotWS();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------------------
  // executeAction — POST /api/v1/dispatch/commit
  // Performs the "Verifying with OR-Tools…" wait then commits.
  // ---------------------------------------------------------------------------
  const executeAction = useCallback(
    async (recommendation_id: string): Promise<ActionResult> => {
      try {
        // Small artificial delay to show the "Verifying with OR-Tools…" micro-animation
        await new Promise((r) => setTimeout(r, 1500));

        const res = await fetch('/api/v1/dispatch/commit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recommendation_id }),
        });

        if (res.status === 409) {
          const body = await res.json();
          const msg: string = body.detail ?? 'Safety interlock violation detected.';
          const isSafetyConflict = msg.startsWith('SafetyConflict');
          addToast('error', `⚠️ ${isSafetyConflict ? 'Safety Conflict: ' : ''}${msg}`);
          return { ok: false, safetyConflict: isSafetyConflict, message: msg };
        }

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          const msg = body.detail ?? `Server error ${res.status}`;
          addToast('error', msg);
          return { ok: false, safetyConflict: false, message: msg };
        }

        const body = await res.json();

        // Update Zustand — transition ghost → actual path
        commitSuggestion(recommendation_id);

        addToast(
          'success',
          `✅ Committed: "${body.proposed_action}" for ${body.target_train_id} at ${body.timestamp}`
        );

        return { ok: true, timestamp: body.timestamp };
      } catch (err) {
        const msg = 'Network error — backend may be offline.';
        addToast('error', msg);
        return { ok: false, safetyConflict: false, message: msg };
      }
    },
    [commitSuggestion, addToast]
  );

  // ---------------------------------------------------------------------------
  // rejectAction — POST /api/v1/dispatch/reject
  // ---------------------------------------------------------------------------
  const rejectAction = useCallback(
    async (recommendation_id: string, reason = 'controller_dismissed'): Promise<void> => {
      // Optimistically remove from UI immediately
      rejectSuggestion(recommendation_id);

      try {
        await fetch('/api/v1/dispatch/reject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recommendation_id, reason }),
        });
      } catch {
        // Fire-and-forget; rejection signal for RL fine-tuning
        console.warn('[ORBIT] Reject signal failed to reach backend — will retry on reconnect');
      }
    },
    [rejectSuggestion]
  );

  // ---------------------------------------------------------------------------
  // previewAction — sets ghost projection state
  // ---------------------------------------------------------------------------
  const previewAction = useCallback(
    (suggestion: AISuggestion) => {
      setPreviewState(suggestion);
    },
    [setPreviewState]
  );

  // ---------------------------------------------------------------------------
  // clearPreview
  // ---------------------------------------------------------------------------
  const clearPreview = useCallback(() => {
    setPreviewState(null);
  }, [setPreviewState]);

  // ---------------------------------------------------------------------------
  // modifyAction — navigate to Simulation Sandbox with suggestion pre-loaded
  // ---------------------------------------------------------------------------
  const modifyAction = useCallback(
    (suggestion: AISuggestion) => {
      navigate('/sandbox', {
        state: {
          preloadedSuggestion: suggestion,
        },
      });
    },
    [navigate]
  );

  return {
    activeSuggestions,
    previewState,
    globalSchedule,
    toasts,
    isConnected,
    executeAction,
    rejectAction,
    previewAction,
    clearPreview,
    modifyAction,
    dismissToast,
  };
}
