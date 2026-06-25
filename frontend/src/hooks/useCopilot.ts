import { useEffect, useCallback } from 'react';
import { useCopilotStore, type AISuggestion } from '../store/useCopilotStore';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type ActionResult =
  | { ok: true; timestamp: string }
  | { ok: false; safetyConflict: boolean; message: string };

// ---------------------------------------------------------------------------
export function useCopilot() {
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
  // overrideAction — POST /api/v1/dispatch/override
  // Performs the "Applying Override…" wait then commits.
  // ---------------------------------------------------------------------------
  const overrideAction = useCallback(
    async (
      recommendation_id: string,
      new_action?: number,
      new_edge?: string
    ): Promise<ActionResult> => {
      try {
        // Small artificial delay to show the "Applying Override…" micro-animation
        await new Promise((r) => setTimeout(r, 1500));

        const bodyPayload: any = { recommendation_id };
        if (new_action !== undefined) bodyPayload.new_action = new_action;
        if (new_edge !== undefined) bodyPayload.new_edge = new_edge;

        const res = await fetch('/api/v1/dispatch/override', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(bodyPayload),
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

        // Update Zustand
        // (Handled by WSSCHEDULE_UPDATED but we can also update locally)
        // No need to commitSuggestion to remove it, we just mark it as overridden.
        
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
  // rejectAction — (now just a local dismiss + acknowledge to backend)
  const rejectAction = useCallback(
    async (recommendation_id: string): Promise<void> => {
      // Optimistically remove from UI immediately
      rejectSuggestion(recommendation_id);

      try {
        await fetch('/api/v1/dispatch/acknowledge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recommendation_id }),
        });
      } catch {
        // Fire-and-forget
        console.warn('[ORBIT] Acknowledge signal failed to reach backend');
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

  return {
    activeSuggestions,
    previewState,
    globalSchedule,
    toasts,
    isConnected,
    overrideAction,
    rejectAction,
    previewAction,
    clearPreview,
    dismissToast,
  };
}
