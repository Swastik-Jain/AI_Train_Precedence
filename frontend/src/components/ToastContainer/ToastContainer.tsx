import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, Info, X } from 'lucide-react';
import { useCopilotStore } from '../../store/useCopilotStore';
import type { ToastNotification } from '../../store/useCopilotStore';
import './ToastContainer.css';

const ICONS: Record<ToastNotification['type'], React.ReactNode> = {
  success: <CheckCircle size={15} />,
  error:   <AlertTriangle size={15} />,
  warning: <AlertTriangle size={15} />,
  info:    <Info size={15} />,
};

const Toast: React.FC<{ toast: ToastNotification }> = ({ toast }) => {
  const { dismissToast } = useCopilotStore();

  // Auto-dismiss after 5s
  useEffect(() => {
    const t = setTimeout(() => dismissToast(toast.id), 5000);
    return () => clearTimeout(t);
  }, [toast.id, dismissToast]);

  return (
    <motion.div
      className={`orbit-toast orbit-toast-${toast.type}`}
      layout
      initial={{ opacity: 0, x: 80, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 80, scale: 0.9 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
    >
      <span className="orbit-toast-icon">{ICONS[toast.type]}</span>
      <span className="orbit-toast-msg">{toast.message}</span>
      <button className="orbit-toast-close" onClick={() => dismissToast(toast.id)}>
        <X size={12} />
      </button>
    </motion.div>
  );
};

export const ToastContainer: React.FC = () => {
  const { toasts } = useCopilotStore();

  return (
    <div className="orbit-toast-container" aria-live="polite">
      <AnimatePresence>
        {toasts.map((t) => (
          <Toast key={t.id} toast={t} />
        ))}
      </AnimatePresence>
    </div>
  );
};
