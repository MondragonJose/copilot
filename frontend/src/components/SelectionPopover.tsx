import { useEffect, useRef } from "react";

interface SelectionPopoverProps {
  top: number;
  left: number;
  quote: string;
  explainLoading?: boolean;
  onExplain: () => void;
  onDismiss: () => void;
}

export default function SelectionPopover({
  top,
  left,
  quote,
  explainLoading,
  onExplain,
  onDismiss,
}: SelectionPopoverProps) {
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let handler: ((e: MouseEvent) => void) | null = null;
    const timer = setTimeout(() => {
      handler = (e: MouseEvent) => {
        if (
          popoverRef.current &&
          !popoverRef.current.contains(e.target as Node)
        ) {
          onDismiss();
        }
      };
      document.addEventListener("mousedown", handler);
    }, 0);
    return () => {
      clearTimeout(timer);
      if (handler) {
        document.removeEventListener("mousedown", handler);
      }
    };
  }, [onDismiss]);

  return (
    <div
      ref={popoverRef}
      className="selection-popover"
      style={{
        position: "fixed",
        top: top + 8,
        left: left,
        transform: "translateX(-50%)",
        zIndex: 1000,
        background: "#fff",
        border: "1px solid #ccc",
        borderRadius: 8,
        boxShadow: "0 2px 12px rgba(0,0,0,0.15)",
        padding: "8px 12px",
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontSize: 14,
      }}
    >
      <span
        style={{
          maxWidth: 200,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          color: "#555",
        }}
      >
        &ldquo;{quote.slice(0, 60)}{quote.length > 60 ? "…" : ""}&rdquo;
      </span>
      <button
        onClick={onExplain}
        disabled={explainLoading}
        style={{
          padding: "4px 12px",
          background: explainLoading ? "#ccc" : "#1a73e8",
          color: "#fff",
          border: "none",
          borderRadius: 4,
          cursor: explainLoading ? "not-allowed" : "pointer",
          fontWeight: 500,
        }}
      >
        {explainLoading ? "Explaining…" : "Explain"}
      </button>
    </div>
  );
}
