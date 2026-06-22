import { useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import PdfViewer from "../components/PdfViewer";

interface ReaderProps {
  paperId: string;
  pdfUrl: string;
  citation?: { page: number; charStart: number; charEnd: number };
  onBack: () => void;
}

export default function Reader({ paperId, pdfUrl, citation, onBack }: ReaderProps) {
  const queryClient = useQueryClient();
  const annotationsKey = ["annotations", paperId];

  const { data: annotations = [] } = useQuery({
    queryKey: annotationsKey,
    queryFn: () => api.getAnnotations(paperId),
  });

  const createAnnotation = useMutation({
    mutationFn: (body: Parameters<typeof api.createAnnotation>[1]) =>
      api.createAnnotation(paperId, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: annotationsKey });
    },
  });

  const handleExplain = useCallback(
    async (info: {
      page: number;
      charStart: number;
      charEnd: number;
      quote: string;
    }) => {
      await createAnnotation.mutateAsync({
        page: info.page,
        char_start: info.charStart,
        char_end: info.charEnd,
        quote: info.quote,
        note: "Explain this passage",
      });
    },
    [createAnnotation],
  );

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        flexDirection: "column",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "8px 16px",
          borderBottom: "1px solid #ddd",
          background: "#fafafa",
        }}
      >
        <button
          onClick={onBack}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 16,
            color: "#1a73e8",
          }}
        >
          &larr; Corpus
        </button>
        <span style={{ fontWeight: 600, fontSize: 16 }}>Reader</span>
        <span style={{ color: "#888", fontSize: 13 }}>Paper {paperId}</span>
      </header>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <div style={{ flex: 1, overflow: "auto" }}>
          <PdfViewer file={pdfUrl} initialPage={citation?.page} onExplain={handleExplain} />
        </div>

        <aside
          style={{
            width: 300,
            borderLeft: "1px solid #ddd",
            overflowY: "auto",
            padding: 12,
            background: "#fafafa",
          }}
        >
          <h3 style={{ margin: "0 0 12px" }}>Annotations</h3>
          {annotations.length === 0 && (
            <p style={{ color: "#888", fontSize: 13 }}>
              Select text in the PDF and click Explain to add an annotation.
            </p>
          )}
          {annotations.map((a) => (
            <div
              key={a.id}
              style={{
                padding: "8px 10px",
                marginBottom: 8,
                background: "#fff",
                border: "1px solid #e0e0e0",
                borderRadius: 6,
                fontSize: 13,
              }}
            >
              <p
                style={{
                  margin: "0 0 4px",
                  fontStyle: "italic",
                  color: "#555",
                }}
              >
                &ldquo;{a.quote.slice(0, 80)}{a.quote.length > 80 ? "…" : ""}&rdquo;
              </p>
              <p style={{ margin: 0, color: "#888" }}>
                p.{a.page} &middot; {a.kind === "ai_explain" ? "AI" : "User"}
              </p>
              {a.note && (
                <p style={{ margin: "4px 0 0", color: "#333" }}>{a.note}</p>
              )}
            </div>
          ))}
        </aside>
      </div>
    </div>
  );
}
