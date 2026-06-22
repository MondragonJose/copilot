import { useCallback, useState } from "react";
import { api } from "../api";
import type { CitationInfo } from "../App";
import type { QAResponse } from "../types";
import QuestionForm from "../components/QuestionForm";
import AnswerCard from "../components/AnswerCard";
import AbstainNotice from "../components/AbstainNotice";

interface AskProps {
  papers: { paperId: string; pdfUrl: string }[];
  onNavigateToPaper: (
    paperId: string,
    pdfUrl: string,
    citation: CitationInfo,
  ) => void;
}

export default function Ask({ papers, onNavigateToPaper }: AskProps) {
  const [question, setQuestion] = useState("");
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(async () => {
    if (!question.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const body: { question: string; paper_ids?: string[] } = {
        question: question.trim(),
      };
      if (selectedPaperIds.length > 0) {
        body.paper_ids = selectedPaperIds;
      }
      const res = await api.ask(body);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, [question, selectedPaperIds, loading]);

  const handleNavigate = useCallback(
    (paperId: string, pdfUrl: string, citation: CitationInfo) => {
      onNavigateToPaper(paperId, pdfUrl, citation);
    },
    [onNavigateToPaper],
  );

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 22, margin: "0 0 4px" }}>Ask</h1>
      <p style={{ fontSize: 13, color: "#888", margin: "0 0 16px" }}>
        Ask a question and get grounded answers with citations.
      </p>

      <QuestionForm
        question={question}
        onQuestionChange={setQuestion}
        onSubmit={handleSubmit}
        loading={loading}
        paperIds={papers.map((p) => p.paperId)}
        selectedPaperIds={selectedPaperIds}
        onSelectedPaperIdsChange={setSelectedPaperIds}
      />

      {loading && (
        <div
          data-testid="ask-loading"
          style={{
            textAlign: "center",
            padding: 32,
            color: "#888",
            fontSize: 14,
          }}
        >
          Searching papers and generating answer&hellip;
        </div>
      )}

      {error && (
        <div
          data-testid="ask-error"
          style={{
            background: "#ffebee",
            border: "1px solid #ffcdd2",
            borderRadius: 8,
            padding: 16,
            color: "#c62828",
            fontSize: 14,
          }}
        >
          {error}
        </div>
      )}

      {result && result.answerable && (
        <AnswerCard
          answer={result.answer!}
          claims={result.claims}
          onNavigateToPaper={handleNavigate}
        />
      )}

      {result && !result.answerable && (
        <AbstainNotice question={result.question} />
      )}
    </div>
  );
}
