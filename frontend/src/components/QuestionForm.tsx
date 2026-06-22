interface QuestionFormProps {
  question: string;
  onQuestionChange: (q: string) => void;
  onSubmit: () => void;
  loading: boolean;
  paperIds: string[];
  selectedPaperIds: string[];
  onSelectedPaperIdsChange: (ids: string[]) => void;
}

export default function QuestionForm({
  question,
  onQuestionChange,
  onSubmit,
  loading,
  paperIds,
  selectedPaperIds,
  onSelectedPaperIdsChange,
}: QuestionFormProps) {
  const togglePaper = (id: string) => {
    if (selectedPaperIds.includes(id)) {
      onSelectedPaperIdsChange(selectedPaperIds.filter((x) => x !== id));
    } else {
      onSelectedPaperIdsChange([...selectedPaperIds, id]);
    }
  };

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e0e0e0",
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
      }}
    >
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          data-testid="ask-input"
          type="text"
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          placeholder="Ask a question about your papers…"
          style={{
            flex: 1,
            padding: "10px 12px",
            fontSize: 15,
            border: "1px solid #ccc",
            borderRadius: 6,
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && question.trim() && !loading) onSubmit();
          }}
        />
        <button
          data-testid="ask-submit"
          onClick={onSubmit}
          disabled={!question.trim() || loading}
          style={{
            padding: "10px 20px",
            background:
              !question.trim() || loading ? "#ccc" : "#1a73e8",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor:
              !question.trim() || loading ? "not-allowed" : "pointer",
            fontWeight: 600,
            fontSize: 14,
          }}
        >
          {loading ? "Asking…" : "Ask"}
        </button>
      </div>

      {paperIds.length > 0 && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: "#888", alignSelf: "center" }}>
            Scope to papers:
          </span>
          {paperIds.map((id) => (
            <label
              key={id}
              style={{
                fontSize: 12,
                display: "flex",
                alignItems: "center",
                gap: 4,
                cursor: "pointer",
                padding: "2px 8px",
                background: selectedPaperIds.includes(id)
                  ? "#e3f2fd"
                  : "#f5f5f5",
                borderRadius: 12,
                border: selectedPaperIds.includes(id)
                  ? "1px solid #90caf9"
                  : "1px solid #e0e0e0",
              }}
            >
              <input
                type="checkbox"
                checked={selectedPaperIds.includes(id)}
                onChange={() => togglePaper(id)}
                style={{ margin: 0 }}
              />
              {id.slice(0, 8)}&hellip;
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
