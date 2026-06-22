import type { CitationInfo } from "../App";
import type { ClaimOut } from "../types";
import CitationLink from "./CitationLink";

interface AnswerCardProps {
  answer: string;
  claims: ClaimOut[];
  onNavigateToPaper: (
    paperId: string,
    pdfUrl: string,
    citation: CitationInfo,
  ) => void;
}

export default function AnswerCard({
  answer,
  claims,
  onNavigateToPaper,
}: AnswerCardProps) {
  const supported = claims.filter((c) => c.supported);
  const unsupported = claims.filter((c) => !c.supported);

  return (
    <div
      data-testid="answer-card"
      style={{
        background: "#fff",
        border: "1px solid #e0e0e0",
        borderRadius: 8,
        padding: 16,
      }}
    >
      <h3 style={{ margin: "0 0 8px", fontSize: 16, color: "#333" }}>
        Answer
      </h3>
      <p
        data-testid="answer-text"
        style={{ margin: "0 0 16px", lineHeight: 1.6, fontSize: 14 }}
      >
        {answer}
      </p>

      {supported.length > 0 && (
        <>
          <h4 style={{ margin: "0 0 8px", fontSize: 14, color: "#555" }}>
            Supporting Evidence
          </h4>
          {supported.map((c, i) => (
            <CitationLink
              key={c.chunk_id + i}
              index={i + 1}
              claim={c}
              onNavigateToPaper={onNavigateToPaper}
            />
          ))}
        </>
      )}

      {unsupported.length > 0 && (
        <>
          <h4
            style={{
              margin: "12px 0 8px",
              fontSize: 14,
              color: "#c62828",
            }}
          >
            Unsupported Claims
          </h4>
          {unsupported.map((c, i) => (
            <CitationLink
              key={c.chunk_id + i}
              index={supported.length + i + 1}
              claim={c}
              onNavigateToPaper={onNavigateToPaper}
            />
          ))}
        </>
      )}
    </div>
  );
}
