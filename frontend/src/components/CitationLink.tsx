import type { CitationInfo } from "../App";
import type { ClaimOut } from "../types";
import { BASE_URL } from "../api";

interface CitationLinkProps {
  index: number;
  claim: ClaimOut;
  onNavigateToPaper: (
    paperId: string,
    pdfUrl: string,
    citation: CitationInfo,
  ) => void;
}

export default function CitationLink({
  index,
  claim,
  onNavigateToPaper,
}: CitationLinkProps) {
  const paperId = claim.paper_id ?? undefined;
  const pdfUrl = paperId
    ? `${BASE_URL}/papers/${paperId}/file`
    : undefined;
  const canNavigate = paperId && pdfUrl && claim.page != null;
  const handleClick = () => {
    if (!canNavigate) return;
    onNavigateToPaper(paperId, pdfUrl, {
      page: claim.page!,
      charStart: claim.char_start ?? 0,
      charEnd: claim.char_end ?? 0,
    });
  };

  return (
    <div
      style={{
        padding: "8px 10px",
        marginBottom: 8,
        background: "#fff",
        border: "1px solid #e0e0e0",
        borderRadius: 6,
        fontSize: 13,
      }}
    >
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <span
          style={{
            background: "#1a73e8",
            color: "#fff",
            borderRadius: "50%",
            width: 20,
            height: 20,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 11,
            fontWeight: 700,
            flexShrink: 0,
            marginTop: 1,
          }}
        >
          {index}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ margin: "0 0 4px", lineHeight: 1.4 }}>
            &ldquo;{claim.quoted_span}&rdquo;
          </p>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            {claim.page != null && (
              <span style={{ color: "#888", fontSize: 12 }}>
                p.{claim.page}
                {claim.char_start != null
                  ? `:${claim.char_start}`
                  : ""}
              </span>
            )}
            {canNavigate ? (
              <button
                onClick={handleClick}
                style={{
                  background: "none",
                  border: "none",
                  color: "#1a73e8",
                  cursor: "pointer",
                  fontSize: 12,
                  padding: 0,
                  textDecoration: "underline",
                }}
                data-testid={`citation-link-${index}`}
              >
                View in Reader
              </button>
            ) : (
              <span style={{ color: "#bbb", fontSize: 12 }}>
                {paperId ? "PDF unavailable" : ""}
              </span>
            )}
          </div>
        </div>
      </div>
      <p
        style={{
          margin: "4px 0 0",
          fontSize: 12,
          color: claim.supported ? "#2e7d32" : "#c62828",
        }}
      >
        {claim.supported ? "Supported" : "Not supported"} &middot; Score:{" "}
        {claim.score.toFixed(2)}
      </p>
    </div>
  );
}
