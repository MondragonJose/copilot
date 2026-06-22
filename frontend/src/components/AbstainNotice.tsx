interface AbstainNoticeProps {
  question: string;
}

export default function AbstainNotice({ question }: AbstainNoticeProps) {
  return (
    <div
      data-testid="abstain-notice"
      style={{
        background: "#fff3e0",
        border: "1px solid #ffe0b2",
        borderRadius: 8,
        padding: 16,
        textAlign: "center",
      }}
    >
      <p
        style={{
          fontSize: 24,
          margin: "0 0 8px",
        }}
      >
        &#x26A0;
      </p>
      <h3
        style={{
          margin: "0 0 4px",
          fontSize: 16,
          color: "#e65100",
        }}
      >
        Cannot answer this question
      </h3>
      <p style={{ margin: 0, fontSize: 14, color: "#bf360c" }}>
        The available papers do not contain enough information to answer &ldquo;
        {question}&rdquo;. Try rephrasing or importing more relevant papers.
      </p>
    </div>
  );
}
