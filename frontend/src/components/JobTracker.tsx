import { useQuery } from "@tanstack/react-query";
import { api, BASE_URL } from "../api";
import type { JobStatusResponse } from "../types";

interface Props {
  jobIds: string[];
  onDismiss: (jobId: string) => void;
  onViewPaper?: (paperId: string, pdfUrl: string) => void;
}

// -----------------------------------------------------------------------
// Single job card — polls while queued / running
// -----------------------------------------------------------------------

function JobCard({
  jobId,
  onDismiss,
  onViewPaper,
}: {
  jobId: string;
  onDismiss: () => void;
  onViewPaper?: (paperId: string, pdfUrl: string) => void;
}) {
  const { data, isLoading, isError } = useQuery<JobStatusResponse>({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      if (s === "done" || s === "dead") return false;
      return 2000;
    },
  });

  if (isLoading) {
    return (
      <div className="job-card" data-testid={`job-${jobId}`}>
        <span className="job-id">{jobId.slice(0, 8)}…</span>
        <span className="status status-queued">fetching</span>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="job-card" data-testid={`job-${jobId}`}>
        <span className="job-id">{jobId.slice(0, 8)}…</span>
        <span className="status status-dead">error</span>
        <button onClick={onDismiss}>✕</button>
      </div>
    );
  }

  const statusClass = `status-${data.status}`;

  return (
    <div className="job-card" data-testid={`job-${jobId}`}>
      <span className={`status ${statusClass}`}>{data.status}</span>
      <span className="job-id">{jobId.slice(0, 8)}…</span>
      {data.stage && <span className="stage">{data.stage}</span>}
      {data.error && <span className="error-msg">{data.error}</span>}
      {data.status === "done" && data.result && onViewPaper && (
        <button
          className="view-btn"
          onClick={() =>
            onViewPaper(
              data.result!,
              `${BASE_URL}/papers/${data.result}/file`,
            )
          }
          data-testid={`view-${jobId}`}
          style={{
            marginLeft: 8,
            padding: "2px 8px",
            fontSize: 12,
            background: "#1a73e8",
            color: "#fff",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          View
        </button>
      )}
      <button className="dismiss-btn" onClick={onDismiss} data-testid={`dismiss-${jobId}`}>
        ✕
      </button>
    </div>
  );
}

// -----------------------------------------------------------------------
// Job list
// -----------------------------------------------------------------------

export function JobTracker({ jobIds, onDismiss, onViewPaper }: Props) {
  if (jobIds.length === 0) return null;

  return (
    <div className="job-tracker" data-testid="job-tracker">
      <h2>Jobs</h2>
      {jobIds.map((id) => (
        <JobCard key={id} jobId={id} onDismiss={() => onDismiss(id)} onViewPaper={onViewPaper} />
      ))}
    </div>
  );
}
