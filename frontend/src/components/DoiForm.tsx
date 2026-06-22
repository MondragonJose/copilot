import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api";

interface Props {
  onJobCreated: (jobId: string) => void;
}

export function DoiForm({ onJobCreated }: Props) {
  const [doi, setDoi] = useState("");

  const mutation = useMutation({
    mutationFn: (d: string) => api.ingestDoi(d),
    onSuccess: (data) => {
      onJobCreated(data.job_id);
      setDoi("");
    },
  });

  return (
    <div className="doi-form">
      <h2>Import by DOI</h2>
      <div className="form-row">
        <input
          type="text"
          placeholder="10.1234/example"
          data-testid="doi-input"
          value={doi}
          onChange={(e) => setDoi(e.target.value)}
        />
        <button
          data-testid="doi-btn"
          disabled={!doi || mutation.isPending}
          onClick={() => mutation.mutate(doi)}
        >
          {mutation.isPending ? "Submitting…" : "Import"}
        </button>
      </div>
      {mutation.isError && (
        <p className="error" data-testid="doi-error">
          {mutation.error.message}
        </p>
      )}
    </div>
  );
}
