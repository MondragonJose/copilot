import { useState, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api";

interface Props {
  onJobCreated: (jobId: string) => void;
}

export function UploadForm({ onJobCreated }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const mutation = useMutation({
    mutationFn: (f: File) => api.ingestFile(f),
    onSuccess: (data) => {
      onJobCreated(data.job_id);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    },
  });

  return (
    <div className="upload-form">
      <h2>Upload PDF</h2>
      <div className="form-row">
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          data-testid="pdf-input"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          data-testid="upload-btn"
          disabled={!file || mutation.isPending}
          onClick={() => file && mutation.mutate(file)}
        >
          {mutation.isPending ? "Uploading…" : "Upload"}
        </button>
      </div>
      {mutation.isError && (
        <p className="error" data-testid="upload-error">
          {mutation.error.message}
        </p>
      )}
    </div>
  );
}
