import { useCallback, useState } from "react";
import { UploadForm } from "../components/UploadForm";
import { DoiForm } from "../components/DoiForm";
import { JobTracker } from "../components/JobTracker";

interface CorpusProps {
  onViewPaper?: (paperId: string, pdfUrl: string) => void;
}

export default function Corpus({ onViewPaper }: CorpusProps) {
  const [jobIds, setJobIds] = useState<string[]>([]);

  const addJob = useCallback((jobId: string) => {
    setJobIds((prev) => [jobId, ...prev]);
  }, []);

  const removeJob = useCallback((jobId: string) => {
    setJobIds((prev) => prev.filter((id) => id !== jobId));
  }, []);

  return (
    <div className="corpus-screen">
      <h1>Import / Corpus</h1>

      <section className="import-section">
        <UploadForm onJobCreated={addJob} />
      </section>

      <section className="import-section">
        <DoiForm onJobCreated={addJob} />
      </section>

      <JobTracker jobIds={jobIds} onDismiss={removeJob} onViewPaper={onViewPaper} />
    </div>
  );
}
