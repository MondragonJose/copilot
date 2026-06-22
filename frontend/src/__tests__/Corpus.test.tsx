import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Corpus from "../screens/Corpus";
import { api } from "../api";

// ---------------------------------------------------------------------------
// Mock the API module
// ---------------------------------------------------------------------------
vi.mock("../api", () => ({
  api: {
    ingestFile: vi.fn(),
    ingestDoi: vi.fn(),
    getJob: vi.fn(),
  },
  BASE_URL: "http://localhost:8000",
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

const mockedApi = vi.mocked(api);

const baseJob = {
  kind: "ingest_pdf",
  stage: null,
  error: null,
  attempts: 0,
  max_attempts: 3,
  created_at: "2025-01-01T00:00:00",
  updated_at: null,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function renderApp(props?: { onViewPaper?: (paperId: string, pdfUrl: string) => void }) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <Corpus {...props} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

// ===================================================================
// Upload form
// ===================================================================

describe("UploadForm", () => {
  it("renders file input and upload button", () => {
    renderApp();
    expect(screen.getByTestId("pdf-input")).toBeInTheDocument();
    expect(screen.getByTestId("upload-btn")).toBeInTheDocument();
  });

  it("disables button when no file is selected", () => {
    renderApp();
    expect(screen.getByTestId("upload-btn")).toBeDisabled();
  });

  it("calls api.ingestFile on submit and shows job card", async () => {
    const user = userEvent.setup();
    mockedApi.ingestFile.mockResolvedValue({ job_id: "job-001" });

    renderApp();

    const file = new File(["dummy"], "test.pdf", { type: "application/pdf" });
    const input = screen.getByTestId("pdf-input");
    await user.upload(input, file);

    const btn = screen.getByTestId("upload-btn");
    expect(btn).not.toBeDisabled();
    await user.click(btn);

    expect(mockedApi.ingestFile).toHaveBeenCalledWith(file);
    // job card appears
    expect(screen.getByTestId("job-job-001")).toBeInTheDocument();
  });

  it("shows error message on upload failure", async () => {
    const user = userEvent.setup();
    mockedApi.ingestFile.mockRejectedValue(new Error("Upload failed"));

    renderApp();

    const file = new File(["d"], "bad.pdf", { type: "application/pdf" });
    await user.upload(screen.getByTestId("pdf-input"), file);
    await user.click(screen.getByTestId("upload-btn"));

    expect(await screen.findByTestId("upload-error")).toHaveTextContent(
      "Upload failed",
    );
  });
});

// ===================================================================
// DOI form
// ===================================================================

describe("DoiForm", () => {
  it("renders text input and import button", () => {
    renderApp();
    expect(screen.getByTestId("doi-input")).toBeInTheDocument();
    expect(screen.getByTestId("doi-btn")).toBeInTheDocument();
  });

  it("disables button when input is empty", () => {
    renderApp();
    expect(screen.getByTestId("doi-btn")).toBeDisabled();
  });

  it("calls api.ingestDoi on submit and shows job card", async () => {
    const user = userEvent.setup();
    mockedApi.ingestDoi.mockResolvedValue({ job_id: "job-002" });

    renderApp();

    await user.type(screen.getByTestId("doi-input"), "10.1234/test");
    await user.click(screen.getByTestId("doi-btn"));

    expect(mockedApi.ingestDoi).toHaveBeenCalledWith("10.1234/test");
    expect(screen.getByTestId("job-job-002")).toBeInTheDocument();
  });

  it("shows error message on DOI failure", async () => {
    const user = userEvent.setup();
    mockedApi.ingestDoi.mockRejectedValue(new Error("DOI not found"));

    renderApp();

    await user.type(screen.getByTestId("doi-input"), "10.xxx/bad");
    await user.click(screen.getByTestId("doi-btn"));

    expect(await screen.findByTestId("doi-error")).toHaveTextContent(
      "DOI not found",
    );
  });
});

// ===================================================================
// JobTracker — polling + status rendering
// ===================================================================

describe("JobTracker", () => {
  it("shows nothing when no jobs exist", () => {
    renderApp();
    expect(screen.queryByTestId("job-tracker")).not.toBeInTheDocument();
  });

  it("polls and displays done status", async () => {
    const user = userEvent.setup();
    mockedApi.ingestDoi.mockResolvedValue({ job_id: "j1" });
    mockedApi.getJob.mockResolvedValue({
      ...baseJob,
      id: "j1",
      status: "done",
      updated_at: "2025-01-01T01:00:00",
    });

    renderApp();

    await user.type(screen.getByTestId("doi-input"), "10.1/abc");
    await user.click(screen.getByTestId("doi-btn"));

    const card = await screen.findByTestId("job-j1");
    expect(card).toHaveTextContent("done");
  });

  it("polls and displays running status with stage", async () => {
    const user = userEvent.setup();
    mockedApi.ingestDoi.mockResolvedValue({ job_id: "j2" });
    mockedApi.getJob.mockResolvedValue({
      ...baseJob,
      id: "j2",
      status: "running",
      stage: "parsing PDF",
      attempts: 1,
    });

    renderApp();

    await user.type(screen.getByTestId("doi-input"), "10.1/running");
    await user.click(screen.getByTestId("doi-btn"));

    const card = await screen.findByTestId("job-j2");
    expect(card).toHaveTextContent("running");
    expect(card).toHaveTextContent("parsing PDF");
  });

  it("polls and displays dead status with error", async () => {
    const user = userEvent.setup();
    mockedApi.ingestDoi.mockResolvedValue({ job_id: "j3" });
    mockedApi.getJob.mockResolvedValue({
      ...baseJob,
      id: "j3",
      status: "dead",
      error: "Parser failed",
      attempts: 3,
    });

    renderApp();

    await user.type(screen.getByTestId("doi-input"), "10.1/dead");
    await user.click(screen.getByTestId("doi-btn"));

    const card = await screen.findByTestId("job-j3");
    expect(card).toHaveTextContent("dead");
    expect(card).toHaveTextContent("Parser failed");
  });

  it("dismisses a job card", async () => {
    const user = userEvent.setup();
    mockedApi.ingestDoi.mockResolvedValue({ job_id: "j4" });
    mockedApi.getJob.mockResolvedValue({
      ...baseJob,
      id: "j4",
      status: "done",
    });

    renderApp();

    await user.type(screen.getByTestId("doi-input"), "10.1/dismiss");
    await user.click(screen.getByTestId("doi-btn"));

    expect(await screen.findByTestId("job-j4")).toBeInTheDocument();

    await user.click(screen.getByTestId("dismiss-j4"));
    expect(screen.queryByTestId("job-j4")).not.toBeInTheDocument();
  });

  it("shows View button for done jobs with result and calls onViewPaper", async () => {
    const user = userEvent.setup();
    const onViewPaper = vi.fn();
    mockedApi.ingestDoi.mockResolvedValue({ job_id: "j5" });
    mockedApi.getJob.mockResolvedValue({
      ...baseJob,
      id: "j5",
      status: "done",
      result: "paper-uuid-123",
      updated_at: "2025-01-01T01:00:00",
    });

    renderApp({ onViewPaper });

    await user.type(screen.getByTestId("doi-input"), "10.1/view");
    await user.click(screen.getByTestId("doi-btn"));

    const viewBtn = await screen.findByTestId("view-j5");
    expect(viewBtn).toBeInTheDocument();

    await user.click(viewBtn);
    expect(onViewPaper).toHaveBeenCalledWith(
      "paper-uuid-123",
      "http://localhost:8000/papers/paper-uuid-123/file",
    );
  });
});
