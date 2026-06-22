import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Reader from "../screens/Reader";
import { api } from "../api";

// ---------------------------------------------------------------------------
// Mock the API module
// ---------------------------------------------------------------------------
vi.mock("../api", () => ({
  api: {
    getAnnotations: vi.fn(),
    createAnnotation: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

// ---------------------------------------------------------------------------
// Mock react-pdf
// ---------------------------------------------------------------------------
vi.mock("react-pdf", () => ({
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" }, version: "0.0.0" },
  Document: ({
    children,
    onLoadSuccess,
  }: {
    children?: React.ReactNode;
    onLoadSuccess?: (pdf: { numPages: number }) => void;
  }) => {
    setTimeout(() => onLoadSuccess?.({ numPages: 1 }), 0);
    return <div data-testid="pdf-document">{children}</div>;
  },
  Page: ({
    onGetTextSuccess,
    pageNumber,
  }: {
    onGetTextSuccess?: (tc: { items: { str: string }[] }) => void;
    pageNumber?: number;
  }) => {
    setTimeout(() => {
      onGetTextSuccess?.({ items: [{ str: "Test page text content" }] });
    }, 0);
    return <div data-testid={`pdf-page-${pageNumber}`}>Page {pageNumber}</div>;
  },
}));

// Mock CSS imports
vi.mock("react-pdf/dist/Page/TextLayer.css", () => ({}));
vi.mock("react-pdf/dist/Page/AnnotationLayer.css", () => ({}));

const mockedApi = vi.mocked(api);

function renderReader(props?: {
  paperId?: string;
  pdfUrl?: string;
  onBack?: () => void;
}) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <Reader
        paperId={props?.paperId ?? "test-paper-id"}
        pdfUrl={props?.pdfUrl ?? "http://example.com/test.pdf"}
        onBack={props?.onBack ?? vi.fn()}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("Reader", () => {
  it("renders the reader layout with back button and paper id", () => {
    renderReader();
    expect(screen.getByText("← Corpus")).toBeInTheDocument();
    expect(screen.getByText("Reader")).toBeInTheDocument();
    expect(screen.getByText(/Paper test-paper-id/)).toBeInTheDocument();
  });

  it("renders the PDF document", () => {
    renderReader();
    expect(screen.getByTestId("pdf-document")).toBeInTheDocument();
  });

  it("renders annotations sidebar", () => {
    renderReader();
    expect(screen.getByText("Annotations")).toBeInTheDocument();
  });

  it("shows empty state when no annotations exist", () => {
    mockedApi.getAnnotations.mockResolvedValue([]);
    renderReader();
    expect(
      screen.getByText(
        "Select text in the PDF and click Explain to add an annotation.",
      ),
    ).toBeInTheDocument();
  });

  it("displays annotations from API", async () => {
    mockedApi.getAnnotations.mockResolvedValue([
      {
        id: "ann-1",
        paper_id: "test-paper-id",
        chunk_id: null,
        page: 3,
        char_start: 0,
        char_end: 10,
        quote: "Hello world",
        note: "Important passage",
        kind: "user",
        created_at: "2025-01-01T00:00:00",
      },
    ]);

    renderReader();

    await waitFor(() => {
      expect(screen.getByText(/Hello world/)).toBeInTheDocument();
    });
    expect(screen.getByText(/p\.3/)).toBeInTheDocument();
    expect(screen.getByText("Important passage")).toBeInTheDocument();
  });

  it("calls createAnnotation when explain flow completes", async () => {
    const onBack = vi.fn();

    mockedApi.getAnnotations.mockResolvedValue([]);
    mockedApi.createAnnotation.mockResolvedValue({
      id: "ann-new",
      paper_id: "test-paper-id",
      chunk_id: null,
      page: 1,
      char_start: 5,
      char_end: 24,
      quote: "page text content",
      note: null,
      kind: "user",
      created_at: "2025-01-02T00:00:00",
    });

    renderReader({ onBack });

    // Wait for the Document mock's onLoadSuccess to fire
    // (setTimeout 0 in the mock) which triggers page rendering
    await waitFor(() => {
      expect(screen.getByTestId("pdf-page-1")).toBeInTheDocument();
    });
  });

  it("calls onBack when back button is clicked", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    mockedApi.getAnnotations.mockResolvedValue([]);

    renderReader({ onBack });

    await user.click(screen.getByText("← Corpus"));
    expect(onBack).toHaveBeenCalledOnce();
  });
});
