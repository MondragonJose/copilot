import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Ask from "../screens/Ask";
import { api } from "../api";

vi.mock("../api", () => ({
  api: {
    ask: vi.fn(),
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

const mockPapers = [
  { paperId: "paper-abc", pdfUrl: "http://example.com/a.pdf" },
  { paperId: "paper-xyz", pdfUrl: "http://example.com/b.pdf" },
];

function renderAsk(props?: {
  papers?: typeof mockPapers;
  onNavigateToPaper?: () => void;
}) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <Ask
        papers={props?.papers ?? mockPapers}
        onNavigateToPaper={props?.onNavigateToPaper ?? vi.fn()}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("Ask screen", () => {
  it("renders question input and submit button", () => {
    renderAsk();
    expect(screen.getByTestId("ask-input")).toBeInTheDocument();
    expect(screen.getByTestId("ask-submit")).toBeInTheDocument();
  });

  it("disables submit when input is empty", () => {
    renderAsk();
    expect(screen.getByTestId("ask-submit")).toBeDisabled();
  });

  it("shows paper scope checkboxes", () => {
    renderAsk();
    expect(screen.getByText(/paper-ab/)).toBeInTheDocument();
    expect(screen.getByText(/paper-xy/)).toBeInTheDocument();
  });

  it("shows loading state while asking", async () => {
    const user = userEvent.setup();
    mockedApi.ask.mockImplementation(
      () =>
        new Promise(() => {
          /* never resolves */
        }),
    );

    renderAsk();

    await user.type(screen.getByTestId("ask-input"), "test question");
    await user.click(screen.getByTestId("ask-submit"));

    expect(await screen.findByTestId("ask-loading")).toHaveTextContent(
      "Searching papers and generating answer…",
    );
  });

  describe("answerable question", () => {
    const answerableResponse = {
      question: "test question",
      answer: "The answer is 42.",
      answerable: true,
      claims: [
        {
          text: "The meaning of life is 42",
          chunk_id: "chunk-1",
          quoted_span: "meaning of life is 42",
          supported: true,
          score: 0.95,
          reason: "matches paper content",
          page: 3,
          char_start: 150,
          char_end: 175,
          paper_id: "paper-abc",
        },
        {
          text: "Everything is connected",
          chunk_id: "chunk-2",
          quoted_span: "Everything is connected",
          supported: false,
          score: 0.12,
          reason: "not found in papers",
          page: 7,
          char_start: 50,
          char_end: 75,
          paper_id: "paper-abc",
        },
      ],
    };

    it("renders answer text and claims", async () => {
      const user = userEvent.setup();
      mockedApi.ask.mockResolvedValue(answerableResponse);

      renderAsk();

      await user.type(screen.getByTestId("ask-input"), "test question");
      await user.click(screen.getByTestId("ask-submit"));

      await waitFor(() => {
        expect(screen.getByTestId("answer-card")).toBeInTheDocument();
      });
      expect(screen.getByTestId("answer-text")).toHaveTextContent(
        "The answer is 42.",
      );
      // supported claim
      expect(screen.getByText(/meaning of life is 42/)).toBeInTheDocument();
      // unsupported claim
      expect(screen.getByText(/Everything is connected/)).toBeInTheDocument();
    });

    it("shows citation links that navigate to Reader", async () => {
      const user = userEvent.setup();
      const onNavigateToPaper = vi.fn();
      mockedApi.ask.mockResolvedValue(answerableResponse);

      renderAsk({ onNavigateToPaper });

      await user.type(screen.getByTestId("ask-input"), "test question");
      await user.click(screen.getByTestId("ask-submit"));

      await waitFor(() => {
        expect(screen.getByTestId("citation-link-1")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("citation-link-1"));
      expect(onNavigateToPaper).toHaveBeenCalledWith(
        "paper-abc",
        "http://localhost:8000/papers/paper-abc/file",
        { page: 3, charStart: 150, charEnd: 175 },
      );
    });

    it("shows supported and unsupported labels", async () => {
      const user = userEvent.setup();
      mockedApi.ask.mockResolvedValue(answerableResponse);

      renderAsk();

      await user.type(screen.getByTestId("ask-input"), "test question");
      await user.click(screen.getByTestId("ask-submit"));

      await waitFor(() => {
        expect(screen.getByText(/^Supported/)).toBeInTheDocument();
      });
      expect(screen.getByText(/^Not supported/)).toBeInTheDocument();
    });
  });

  describe("unanswerable question", () => {
    const unanswerableResponse = {
      question: "unknown question",
      answer: null,
      answerable: false,
      claims: [],
    };

    it("renders abstain notice instead of answer", async () => {
      const user = userEvent.setup();
      mockedApi.ask.mockResolvedValue(unanswerableResponse);

      renderAsk();

      await user.type(screen.getByTestId("ask-input"), "unknown question");
      await user.click(screen.getByTestId("ask-submit"));

      await waitFor(() => {
        expect(screen.getByTestId("abstain-notice")).toBeInTheDocument();
      });
      expect(
        screen.getByText(/Cannot answer this question/),
      ).toBeInTheDocument();
    });

    it("does not render answer card", async () => {
      const user = userEvent.setup();
      mockedApi.ask.mockResolvedValue(unanswerableResponse);

      renderAsk();

      await user.type(screen.getByTestId("ask-input"), "unknown question");
      await user.click(screen.getByTestId("ask-submit"));

      await waitFor(() => {
        expect(screen.getByTestId("abstain-notice")).toBeInTheDocument();
      });
      expect(screen.queryByTestId("answer-card")).not.toBeInTheDocument();
    });

    it("shows the user's question in the abstain message", async () => {
      const user = userEvent.setup();
      mockedApi.ask.mockResolvedValue(unanswerableResponse);

      renderAsk();

      await user.type(screen.getByTestId("ask-input"), "unknown question");
      await user.click(screen.getByTestId("ask-submit"));

      await waitFor(() => {
        expect(screen.getByText(/unknown question/)).toBeInTheDocument();
      });
    });
  });

  it("shows error state on API failure", async () => {
    const user = userEvent.setup();
    mockedApi.ask.mockRejectedValue(new Error("API is down"));

    renderAsk();

    await user.type(screen.getByTestId("ask-input"), "fail question");
    await user.click(screen.getByTestId("ask-submit"));

    await waitFor(() => {
      expect(screen.getByTestId("ask-error")).toHaveTextContent("API is down");
    });
  });
});
