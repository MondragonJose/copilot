import { useCallback, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Corpus from "./screens/Corpus";
import Reader from "./screens/Reader";
import Ask from "./screens/Ask";
import "./App.css";

const queryClient = new QueryClient();

interface PaperInfo {
  paperId: string;
  pdfUrl: string;
}

export interface CitationInfo {
  page: number;
  charStart: number;
  charEnd: number;
}

type View =
  | { screen: "corpus" }
  | {
      screen: "reader";
      paperId: string;
      pdfUrl: string;
      citation?: CitationInfo;
    }
  | { screen: "ask" };

export default function App() {
  const [view, setView] = useState<View>({ screen: "corpus" });
  const [papers, setPapers] = useState<PaperInfo[]>([]);

  const handleViewPaper = useCallback(
    (paperId: string, pdfUrl: string, citation?: CitationInfo) => {
      setView({ screen: "reader", paperId, pdfUrl, citation });
    },
    [],
  );

  const handlePaperDiscovered = useCallback(
    (paperId: string, pdfUrl: string) => {
      setPapers((prev) => {
        if (prev.some((p) => p.paperId === paperId)) return prev;
        return [...prev, { paperId, pdfUrl }];
      });
    },
    [],
  );

  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <nav
          style={{
            display: "flex",
            gap: 16,
            padding: "8px 16px",
            borderBottom: "1px solid #ddd",
            background: "#f5f5f5",
          }}
        >
          <button
            onClick={() => setView({ screen: "corpus" })}
            style={{
              fontWeight: view.screen === "corpus" ? 700 : 400,
              background: "none",
              border: "none",
              cursor: "pointer",
              fontSize: 15,
            }}
          >
            Corpus
          </button>
          <button
            onClick={() => setView({ screen: "ask" })}
            style={{
              fontWeight: view.screen === "ask" ? 700 : 400,
              background: "none",
              border: "none",
              cursor: "pointer",
              fontSize: 15,
            }}
          >
            Ask
          </button>
        </nav>

        <div style={{ flex: 1, overflow: "hidden" }}>
          {view.screen === "corpus" && (
            <Corpus
              onViewPaper={(paperId, pdfUrl) => {
                handlePaperDiscovered(paperId, pdfUrl);
                handleViewPaper(paperId, pdfUrl);
              }}
            />
          )}
          {view.screen === "reader" && (
            <Reader
              paperId={view.paperId}
              pdfUrl={view.pdfUrl}
              citation={view.citation}
              onBack={() => setView({ screen: "corpus" })}
            />
          )}
          {view.screen === "ask" && (
            <Ask
              papers={papers}
              onNavigateToPaper={handleViewPaper}
            />
          )}
        </div>
      </div>
    </QueryClientProvider>
  );
}
