import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import { getPageNumberFromNode, selectionToOffsets } from "./pdf-utils";
import SelectionPopover from "./SelectionPopover";

pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;

interface SelectionInfo {
  page: number;
  charStart: number;
  charEnd: number;
  quote: string;
  top: number;
  left: number;
}

interface PdfViewerProps {
  file: string;
  initialPage?: number;
  onExplain: (info: {
    page: number;
    charStart: number;
    charEnd: number;
    quote: string;
  }) => Promise<void>;
}

export default function PdfViewer({ file, initialPage, onExplain }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [selection, setSelection] = useState<SelectionInfo | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const pageTextsRef = useRef<Map<number, readonly { str: string }[]>>(new Map());
  const viewerRef = useRef<HTMLDivElement>(null);
  const initialPageScrolled = useRef(false);

  useEffect(() => {
    pageTextsRef.current.clear();
    setNumPages(null);
    setSelection(null);
    initialPageScrolled.current = false;
  }, [file]);

  useEffect(() => {
    if (
      numPages &&
      initialPage != null &&
      !initialPageScrolled.current
    ) {
      const el = document.querySelector(
        `[data-page-number="${initialPage}"]`,
      );
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        initialPageScrolled.current = true;
      }
    }
  }, [numPages, initialPage]);

  const handleLoadSuccess = useCallback((pdf: { numPages: number }) => {
    setNumPages(pdf.numPages);
  }, []);

  const handleGetTextSuccess = useCallback(
    (pageNumber: number) =>
      (textContent: { items: readonly { str?: string }[] }) => {
        const textItems = textContent.items.filter(
          (item): item is { str: string } =>
            typeof item.str === "string",
        );
        pageTextsRef.current.set(pageNumber, textItems);
      },
    [],
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest(".selection-popover")) return;
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.rangeCount) return;

      const range = sel.getRangeAt(0);
      const pageNumber = getPageNumberFromNode(range.startContainer);
      if (!pageNumber) return;

      const items = pageTextsRef.current.get(pageNumber);
      if (!items) return;

      const selectedText = sel.toString();
      const offsets = selectionToOffsets(items, selectedText);
      if (!offsets) return;

      const rect = range.getBoundingClientRect();
      setSelection({
        page: pageNumber,
        charStart: offsets.charStart,
        charEnd: offsets.charEnd,
        quote: selectedText,
        top: rect.bottom + window.scrollY,
        left: rect.left + rect.width / 2,
      });
    },
    [],
  );

  const handleExplain = useCallback(async () => {
    if (!selection) return;
    setExplainLoading(true);
    try {
      await onExplain({
        page: selection.page,
        charStart: selection.charStart,
        charEnd: selection.charEnd,
        quote: selection.quote,
      });
      setSelection(null);
    } finally {
      setExplainLoading(false);
    }
  }, [selection, onExplain]);

  const pages = [];
  if (numPages) {
    for (let i = 1; i <= numPages; i++) {
      const pageNumber = i;
      pages.push(
        <div key={pageNumber} data-page-number={pageNumber}>
          <Page
            pageNumber={pageNumber}
            onGetTextSuccess={handleGetTextSuccess(pageNumber)}
          />
        </div>,
      );
    }
  }

  return (
    <div
      ref={viewerRef}
      className="pdf-viewer"
      onMouseUp={handleMouseUp}
      style={{ position: "relative", userSelect: "text" }}
    >
      <Document file={file} onLoadSuccess={handleLoadSuccess}>
        {pages}
      </Document>
      {selection && (
        <SelectionPopover
          top={selection.top}
          left={selection.left}
          quote={selection.quote}
          explainLoading={explainLoading}
          onExplain={handleExplain}
          onDismiss={() => setSelection(null)}
        />
      )}
    </div>
  );
}
