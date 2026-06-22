export function selectionToOffsets(
  items: readonly { str: string }[],
  selectedText: string,
): { charStart: number; charEnd: number } | null {
  const fullText = items.map((item) => item.str).join(" ");
  const normalize = (s: string) => s.replace(/\s+/g, " ").trim();
  const normFull = normalize(fullText);
  const normSelected = normalize(selectedText);
  if (!normSelected) return null;
  const start = normFull.indexOf(normSelected);
  if (start === -1) return null;
  return { charStart: start, charEnd: start + normSelected.length };
}

export function getPageNumberFromNode(
  node: Node | null,
): number | null {
  let el = node instanceof Element ? node : node?.parentElement ?? null;
  while (el) {
    const val = el.getAttribute("data-page-number");
    if (val !== null) return parseInt(val, 10);
    el = el.parentElement;
  }
  return null;
}
