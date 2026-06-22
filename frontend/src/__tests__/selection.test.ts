import { describe, it, expect } from "vitest";
import { selectionToOffsets, getPageNumberFromNode } from "../components/pdf-utils";

describe("selectionToOffsets", () => {
  it("maps a single word from one text item", () => {
    const items = [{ str: "Hello" }];
    expect(selectionToOffsets(items, "Hello")).toEqual({
      charStart: 0,
      charEnd: 5,
    });
  });

  it("maps a word from joined space-separated items", () => {
    const items = [{ str: "Hello" }, { str: "World" }];
    expect(selectionToOffsets(items, "World")).toEqual({
      charStart: 6,
      charEnd: 11,
    });
  });

  it("maps multi-word selection across items", () => {
    const items = [{ str: "The" }, { str: "quick" }, { str: "brown" }, { str: "fox" }];
    expect(selectionToOffsets(items, "quick brown")).toEqual({
      charStart: 4,
      charEnd: 15,
    });
  });

  it("normalizes whitespace in selected text", () => {
    const items = [{ str: "Hello" }, { str: "World" }];
    expect(selectionToOffsets(items, "Hello   World")).toEqual({
      charStart: 0,
      charEnd: 11,
    });
  });

  it("returns null when selected text not found", () => {
    const items = [{ str: "Hello" }, { str: "World" }];
    expect(selectionToOffsets(items, "Goodbye")).toBeNull();
  });

  it("returns null for empty selected text", () => {
    const items = [{ str: "Hello" }];
    expect(selectionToOffsets(items, "")).toBeNull();
  });

  it("returns null for whitespace-only selected text", () => {
    const items = [{ str: "Hello" }];
    expect(selectionToOffsets(items, "   ")).toBeNull();
  });

  it("handles repeated words correctly (finds first occurrence)", () => {
    const items = [{ str: "foo" }, { str: "bar" }, { str: "foo" }];
    expect(selectionToOffsets(items, "foo")).toEqual({
      charStart: 0,
      charEnd: 3,
    });
  });

  it("handles items with internal punctuation", () => {
    const items = [{ str: "Hello," }, { str: "world!" }];
    expect(selectionToOffsets(items, "Hello, world!")).toEqual({
      charStart: 0,
      charEnd: 13,
    });
  });

  it("matches case-sensitively", () => {
    const items = [{ str: "Hello" }, { str: "World" }];
    expect(selectionToOffsets(items, "hello")).toBeNull();
  });
});

describe("getPageNumberFromNode", () => {
  it("returns page number from element with data-page-number", () => {
    const el = document.createElement("div");
    el.setAttribute("data-page-number", "3");
    expect(getPageNumberFromNode(el)).toBe(3);
  });

  it("traverses up to find data-page-number on parent", () => {
    const parent = document.createElement("div");
    parent.setAttribute("data-page-number", "5");
    const child = document.createElement("span");
    parent.appendChild(child);
    expect(getPageNumberFromNode(child)).toBe(5);
  });

  it("returns null when no data-page-number found", () => {
    const el = document.createElement("div");
    expect(getPageNumberFromNode(el)).toBeNull();
  });

  it("returns null for null input", () => {
    expect(getPageNumberFromNode(null)).toBeNull();
  });

  it("handles a text node by going to parentElement", () => {
    const parent = document.createElement("div");
    parent.setAttribute("data-page-number", "7");
    const textNode = document.createTextNode("hello");
    parent.appendChild(textNode);
    expect(getPageNumberFromNode(textNode)).toBe(7);
  });
});
