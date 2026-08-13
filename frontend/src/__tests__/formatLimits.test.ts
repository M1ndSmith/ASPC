import { describe, expect, it } from "vitest";
import { formatLimits, limitAt, shortId } from "@/lib/format";

describe("formatLimits", () => {
  it("formats scalar limits", () => {
    expect(formatLimits({ center: 10, ucl: 13, lcl: 7 }, 2)).toBe(
      "UCL 13.00 · CL 10.00 · LCL 7.00",
    );
  });

  it("formats variable (array) limits as a range", () => {
    expect(formatLimits({ center: 0.1, ucl: [0.2, 0.3, 0.25], lcl: [0, 0.05, 0.01] }, 2)).toBe(
      "UCL 0.20…0.30 · CL 0.10 · LCL 0.00…0.05",
    );
  });
});

describe("limitAt", () => {
  it("returns scalar or indexed value", () => {
    expect(limitAt(5, 0)).toBe(5);
    expect(limitAt([1, 2, 3], 1)).toBe(2);
    expect(limitAt(undefined, 0)).toBeUndefined();
  });
});

describe("shortId", () => {
  it("truncates long ids", () => {
    expect(shortId("abcdefghijklmnop", 8)).toBe("abcdefgh…");
    expect(shortId("abc", 8)).toBe("abc");
  });
});
