import { expect, type Page } from "@playwright/test";
import path from "path";

export const SPC_CSV = path.join(
  __dirname,
  "..",
  "..",
  "resilience_data",
  "cases",
  "spc",
  "imr_in_control_n50.csv",
);

export const MSA_CSV = path.join(
  __dirname,
  "..",
  "..",
  "resilience_data",
  "cases",
  "msa",
  "gage_rr_excellent.csv",
);

export async function expectNoUnreachableApi(page: Page) {
  await expect(page.getByText(/Cannot reach API/i)).toHaveCount(0);
}

export async function expectShellTitle(page: Page, title: string) {
  await expect(page.getByRole("heading", { name: title, level: 1 })).toBeVisible();
}
