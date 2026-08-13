import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle, SPC_CSV } from "./helpers";

test("analyze uploads I-MR CSV and shows a result", async ({ page }) => {
  await page.goto("/analyze");
  await expectShellTitle(page, "Analyze");
  await expect(page.getByRole("heading", { name: "Upload" })).toBeVisible();
  await expectNoUnreachableApi(page);

  await page.locator("#file").setInputFiles(SPC_CSV);
  await page.getByRole("button", { name: /run analysis/i }).click();
  await expect(page.getByRole("heading", { name: "Result" })).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText(/Run ID/)).toBeVisible();
  await expectNoUnreachableApi(page);
});
