import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

test("runs list loads and opens a detail page when a run exists", async ({ page }) => {
  await page.goto("/runs");
  await expectShellTitle(page, "Runs");
  await expect(page.getByText(/Audit trail of batch analyses/i)).toBeVisible();
  await expectNoUnreachableApi(page);

  const empty = page.getByText("No analysis runs stored yet.");
  const firstRun = page.locator("table tbody tr td a").first();
  await expect(empty.or(firstRun)).toBeVisible({ timeout: 20_000 });

  if (await firstRun.isVisible()) {
    await firstRun.click();
    await expect(page).toHaveURL(/\/runs\/.+/);
    await expectShellTitle(page, "Run detail");
    await expect(page.getByText(/Run Report|Summary/i).first()).toBeVisible();
    await expectNoUnreachableApi(page);
  }
});
