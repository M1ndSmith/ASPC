import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

test("lab lists catalog cases and runs one", async ({ page }) => {
  await page.goto("/lab");
  await expectShellTitle(page, "Resilience Lab");
  await expect(page.getByRole("heading", { name: "Cases" })).toBeVisible();
  await expectNoUnreachableApi(page);

  const runBtn = page.getByRole("button", { name: /^Run$/ }).first();
  await expect(runBtn).toBeVisible({ timeout: 20_000 });
  await runBtn.click();
  await expect(page.getByRole("heading", { name: "Result" })).toBeVisible({ timeout: 60_000 });
  await expectNoUnreachableApi(page);
});
