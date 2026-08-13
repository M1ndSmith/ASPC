import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

test("onboarding loads sample and establishes Phase I", async ({ page }) => {
  await page.goto("/onboarding");
  await expectShellTitle(page, "Onboarding");
  await expect(page.getByRole("heading", { name: /Step 1 of 4/i })).toBeVisible();
  await expect(page.getByText("Sample + establish")).toBeVisible();
  await expectNoUnreachableApi(page);

  await page.getByRole("button", { name: /load sample & establish/i }).click();
  await expect(page.getByRole("heading", { name: /Step 2 of 4/i })).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText(/Run /)).toBeVisible();
  await expectNoUnreachableApi(page);
});
