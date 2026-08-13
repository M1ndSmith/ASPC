import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle, SPC_CSV } from "./helpers";

test("capability study runs against uploaded measurements", async ({ page }) => {
  await page.goto("/capability");
  await expectShellTitle(page, "Capability");
  await expect(page.getByRole("heading", { name: "Study setup" })).toBeVisible();
  await expectNoUnreachableApi(page);

  await page.locator("#cap_file").setInputFiles(SPC_CSV);
  await page.getByLabel("USL").fill("110");
  await page.getByLabel("LSL").fill("90");
  await page.getByRole("button", { name: /run capability/i }).click();
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("Run", { exact: true })).toBeVisible();
  await expectNoUnreachableApi(page);
});
