import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle, MSA_CSV } from "./helpers";

test("msa gage R&R upload produces a batch result", async ({ page }) => {
  await page.goto("/msa");
  await expectShellTitle(page, "MSA");
  await expect(page.getByRole("heading", { name: "Study upload" })).toBeVisible();
  await expectNoUnreachableApi(page);

  await page.locator("#msa_file").setInputFiles(MSA_CSV);
  await page.getByLabel("Study type").selectOption("gage_rr");
  await page.getByRole("button", { name: /run msa/i }).click();
  await expect(page.getByRole("heading", { name: "Batch MSA result" })).toBeVisible({
    timeout: 45_000,
  });
  await expect(page.getByText("Run", { exact: true })).toBeVisible();
  await expectNoUnreachableApi(page);
});
