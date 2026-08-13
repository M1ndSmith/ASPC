import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

test("overview dashboard chrome and action tiles", async ({ page }) => {
  await page.goto("/");
  await expectShellTitle(page, "Dashboard");
  await expectNoUnreachableApi(page);
  await expect(page.getByRole("link", { name: "Onboarding", exact: true }).first()).toBeVisible();
  await expect(page.getByText(/Sample → freeze → go-live|Batch control charts|Phase II streams|Resilience cases/i).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: /recent runs/i })).toBeVisible();
});
