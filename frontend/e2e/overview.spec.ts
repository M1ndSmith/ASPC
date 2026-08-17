import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

test("overview dashboard chrome and action tiles", async ({ page }) => {
  await page.goto("/");
  await expectShellTitle(page, "Overview");
  await expectNoUnreachableApi(page);
  await expect(page.getByText("Operator Console")).toHaveCount(0);
  await expect(page.getByText("Online").or(page.getByText("Down")).first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Onboarding", exact: true }).first()).toBeVisible();
  await expect(
    page.getByText(/Use a sample file|Upload a spreadsheet|Watch measurements as they arrive|Try known good and bad examples/i).first(),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: /recent runs/i })).toBeVisible();
});

test("overview hover tips explain nav and account", async ({ page }) => {
  await page.goto("/");
  await expectShellTitle(page, "Overview");

  await page.locator("aside nav").getByRole("link", { name: "Overview", exact: true }).hover();
  await expect(page.getByRole("tooltip", { name: /Home: is the system up/i })).toBeVisible();

  await page.getByRole("button", { name: /account menu/i }).hover();
  await expect(page.getByRole("tooltip", { name: /Your account/i })).toBeVisible();
});
