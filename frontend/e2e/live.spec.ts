import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

test("live page shows stream controls without requiring Kafka traffic", async ({ page }) => {
  await page.goto("/live");
  await expectShellTitle(page, "Live");
  await expect(page.getByText(/Phase II stream against frozen limits/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Stream" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Connect", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Disconnect" })).toBeVisible();
  await expect(page.getByText("Disconnected")).toBeVisible();
  await expectNoUnreachableApi(page);
});
