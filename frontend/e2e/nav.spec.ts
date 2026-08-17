import { test, expect } from "@playwright/test";
import { expectNoUnreachableApi, expectShellTitle } from "./helpers";

const PAGES = [
  { name: "Overview", path: "/", title: "Overview" },
  { name: "Onboarding", path: "/onboarding", title: "Onboarding" },
  { name: "Live", path: "/live", title: "Live" },
  { name: "Analyze", path: "/analyze", title: "Analyze" },
  { name: "Capability", path: "/capability", title: "Capability" },
  { name: "MSA", path: "/msa", title: "MSA" },
  { name: "Runs", path: "/runs", title: "Runs" },
  { name: "Lab", path: "/lab", title: "Resilience Lab" },
] as const;

test("sidebar navigates every operator page", async ({ page }) => {
  await page.goto("/");
  await expectShellTitle(page, "Overview");

  for (const item of PAGES) {
    await page.locator("aside nav").getByRole("link", { name: item.name, exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`${item.path.replace("/", "\\/")}$`));
    await expectShellTitle(page, item.title);
    await expectNoUnreachableApi(page);
  }
});
