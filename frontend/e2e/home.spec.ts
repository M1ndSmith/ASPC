import { test, expect } from "@playwright/test";

/**
 * Smoke: loads / when the Next.js server is up.
 * Skips cleanly if nothing is listening (CI without frontend server).
 */
test("home page loads", async ({ page, baseURL }) => {
  test.skip(!baseURL, "No PLAYWRIGHT_BASE_URL / baseURL configured");

  let reachable = false;
  try {
    const res = await page.request.get(baseURL!, { timeout: 3000 });
    reachable = res.ok() || res.status() < 500;
  } catch {
    reachable = false;
  }
  test.skip(!reachable, "Next.js server not running — skipping smoke e2e");

  await page.goto("/");
  await expect(page.getByRole("heading", { name: /dashboard|aspc|overview/i })).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByRole("link", { name: /live/i })).toBeVisible();
});
