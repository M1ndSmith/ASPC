import { test as setup, expect } from "@playwright/test";
import path from "path";
import fs from "fs";

const authFile = path.join(__dirname, ".auth/user.json");

setup("authenticate", async ({ page, baseURL }) => {
  const username = process.env.E2E_USERNAME || "admin";
  const password = process.env.E2E_PASSWORD || "change-me-admin-password";

  let reachable = false;
  try {
    const res = await page.request.get(baseURL || "http://localhost:3000/login", { timeout: 5000 });
    reachable = res.ok() || res.status() < 500;
  } catch {
    reachable = false;
  }
  if (!reachable) {
    throw new Error(
      `UI not reachable at ${baseURL}. Start Compose: docker compose -f deploy/compose/docker-compose.yml --env-file deploy/compose/.env up -d`,
    );
  }

  fs.mkdirSync(path.dirname(authFile), { recursive: true });

  await page.goto("/login");
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page).toHaveURL(/\/$/, { timeout: 20_000 });
  await expect(page.getByRole("heading", { name: "Dashboard", level: 1 })).toBeVisible();
  await page.context().storageState({ path: authFile });
});
