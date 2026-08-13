"use client";

import { Suspense } from "react";
import LivePageInner from "./live-inner";

export default function LivePage() {
  return (
    <Suspense fallback={<p className="text-sm text-aspc-muted">Loading live…</p>}>
      <LivePageInner />
    </Suspense>
  );
}
