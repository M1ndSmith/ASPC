"use client";

import { Suspense } from "react";
import LivePageInner from "./live-inner";
import { Spinner } from "@/components/ui";

export default function LivePage() {
  return (
    <Suspense fallback={<Spinner />}>
      <LivePageInner />
    </Suspense>
  );
}
