// vite.config.ts declares this as vitest's setupFiles entry, but the file is
// absent upstream, which makes every `npm run test` invocation fail before it
// collects a single test. Restored here so the suite can run.
import "@testing-library/jest-dom/vitest";
