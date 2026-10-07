import { describe, expect, it } from "vitest";
import { APP_NAME, applyBrand } from "./branding";

describe("applyBrand", () => {
  it("renames the product in user-facing copy", () => {
    expect(applyBrand("Restart Frigate")).toBe(`Restart ${APP_NAME}`);
    expect(applyBrand("Frigate is Restarting")).toBe(
      `${APP_NAME} is Restarting`,
    );
    expect(applyBrand("Live - Frigate")).toBe(`Live - ${APP_NAME}`);
  });

  it("renames every occurrence in a string", () => {
    expect(applyBrand("Frigate stops, then Frigate starts")).toBe(
      `${APP_NAME} stops, then ${APP_NAME} starts`,
    );
  });

  it("leaves the Frigate+ service name alone", () => {
    expect(applyBrand("Submit to Frigate+")).toBe("Submit to Frigate+");
    expect(applyBrand("Frigate+ Settings - Frigate")).toBe(
      `Frigate+ Settings - ${APP_NAME}`,
    );
    expect(applyBrand("Confirm this label for Frigate Plus")).toBe(
      "Confirm this label for Frigate Plus",
    );
  });

  it("leaves lowercase identifiers and paths alone", () => {
    expect(applyBrand("e.g., frigate.record")).toBe("e.g., frigate.record");
    expect(applyBrand("/media/frigate/recordings")).toBe(
      "/media/frigate/recordings",
    );
    expect(applyBrand("https://docs.frigate.video/")).toBe(
      "https://docs.frigate.video/",
    );
  });
});
