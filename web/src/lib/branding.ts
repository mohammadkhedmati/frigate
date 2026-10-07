/**
 * Product branding for this fork.
 *
 * Upstream's product name appears ~5,000 times across web/public/locales/**, in
 * 51 languages. Rewriting it at render time (see the "brand" postProcessor in
 * web/src/utils/i18n.ts) keeps those locale bundles byte-identical to upstream,
 * so pulling from blakeblackshear/frigate never conflicts on translation files.
 */

export const APP_NAME = "MehrsunAI";

/**
 * Matches upstream's product name, excluding names we do not own.
 *
 * "Frigate+" and "Frigate Plus" name Frigate, Inc.'s paid cloud service, so they
 * must keep their original name. Requiring a capital F also protects lowercase
 * identifiers that happen to appear in translated copy: logger names such as
 * "frigate.record", config values, docs URLs, and the /media/frigate path.
 */
const BRAND_PATTERN = /Frigate(?!\+|\s+Plus\b)/g;

/** Replace upstream's product name with ours in a user-facing string. */
export function applyBrand(value: string): string {
  return value.replace(BRAND_PATTERN, APP_NAME);
}
