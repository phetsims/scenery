// Copyright 2021-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Text Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

type TTextDrawable = {
  markPaintDirty(): void;
  markDirtyText(): void;
  markDirtyFont(): void;
  markDirtyBounds(): void;
};
export default TTextDrawable;