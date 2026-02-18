// Copyright 2021-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for an Image Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

type TImageDrawable = {
  markPaintDirty(): void;
  markDirtyImage(): void;
  markDirtyImageOpacity(): void;
  markDirtyMipmap(): void;
};
export default TImageDrawable;