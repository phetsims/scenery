// Copyright 2021-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Paintable Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

type TPaintableDrawable = {
  markDirtyFill(): void;
  markDirtyStroke(): void;
  markDirtyLineWidth(): void;
  markDirtyLineOptions(): void;
  markDirtyCachedPaints(): void;
};
export default TPaintableDrawable;