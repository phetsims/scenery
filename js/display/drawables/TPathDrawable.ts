// Copyright 2021-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Path Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

type TPathDrawable = {
  markPaintDirty(): void;
  markDirtyShape(): void;
};
export default TPathDrawable;