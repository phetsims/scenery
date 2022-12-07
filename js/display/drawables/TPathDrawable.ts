// Copyright 2021-2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Path Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TPathDrawable = {
  markPaintDirty(): void;
  markDirtyShape(): void;
};
export default TPathDrawable;
