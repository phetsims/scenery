// Copyright 2021-2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Paintable Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TPaintableDrawable = {
  markDirtyFill(): void;
  markDirtyStroke(): void;
  markDirtyLineWidth(): void;
  markDirtyLineOptions(): void;
  markDirtyCachedPaints(): void;
};
export default TPaintableDrawable;
