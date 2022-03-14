// Copyright 2021, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Paintable Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export default interface IPaintableDrawable {
  markDirtyFill(): void;
  markDirtyStroke(): void;
  markDirtyLineWidth(): void;
  markDirtyLineOptions(): void;
  markDirtyCachedPaints(): void;
} // eslint-disable-line
