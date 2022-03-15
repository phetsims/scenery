// Copyright 2021-2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Path Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export default interface IPathDrawable {
  markPaintDirty(): void;
  markDirtyShape(): void;
} // eslint-disable-line
