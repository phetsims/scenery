// Copyright 2021, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Text Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export default interface ITextDrawable {
  markPaintDirty(): void;
  markDirtyText(): void;
  markDirtyFont(): void;
  markDirtyBounds(): void;
} // eslint-disable-line
