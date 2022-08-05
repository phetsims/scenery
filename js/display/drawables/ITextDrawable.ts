// Copyright 2021-2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Text Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type ITextDrawable = {
  markPaintDirty(): void;
  markDirtyText(): void;
  markDirtyFont(): void;
  markDirtyBounds(): void;
};
export default ITextDrawable // eslint-disable-line
