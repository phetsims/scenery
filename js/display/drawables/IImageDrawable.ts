// Copyright 2021, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for an Image Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

interface IImageDrawable {
  markPaintDirty(): void;
  markDirtyImage(): void;
  markDirtyImageOpacity(): void;
  markDirtyMipmap(): void;
}

export default IImageDrawable;