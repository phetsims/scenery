// Copyright 2021-2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for an Image Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type IImageDrawable = {
  markPaintDirty(): void;
  markDirtyImage(): void;
  markDirtyImageOpacity(): void;
  markDirtyMipmap(): void;
};
export default IImageDrawable // eslint-disable-line
