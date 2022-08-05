// Copyright 2021-2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Rectangle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IPathDrawable } from '../../imports.js';

type IRectangleDrawable = {
  markDirtyRectangle(): void;
  markDirtyX(): void;
  markDirtyY(): void;
  markDirtyWidth(): void;
  markDirtyHeight(): void;
  markDirtyCornerXRadius(): void;
  markDirtyCornerYRadius(): void;
} & IPathDrawable;
export default IRectangleDrawable // eslint-disable-line
