// Copyright 2021, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Path Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IPathDrawable } from '../../imports.js';

interface IRectangleDrawable extends IPathDrawable {
  markDirtyRectangle(): void;
  markDirtyX(): void;
  markDirtyY(): void;
  markDirtyWidth(): void;
  markDirtyHeight(): void;
  markDirtyCornerXRadius(): void;
  markDirtyCornerYRadius(): void;
}

export default IRectangleDrawable;