// Copyright 2021-2025, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Rectangle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import type TPathDrawable from '../../display/drawables/TPathDrawable.js';

type TRectangleDrawable = {
  markDirtyRectangle(): void;
  markDirtyX(): void;
  markDirtyY(): void;
  markDirtyWidth(): void;
  markDirtyHeight(): void;
  markDirtyCornerXRadius(): void;
  markDirtyCornerYRadius(): void;
} & TPathDrawable;
export default TRectangleDrawable;