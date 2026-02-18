// Copyright 2021-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Rectangle Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
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