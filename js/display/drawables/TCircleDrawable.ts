// Copyright 2022-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Circle Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import type TPathDrawable from '../../display/drawables/TPathDrawable.js';

type TCircleDrawable = {
  markDirtyRadius(): void;
} & TPathDrawable;
export default TCircleDrawable;