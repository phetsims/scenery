// Copyright 2022-2025, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Circle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import type TPathDrawable from '../../display/drawables/TPathDrawable.js';

type TCircleDrawable = {
  markDirtyRadius(): void;
} & TPathDrawable;
export default TCircleDrawable;