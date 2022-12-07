// Copyright 2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Circle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { TPathDrawable } from '../../imports.js';

type TCircleDrawable = {
  markDirtyRadius(): void;
} & TPathDrawable;
export default TCircleDrawable;
