// Copyright 2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Circle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IPathDrawable } from '../../imports.js';

type TCircleDrawable = {
  markDirtyRadius(): void;
} & IPathDrawable;
export default TCircleDrawable // eslint-disable-line
