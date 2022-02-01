// Copyright 2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Circle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IPathDrawable } from '../../imports.js';

interface ICircleDrawable extends IPathDrawable {
  markDirtyRadius(): void;
}

export default ICircleDrawable;