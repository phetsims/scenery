// Copyright 2022, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Circle Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IPathDrawable } from '../../imports.js';

export default interface ICircleDrawable extends IPathDrawable {
  markDirtyRadius(): void;
} // eslint-disable-line
