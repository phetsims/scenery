// Copyright 2023, University of Colorado Boulder

/**
 * Represents a point of an intersection (with rational t and point) along a segment.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, BigRationalVector2, scenery } from '../../../imports.js';

export default class RationalIntersection {
  public constructor( public readonly t: BigRational, public readonly point: BigRationalVector2 ) {}
}

scenery.register( 'RationalIntersection', RationalIntersection );
