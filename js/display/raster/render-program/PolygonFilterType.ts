// Copyright 2023, University of Colorado Boulder

/**
 * Controls how polygons get filtered when output
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

enum PolygonFilterType {
  Box = 0,
  Bilinear = 1,
  MitchellNetravali = 2
}

export default PolygonFilterType;

scenery.register( 'PolygonFilterType', PolygonFilterType );
