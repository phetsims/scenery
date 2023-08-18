// Copyright 2023, University of Colorado Boulder

/**
 * Controls how images get resampled when output
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

enum RenderResampleType {
  NearestNeighbor = 0,
  AnalyticMitchellNetravali = 1,
  Bilinear = 2,
  MitchellNetravali = 3,
  AnalyticBox = 4,
  AnalyticBilinear = 5
}

export default RenderResampleType;

scenery.register( 'RenderResampleType', RenderResampleType );
