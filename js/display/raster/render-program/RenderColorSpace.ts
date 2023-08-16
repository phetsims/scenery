// Copyright 2023, University of Colorado Boulder

/**
 * Represents a colorspace (typically for handling blending)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

enum RenderColorSpace {
  LinearUnpremultipliedSRGB = 0,
  SRGB = 1,
  Oklab = 2
}

export default RenderColorSpace;

scenery.register( 'RenderColorSpace', RenderColorSpace );
