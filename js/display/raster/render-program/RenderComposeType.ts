// Copyright 2023, University of Colorado Boulder

/**
 * Porter-duff compositing types
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

enum RenderComposeType {
  Over = 0,
  In = 1,
  Out = 2,
  Atop = 3,
  Xor = 4,
  Plus = 5,
  PlusLighter = 6
}

export default RenderComposeType;

scenery.register( 'RenderComposeType', RenderComposeType );
