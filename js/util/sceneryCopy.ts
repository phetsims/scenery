// Copyright 2022, University of Colorado Boulder

/**
 * Copies a generalized Scenery object
 * @deprecated
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, sceneryDeserialize, scenerySerialize } from '../imports.js';

const sceneryCopy = ( value: unknown ): unknown => {
  return sceneryDeserialize( scenerySerialize( value ) );
};

scenery.register( 'sceneryCopy', sceneryCopy );
export default sceneryCopy;
