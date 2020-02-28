// Copyright 2015-2020, University of Colorado Boulder

/**
 * A Node meant to just take up vertical space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inherit from '../../../phet-core/js/inherit.js';
import scenery from '../scenery.js';
import Spacer from './Spacer.js';

/**
 * Creates a strut with x=0 and y in the range [0,height].
 * @public
 * @constructor
 * @extends Spacer
 *
 * @param {number} height - Height of the strut
 * @param {Object} [options] - Passed to Spacer/Node
 */
function VStrut( height, options ) {
  Spacer.call( this, 0, height, options );
}

scenery.register( 'VStrut', VStrut );

inherit( Spacer, VStrut );
export default VStrut;