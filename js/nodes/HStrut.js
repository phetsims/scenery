// Copyright 2015-2019, University of Colorado Boulder

/**
 * A Node meant to just take up horizontal space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const scenery = require( 'SCENERY/scenery' );
  const Spacer = require( 'SCENERY/nodes/Spacer' );

  /**
   * Creates a strut with x in the range [0,width] and y=0.
   * @public
   * @constructor
   * @extends Spacer
   *
   * @param {number} width - Width of the strut
   * @param {Object} [options] - Passed to Spacer/Node
   */
  function HStrut( width, options ) {
    Spacer.call( this, width, 0, options );
  }

  scenery.register( 'HStrut', HStrut );

  return inherit( Spacer, HStrut );
} );
