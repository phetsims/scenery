// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * TODO: doc
   */
  function HandleDownlistener() {
  }

  scenery.register( 'HandleDownlistener', HandleDownlistener );

  inherit( Object, HandleDownlistener, {
    down: function( event ) {
      event.handle();
    }
  } );

  return HandleDownlistener;
} );
