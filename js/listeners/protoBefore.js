// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO: doc      AND MOVE TO PHET_CORE?
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  function protoBefore( prototype, eventName, method ) {
    var original = prototype[ eventName ];

    if ( original ) {
      prototype[ eventName ] = function() {
        var args = Array.prototype.slice.call( arguments );
        method.apply( this, args );
        original.apply( this, args );
      };
    }
    else {
      prototype[ eventName ] = method;
    }
  }

  scenery.register( 'protoBefore', protoBefore );

  return protoBefore;
} );
