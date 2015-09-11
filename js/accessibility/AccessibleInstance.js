// Copyright 2002-2014, University of Colorado Boulder

/**
 * An instance that is synchronously created, for handling accessibility needs.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Events = require( 'AXON/Events' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  scenery.AccessibleInstance = function AccessibleInstance( display, trail ) {
    this.initializeAccessibleInstance( display, trail );
  };
  var AccessibleInstance = scenery.AccessibleInstance;

  inherit( Events, AccessibleInstance, {
    /**
     * @param {DOMElement} [domElement] - If not included here, subtype is responsible for setting it in the constructor.
     */
    initializeAccessibleInstance: function( display, trail ) {
      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      this.display = display;
      this.trail = trail;

      return this;
    }
  } );

  return AccessibleInstance;
} );
