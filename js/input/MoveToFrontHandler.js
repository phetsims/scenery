//  Copyright 2002-2015, University of Colorado Boulder

/**
 * A handler designed to be used with ChainInputListener that moves a node to the front.
 *
 * TODO: How are developers going to know which files in this directory are handers compatible with chain?
 * TODO: Should we create a separate directory, use a naming convention or make them instances on ChainInputListener?
 *
 * TODO: This file is highly volatile and not ready for public consumption.  The API and implementation are subject to
 * change, and this disclaimer will be removed when the code is ready for review or usage.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   *
   * @constructor
   */
  function MoveToFrontHandler( node ) {
    this.node = node;
  }

  scenery.MoveToFrontHandler = MoveToFrontHandler;

  return inherit( Object, MoveToFrontHandler, {
    start: function( event, trail, chainInputListener ) {
      this.node.moveToFront();
      chainInputListener.nextStart( event, trail );
    }
  } );
} );