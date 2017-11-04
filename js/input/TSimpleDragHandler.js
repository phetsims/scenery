// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );
  var TObject = require( 'ifphetio!PHET_IO/types/TObject' );

  /**
   * Wrapper type for phet/tandem's SimpleDragHandler class.
   * @param {SimpleDragHandler} simpleDragHandler
   * @param {string} phetioID
   * @constructor
   */
  function TSimpleDragHandler( simpleDragHandler, phetioID ) {
    assert && assertInstanceOf( simpleDragHandler, phet.scenery.SimpleDragHandler );
    TObject.call( this, simpleDragHandler, phetioID );
  }

  phetioInherit( TObject, 'TSimpleDragHandler', TSimpleDragHandler, {}, {
    documentation: 'Drag listener for objects that can be dragged by the user.',
    events: [ 'dragStarted', 'dragged', 'dragEnded' ]
  } );

  scenery.register( 'TSimpleDragHandler', TSimpleDragHandler );

  return TSimpleDragHandler;
} );

