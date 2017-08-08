// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );
  var TObject = require( 'ifphetio!PHET_IO/types/TObject' );
  var toEventOnEmit = require( 'ifphetio!PHET_IO/toEventOnEmit' );

  /**
   * Wrapper type for phet/tandem's SimpleDragHandler class.
   * @param simpleDragHandler
   * @param phetioID
   * @constructor
   */
  function TSimpleDragHandler( simpleDragHandler, phetioID ) {
    TObject.call( this, simpleDragHandler, phetioID );
    assertInstanceOf( simpleDragHandler, phet.scenery.SimpleDragHandler );

    var toXY = function( x, y ) { return { x: x, y: y }; };
    toEventOnEmit( simpleDragHandler.startedCallbacksForDragStartedEmitter, simpleDragHandler.endedCallbacksForDragStartedEmitter, 'user', phetioID, this.constructor, 'dragStarted', toXY );
    toEventOnEmit( simpleDragHandler.startedCallbacksForDraggedEmitter, simpleDragHandler.endedCallbacksForDraggedEmitter, 'user', phetioID, this.constructor, 'dragged', toXY );
    toEventOnEmit( simpleDragHandler.startedCallbacksForDragEndedEmitter, simpleDragHandler.endedCallbacksForDragEndedEmitter, 'user', phetioID, this.constructor, 'dragEnded' );
  }

  phetioInherit( TObject, 'TSimpleDragHandler', TSimpleDragHandler, {}, {
    documentation: 'Drag listener for objects that can be dragged by the user.',
    events: [ 'dragStarted', 'dragged', 'dragEnded' ]
  } );

  scenery.register( 'TSimpleDragHandler', TSimpleDragHandler );

  return TSimpleDragHandler;
} );

