// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'PHET_IO/phetioInherit' );
  var phetioNamespace = require( 'PHET_IO/phetioNamespace' );
  var TNode = require( 'PHET_IO/types/scenery/nodes/TNode' );
  var toEventOnEmit = require( 'PHET_IO/events/toEventOnEmit' );

  /**
   * Wrapper type for phet/scenery's BarrierRectangle
   * @param barrierRectangle
   * @param phetioID
   * @constructor
   */
  function TBarrierRectangle( barrierRectangle, phetioID ) {
    assertInstanceOf( barrierRectangle, phet.scenery.Rectangle );
    TNode.call( this, barrierRectangle, phetioID );

    toEventOnEmit( barrierRectangle.startedCallbacksForFiredEmitter,
      barrierRectangle.endedCallbacksForFiredEmitter,
      'user',
      phetioID,
      TBarrierRectangle,
      'fired' );
  }

  phetioInherit( TNode, 'TBarrierRectangle', TBarrierRectangle, {}, {
    documentation: 'Shown when a dialog is present, so that clicking on the invisible barrier rectangle will dismiss the dialog',
    events: [ 'fired' ]
  } );

  phetioNamespace.register( 'TBarrierRectangle', TBarrierRectangle );

  return TBarrierRectangle;
} );

