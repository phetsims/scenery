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
  var TObject = require( 'PHET_IO/types/TObject' );
  var toEventOnEmit = require( 'PHET_IO/events/toEventOnEmit' );

  /**
   * Wrapper type for phet/scenery's TandemButtonListener
   * @param tandemButtonListener
   * @param phetioID
   * @constructor
   */
  function TTandemButtonListener( tandemButtonListener, phetioID ) {
    TObject.call( this, tandemButtonListener, phetioID );
    assertInstanceOf( tandemButtonListener, phet.tandem.TandemButtonListener );

    toEventOnEmit( tandemButtonListener.startedCallbacksForUpEmitter, tandemButtonListener.endedCallbacksForUpEmitter, 'user', phetioID, TTandemButtonListener, 'up' );
    toEventOnEmit( tandemButtonListener.startedCallbacksForOverEmitter, tandemButtonListener.endedCallbacksForOverEmitter, 'user', phetioID, TTandemButtonListener, 'over' );
    toEventOnEmit( tandemButtonListener.startedCallbacksForDownEmitter, tandemButtonListener.endedCallbacksForDownEmitter, 'user', phetioID, TTandemButtonListener, 'down' );
    toEventOnEmit( tandemButtonListener.startedCallbacksForOutEmitter, tandemButtonListener.endedCallbacksForOutEmitter, 'user', phetioID, TTandemButtonListener, 'out' );
    toEventOnEmit( tandemButtonListener.startedCallbacksForFireEmitter, tandemButtonListener.endedCallbacksForFireEmitter, 'user', phetioID, TTandemButtonListener, 'fire' );
  }

  phetioInherit( TObject, 'TTandemButtonListener', TTandemButtonListener, {}, {
    documentation: 'Button listener',
    events: [ 'up', 'over', 'down', 'out', 'fire' ]
  } );

  phetioNamespace.register( 'TTandemButtonListener', TTandemButtonListener );

  return TTandemButtonListener;
} );

