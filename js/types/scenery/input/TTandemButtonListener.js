// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'PHET_IO/phetioInherit' );
  var phetioNamespace = require( 'PHET_IO/phetioNamespace' );
  var TObject = require( 'PHET_IO/types/TObject' );
  var toEventOnEmit = require( 'PHET_IO/events/toEventOnEmit' );

  function TTandemButtonListener( tandemButtonListener, phetioID ) {
    TObject.call( this, tandemButtonListener, phetioID );
    assertInstanceOf( tandemButtonListener, phet.tandem.TandemButtonListener );

    toEventOnEmit( tandemButtonListener, 'CallbacksForUpEmitter', 'user', phetioID, TTandemButtonListener, 'up' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForOverEmitter', 'user', phetioID, TTandemButtonListener, 'over' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForDownEmitter', 'user', phetioID, TTandemButtonListener, 'down' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForOutEmitter', 'user', phetioID, TTandemButtonListener, 'out' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForFireEmitter', 'user', phetioID, TTandemButtonListener, 'fire' );
  }

  phetioInherit( TObject, 'TTandemButtonListener', TTandemButtonListener, {}, {
    documentation: 'Button listener',
    events: [ 'up', 'over', 'down', 'out', 'fire' ]
  } );

  phetioNamespace.register( 'TTandemButtonListener', TTandemButtonListener );

  return TTandemButtonListener;
} );

