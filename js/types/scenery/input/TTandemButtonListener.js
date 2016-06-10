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
  var TObject = require( 'PHET_IO/api/TObject' );
  var toEventOnEmit = require( 'PHET_IO/events/toEventOnEmit' );

  var TTandemButtonListener = phetioInherit( TObject, 'TTandemButtonListener', function( tandemButtonListener, phetioID ) {
    TObject.call( this, tandemButtonListener, phetioID );
    assertInstanceOf( tandemButtonListener, phet.tandem.TandemButtonListener );

    toEventOnEmit( tandemButtonListener, 'CallbacksForUpEmitter', 'user', phetioID, 'up' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForOverEmitter', 'user', phetioID, 'over' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForDownEmitter', 'user', phetioID, 'down' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForOutEmitter', 'user', phetioID, 'out' );
    toEventOnEmit( tandemButtonListener, 'CallbacksForFireEmitter', 'user', phetioID, 'fire' );
  }, {}, {
    documentation: 'Button listener',
    events: [ 'up', 'over', 'down', 'out', 'fire' ]
  } );

  phetioNamespace.register( 'TTandemButtonListener', TTandemButtonListener );

  return TTandemButtonListener;
} );

