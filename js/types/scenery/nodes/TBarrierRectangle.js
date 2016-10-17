// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare
 */
define( function( require ) {
  'use strict';

  // modules
  var phetioInherit = require( 'PHET_IO/phetioInherit' );
  var phetioNamespace = require( 'PHET_IO/phetioNamespace' );
  var TNode = require( 'PHET_IO/types/scenery/nodes/TNode' );
  var toEventOnStatic = require( 'PHET_IO/events/toEventOnStatic' );

  function TBarrierRectangle( barrierRectangle, phetioID ) {
    TNode.call( this, barrierRectangle, phetioID );

    toEventOnStatic( barrierRectangle, 'CallbacksForFired', 'user', phetioID, TBarrierRectangle, 'fired' );
  }

  phetioInherit( TNode, 'TBarrierRectangle', TBarrierRectangle, {}, {
    documentation: 'Shown when a dialog is present, so that clicking on the invisible barrier rectangle will dismiss the dialog',
    events: [ 'fired' ]
  } );

  phetioNamespace.register( 'TBarrierRectangle', TBarrierRectangle );

  return TBarrierRectangle;
} );

