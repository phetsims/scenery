// Copyright 2016, University of Colorado Boulder

/**
 * IO type for SCENERY ButtonListener (not SUN ButtonListener)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var ObjectIO = require( 'ifphetio!PHET_IO/types/ObjectIO' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );

  /**
   * @param {ButtonListener} buttonListener
   * @param {string} phetioID
   * @constructor
   */
  function ButtonListenerIO( buttonListener, phetioID ) {
    assert && assertInstanceOf( buttonListener, phet.scenery.ButtonListener );
    ObjectIO.call( this, buttonListener, phetioID );
  }

  phetioInherit( ObjectIO, 'ButtonListenerIO', ButtonListenerIO, {}, {
    documentation: 'Button listener',
    events: [ 'up', 'over', 'down', 'out', 'fire' ]
  } );

  scenery.register( 'ButtonListenerIO', ButtonListenerIO );

  return ButtonListenerIO;
} );