// Copyright 2016, University of Colorado Boulder

/**
 * Wrapper type for SCENERY ButtonListener (not SUN ButtonListener)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var TObject = require( 'ifphetio!PHET_IO/types/TObject' );

  /**
   * @param {ButtonListener} buttonListener
   * @param {string} phetioID
   * @constructor
   */
  function TButtonListener( buttonListener, phetioID ) {
    assert && assertInstanceOf( buttonListener, phet.scenery.ButtonListener );
    TObject.call( this, buttonListener, phetioID );
  }

  phetioInherit( TObject, 'TButtonListener', TButtonListener, {}, {
    documentation: 'Button listener',
    events: [ 'up', 'over', 'down', 'out', 'fire' ]
  } );

  scenery.register( 'TButtonListener', TButtonListener );

  return TButtonListener;
} );