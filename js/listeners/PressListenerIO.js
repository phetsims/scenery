// Copyright 2016, University of Colorado Boulder

/**
 * IO type for PressListener
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var ObjectIO = require( 'ifphetio!PHET_IO/types/ObjectIO' );

  /**
   * @param {PressListener} pressListener
   * @param {string} phetioID
   * @constructor
   */
  function PressListenerIO( pressListener, phetioID ) {
    assert && assertInstanceOf( pressListener, phet.scenery.PressListener );
    ObjectIO.call( this, pressListener, phetioID );
  }

  phetioInherit( ObjectIO, 'PressListenerIO', PressListenerIO, {}, {
    documentation: 'Input listener for something that can be pressed.',
    events: [ 'press', 'release' ]
  } );

  scenery.register( 'PressListenerIO', PressListenerIO );

  return PressListenerIO;
} );

